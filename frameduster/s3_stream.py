import glob
import os.path
import shutil
import threading
import io
import struct
import hashlib
from tarfile import BLOCKSIZE
from loguru import logger
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pymongo.errors

from frameduster.config import config
from frameduster.s3 import _s3_client, _mongo_cache, _stream_cache, _stream_complete_cache
from frameduster.mongodb import mongo_connection, _write_document


class ConnectedTarBuffer:
    # Various constants from tarfile.py that we need
    NUL = b"\0"
    BLOCKSIZE = 512
    RECORDSIZE = BLOCKSIZE * 20
    MIN_SIZE = 5 * 1024 * 1024

    def stn(self, s, length, encoding, errors):
        s = s.encode(encoding, errors)
        return s[:length] + (length - len(s)) * self.NUL

    def itn(self, n, digits=8):
        n = int(n)
        if 0 <= n < 8 ** (digits - 1):
            s = bytes("%0*o" % (digits - 1, n), "ascii") + self.NUL
        elif -256 ** (digits - 1) <= n < 256 ** (digits - 1):
            if n >= 0:
                s = bytearray([0o200])
            else:
                s = bytearray([0o377])
                n = 256 ** digits + n

            for i in range(digits - 1):
                s.insert(1, n & 0o377)
                n >>= 8
        else:
            raise ValueError("overflow in number field")

        return s

    def calc_chksums(self, buf):
        unsigned_chksum = 256 + sum(struct.unpack_from("148B8x356B", buf))
        signed_chksum = 256 + sum(struct.unpack_from("148b8x356b", buf))
        return unsigned_chksum, signed_chksum

    def __init__(self, dest, upload_id=None):

        self.buf = io.BytesIO()
        self.buf_size = 0

        self.upload_id = upload_id
        self.key = dest
        self.part_number = 1

        # if no multipart upload is provided, open one
        if not self.upload_id:
            self.open()

    def add_file_record(self, filename, filesize):
        REGTYPE = b"0"  # regular file
        encoding = "utf-8"
        LENGTH_NAME = 100
        GNU_MAGIC = b"ustar  \0"  # magic gnu tar string
        errors = "surrogateescape"

        # Copied from TarInfo.tobuf()
        tarinfo = {
            "name": filename,
            "mode": 0o644,
            "uid": 0,
            "gid": 0,
            "size": filesize,
            "mtime": 0,
            "chksum": 0,
            "type": REGTYPE,
            "linkname": "",
            "uname": "",
            "gname": "",
            "devmajor": 0,
            "devminor": 0,
            "magic": GNU_MAGIC
        }

        if len(tarinfo["name"].encode(encoding, errors)) > LENGTH_NAME:
            raise Exception("Filename is too long for tar file header.")

        devmajor = self.stn("", 8, encoding, errors)
        devminor = self.stn("", 8, encoding, errors)

        parts = [
            self.stn(tarinfo.get("name", ""), 100, encoding, errors),
            self.itn(tarinfo.get("mode", 0) & 0o7777, 8),
            self.itn(tarinfo.get("uid", 0), 8),
            self.itn(tarinfo.get("gid", 0), 8),
            self.itn(tarinfo.get("size", 0), 12),
            self.itn(tarinfo.get("mtime", 0), 12),
            b"        ",  # checksum field
            tarinfo.get("type", REGTYPE),
            self.stn(tarinfo.get("linkname", ""), 100, encoding, errors),
            tarinfo.get("magic", GNU_MAGIC),
            self.stn(tarinfo.get("uname", ""), 32, encoding, errors),
            self.stn(tarinfo.get("gname", ""), 32, encoding, errors),
            devmajor,
            devminor,
            self.stn(tarinfo.get("prefix", ""), 155, encoding, errors)
        ]
        buf = struct.pack("%ds" % BLOCKSIZE, b"".join(parts))
        chksum = self.calc_chksums(buf[-BLOCKSIZE:])[0]
        buf = buf[:-364] + bytes("%06o\0" % chksum, "ascii") + buf[-357:]
        self.write(buf)

    def complete_file_record(self, size):
        # write the end-of-file marker
        blocks, remainder = divmod(size, BLOCKSIZE)
        if remainder > 0:
            self.write(self.NUL * (BLOCKSIZE - remainder))

    def add_file(self, file_bytes, name, size):
        # record
        self.add_file_record(name, size)

        # content
        self.write(file_bytes)

        # complete
        self.complete_file_record(size)

    def write(self, data):
        self.buf.write(data)
        self.buf_size += len(data)

    def send_buf(self, part_number=None):
        if not part_number:
            part_number = self.part_number
        etag = _s3_client.upload_part(Body=self.buf.getbuffer().tobytes(),
                                      Bucket=config['s3']['bucket'],
                                      UploadId=self.upload_id,
                                      Key=self.key,
                                      PartNumber=part_number,
                                      ContentLength=self.buf_size)['ETag']
        part_pk = {'ETag': etag, 'PartNumber': part_number}
        self.part_number += 1

        return part_pk

    def clear(self):
        self.buf = io.BytesIO()
        self.buf_size = 0

    def open(self):
        self.upload_id = _s3_client.create_multipart_upload(Bucket=config['s3']['bucket'],
                                                            Key=self.key)['UploadId']
        self.part_number = 1

    def close(self, parts):
        try:
            _s3_client.complete_multipart_upload(Bucket=config['s3']['bucket'],
                                                 UploadId=self.upload_id,
                                                 Key=self.key,
                                                 MultipartUpload={'Parts': parts})
        except Exception as e:
            print("Error while closing file : ", self.key, " : ", e)

    def abort(self):
        _s3_client.abort_multipart_upload(Bucket=config['s3']['bucket'],
                                          UploadId=self.upload_id,
                                          Key=self.key)

    def complete_tar_file(self):
        self.write(self.NUL * (BLOCKSIZE * 2))
        self.buf_size += (BLOCKSIZE * 2)
        blocks, remainder = divmod(self.buf_size, self.RECORDSIZE)
        if remainder > 0:
            self.write(self.NUL * (BLOCKSIZE * 2))


_s3_stream = {}
_s3_uploading_count = {}
_s3_done_event = {}
_s3_stream_size = {}
_s3_complete_lock = {}
_mongo_buffer = {}
_completing = []

# Thread safety
_s3_stream_lock = {}
_s3_thread_lock = threading.Lock()


def _complete_upload(upload_dir):
    upload_file_path = os.path.join(upload_dir, 'upload.json')

    # Extract every information from path
    stub, _ = os.path.split(upload_dir)
    stub, dataset = os.path.split(stub)
    _, function_id = os.path.split(stub)

    # Load stream info
    stream_info = json.load(open(upload_file_path, 'r'))
    key, upload_id = stream_info
    tar_name = os.path.split(key)[-1]

    # Gather parts
    parts = []
    for part_path in glob.iglob(f'{upload_dir}/parts/*'):
        parts.append(json.load(open(part_path, 'r')))

    # Close the multipart upload
    ConnectedTarBuffer(key, upload_id).close(parts)

    # Send all mongo documents
    file_name_field = str(os.path.split(key)[1]).split('!')[0]
    pbar = tqdm(glob.glob(f'{upload_dir}/mongo/*'))
    pbar.set_description_str(f"Updating database for {key}...")
    with ThreadPoolExecutor(max_workers=250) as workers:
        for doc_path in pbar:
            doc = json.load(open(doc_path, 'r'))
            workers.submit(_write_document, doc)

    # Update the s3 collections
    try:
        mongo_connection[config['project_name'] + "-pipeline-s3"].insert_one({
            "_id": tar_name,
            "dataset": dataset,
            'field': file_name_field
        })
        mongo_connection[dataset + "-dataset-s3"].insert_one({
            "_id": tar_name,
            "dataset": dataset,
            'field': file_name_field
        })
    except pymongo.errors.DuplicateKeyError:
        pass

    logger.success(f"Remote file {key} completed")


def _flush_mongo_buffer(function_id, dataset, part_number, mongo_buf=None):
    fid_dataset = f'{function_id}/{dataset}'

    if not mongo_buf:
        mongo_buf = _mongo_buffer[fid_dataset]

    # **** Save the mongo documents
    for i, mongo_doc in enumerate(mongo_buf):
        mongo_full_path = f'{_mongo_cache}/{function_id}/{dataset}'
        os.makedirs(mongo_full_path, exist_ok=True)

        mongo_file_name = f'mongo-{part_number}-{i}.json'
        mongo_file_path = os.path.join(mongo_full_path, mongo_file_name)

        json.dump(mongo_doc, open(mongo_file_path, 'w'))
    # ****


def _recover_s3_stream(function_id):
    for upload_file in glob.iglob(f'{_stream_cache}/{function_id}/*.json'):
        remote_key, upload_id = json.load(open(upload_file, 'r'))

        dataset_name = os.path.splitext(os.path.basename(upload_file))[0]
        fid_dataset = f'{function_id}/{dataset_name}'

        part_dir = f'{_stream_cache}/{function_id}/{dataset_name}'
        if not os.path.isdir(part_dir):
            os.remove(upload_file)
            continue

        # list the numbers of the parts
        part_numbers = []
        for part_name in os.listdir(f'{_stream_cache}/{function_id}/{dataset_name}'):
            part_numbers.append(int(os.path.splitext(part_name)[0].split('-')[1]))

        # list the part numbers of the mongo documents
        mongo_part_numbers = []
        for mongo_name in os.listdir(f'{_mongo_cache}/{function_id}/{dataset_name}'):
            pn = int(os.path.splitext(mongo_name)[0].split('-')[1])

            # remove mongo docs that have no part
            if pn not in part_numbers:
                os.remove(f'{_mongo_cache}/{function_id}/{dataset_name}/{mongo_name}')

            mongo_part_numbers.append(pn)

        # remove parts that have no mongo doc
        for pn in part_numbers:
            if pn not in mongo_part_numbers:
                part_numbers.remove(pn)
                os.remove(f'{_stream_cache}/{function_id}/{dataset_name}/part-{pn}.json')

        if 'shared_context' in config:
            config['shared_context']['s3_stream'][fid_dataset] = (remote_key, upload_id)
            config['shared_context']['s3_stream_part'] = max(part_numbers) + 1
            config['shared_context']['s3_stream_user'][fid_dataset] = 0

            # request the manager listener for creation of more complex objects
            req_tuple = (fid_dataset, "s3_complete_comm")
            config['shared_context']['request_queue'].put(req_tuple)
            while config['shared_context']['ack_queue'].get() != req_tuple:
                pass

            # Configure local objects
            _s3_stream[fid_dataset] = ConnectedTarBuffer(remote_key, upload_id)
            config['shared_context']['s3_stream_user'][fid_dataset] += 1
            _s3_complete_lock[fid_dataset] = threading.Lock()
            threading.Thread(target=_complete_request_listener, args=(fid_dataset,)).start()

        _s3_uploading_count[fid_dataset] = 0
        _s3_done_event[fid_dataset] = threading.Event()

    # Recover the complete cache
    for dataset_path in glob.glob(f'{_stream_complete_cache}/{function_id}/*'):
        dataset_name = os.path.basename(dataset_path)

        for complete_path in glob.glob(f'{_stream_complete_cache}/{function_id}/{dataset_name}/*'):
            complete_name = os.path.basename(complete_path)
            try:
                json.load(open(f'{complete_path}/upload.json'))
            except:
                logger.error(f"Failed to complete upload {dataset_path}/{complete_name}")
                continue

            logger.info(f"Completing upload {dataset_path}/{complete_name}...")
            threading.Thread(target=_complete_upload, args=(complete_path,)).start()



def _upload_part(function_id, dataset, part_number, mongo_buf):
    """
    This is the most time costly part of a file write operation.
    """
    fid_dataset = f'{function_id}/{dataset}'

    # Upload part
    part_pk = _s3_stream[fid_dataset].send_buf(part_number)

    # Save the part ETag and PartNumber
    part_full_path = f'{_stream_cache}/{function_id}/{dataset}'
    os.makedirs(part_full_path, exist_ok=True)

    part_file_name = f'part-{part_number}.json'
    part_file_path = os.path.join(part_full_path, part_file_name)

    # Flush mongo buffer
    _flush_mongo_buffer(function_id, dataset, part_number, mongo_buf)

    json.dump(part_pk, open(part_file_path, 'w'))

    # Decrease running uploads count and notify if there is no more
    _s3_uploading_count[fid_dataset] -= 1
    if _s3_uploading_count[fid_dataset] == 0:
        _s3_done_event[fid_dataset].set()



def _write_image_stream(document, image_file, function_id, file_name_field):
    # There are two main locks used in this function : LOCAL complete and GLOBAL complete
    # -----------------------------------------------------------------
    #               xxxxx : LOCAL        ===== : GLOBAL
    # Lock:                    [===========[x=]xxxxxxxxxxx][==============]
    # Operation:   [CHECK FILE][CHECK STREAM][PROCESS FILE][CHECK COMPLETE]
    # -----------------------------------------------------------------
    # Local complete lock has action on the _complete_request_listener function below.
    # It prevents the local data to be sent to the completing process and therefore it
    # prevents the batch to be fully closed before the local file is finished being processed.

    global _cache_size
    global _s3_stream
    global _mongo_buffer

    if 'dataset' not in document:
        raise Exception('Every uploaded file must be linked to a dataset')

    dataset = document['dataset']

    file_name = document[file_name_field]
    fid_dataset = f'{function_id}/{dataset}'

    # if a file with this name already exists, abort
    if not mongo_connection[config['project_name'] + '-pipeline'].find_one_and_update(
            {'_id': document['_id'], '_tar_' + file_name_field: {'$exists': False}},
            {'$set': {'_tar_' + file_name_field: config['machine_id']}},
            {}):
        return

    # ----------------- at this point, the file has been checked -----------------

    # All the following must be done one thread at a time per dataset

    # Potential upload thread that needs to be joined at the end of the file
    upload_thread = None

    # Create the stream thread lock
    with _s3_thread_lock:
        if fid_dataset not in _s3_stream_lock:
            _s3_stream_lock[fid_dataset] = threading.Lock()
    # Lock the threads for this stream
    with _s3_stream_lock[fid_dataset]:
        # ****
        # Ensure that a stream exists
        # ****

        # *************** LOCK GLOBAL COMPLETE ***************
        # The stream cannot be used when complete is ongoing
        if 'shared_context' in config:
            # ********* BRUTAL LOCK *********
            # To create only one complete lock
            with config['shared_context']['s3_stream_lock']:
                if not fid_dataset in config['shared_context']['s3_complete_lock']:
                    req_tuple = (fid_dataset, 's3_complete_lock')

                    # Send request to create the lock
                    config['shared_context']['request_queue'].put(req_tuple)

                    # Wait for acknowledgment
                    while config['shared_context']['ack_queue'].get() != req_tuple:
                        pass

            # ********* BRUTAL LOCK *********
            config['shared_context']['s3_complete_lock'][fid_dataset].acquire()

        fid_dataset = f'{function_id}/{dataset}'
        if fid_dataset not in _s3_stream:

            # If a stream has already been created by another process, get it.
            if 'shared_context' in config and fid_dataset in config['shared_context']['s3_streams']:
                _s3_stream[fid_dataset] = config['shared_context']['s3_streams'][fid_dataset]

            # Otherwise a new stream must be created
            else:
                # Compute remote tar name from the _id of
                # the current document (first of the batch).
                tar_prefix = file_name_field + "!"
                tar_name = tar_prefix + hashlib.md5(document['_id'].encode()).hexdigest() + ".tar"
                tar_dest = '/'.join([dataset, tar_name])

                # Create the new stream
                tar_stream = ConnectedTarBuffer(tar_dest)

                # Save it locally
                _s3_stream[fid_dataset] = tar_stream

                stream_wrapped = (tar_stream.key, tar_stream.upload_id)

                # Save it globally and initialize the stream shared context
                if 'shared_context' in config:
                    config['shared_context']['s3_stream'][fid_dataset] = stream_wrapped
                    config['shared_context']['s3_stream_user'][fid_dataset] = 0
                    config['shared_context']['s3_stream_part'][fid_dataset] = 1

                    # request the manager listener for creation of more complex objects
                    req_tuple = (fid_dataset, "s3_complete_comm")
                    config['shared_context']['request_queue'].put(req_tuple)
                    while config['shared_context']['ack_queue'].get() != req_tuple:
                        pass

                # Save it as a file for recovery
                stream_full_path = f'{_stream_cache}/{function_id}'
                os.makedirs(stream_full_path, exist_ok=True)
                stream_file_path = f'{stream_full_path}/{dataset}.json'

                json.dump(stream_wrapped, open(stream_file_path, 'w'))

            if 'shared_context' in config:
                config['shared_context']['s3_stream_user'][fid_dataset] += 1
                _s3_complete_lock[fid_dataset] = threading.Lock()
                threading.Thread(target=_complete_request_listener, args=(fid_dataset,)).start()

            _s3_uploading_count[fid_dataset] = 0
            _s3_done_event[fid_dataset] = threading.Event()

        # *************** LOCK LOCAL COMPLETE ***************                   # -------------------------------
        if 'shared_config' in config:                                           # -------------------------------
            _s3_complete_lock[fid_dataset].acquire()                            # ****** OVERLAPPING LOCKS ******                                  #
                                                                                # -------------------------------
        # *************** UNLOCK GLOBAL COMPLETE ***************                # Allows complete to be triggered
        if 'shared_context' in config:                                          # but not completed until
            config['shared_context']['s3_complete_lock'][fid_dataset].release() # this file has been processed.

        # ----------------- at this point, the stream exists -----------------
        # --------------- and the file can be safely processed ---------------

        # **** Add the file to the local stream ****
        fake_file = io.BytesIO()
        fake_file.name = document[file_name_field]
        image_file.save(fake_file)
        file_size = fake_file.getbuffer().nbytes
        file_start_byte = _s3_stream[fid_dataset].buf_size

        _s3_stream[fid_dataset].add_file(fake_file.getvalue(),
                                         file_name,
                                         file_size)
        # ****

        # **** Add the mongo document to the mongo buffer
        if not fid_dataset in _mongo_buffer:
            _mongo_buffer[fid_dataset] = []

        # modify the mongo_documents with the name of the tar and the position of the file in it
        document[f'_tar_{file_name_field}'] = _s3_stream[fid_dataset].key
        document[f'_pos_{file_name_field}'] = file_start_byte

        _mongo_buffer[fid_dataset].append(document)
        # ****

        # **** Modify all global tracking variables
        if 'shared_context' in config:
            # ********* LOCK VARIABLES *********
            with config['shared_context']['s3_stream_lock']:

                # Get and increase part number if needed
                if _s3_stream[fid_dataset].buf_size > ConnectedTarBuffer.MIN_SIZE:
                    part_number = config['shared_context']['s3_stream_part'][fid_dataset]
                    config['shared_context']['s3_stream_part'][fid_dataset] += 1

                # Increase stream size
                config['shared_context']['s3_stream_size'][fid_dataset] += file_size

            # ********* UNLOCK VARIABLES *********
        else:
            if fid_dataset not in _s3_stream_size:
                _s3_stream_size[fid_dataset] = 0
            _s3_stream_size[fid_dataset] += file_size

            if _s3_stream[fid_dataset].buf_size > ConnectedTarBuffer.MIN_SIZE:
                part_number = _s3_stream[fid_dataset].part_number
        # ****

        # **** Flush the local buffers if needed
        if _s3_stream[fid_dataset].buf_size > ConnectedTarBuffer.MIN_SIZE:
            # Increase the running uploads count
            _s3_uploading_count[fid_dataset] += 1

            # Defuse the event
            _s3_done_event[fid_dataset].unset()

            # Launch the uploading thread which :
            # - uploads the part
            # - saves it
            # - flushes the mongo_buffer
            # - decrease the running uploads count
            upload_thread = threading.Thread(target=_upload_part,
                             args=(function_id,
                                   dataset,
                                   part_number,
                                   _mongo_buffer[fid_dataset]))
            upload_thread.start()
        # ****

        # *************** UNLOCK LOCAL COMPLETE ***************
        # Now if complete was ongoing, it can be completed with the current file
        if 'shared_context' in config:
            _s3_complete_lock[fid_dataset].release()

        # ------------ at this point, the file has been processed ------------
        # ------- we need to check if the batch needs to be completed --------

        if 'shared_context' in config:

            # *************** LOCK GLOBAL COMPLETE ***************
            # This complete check must be done one process at a time
            # If one process detects that complete must be triggered
            # all the other processes can leave.
            while not config['shared_context']['s3_complete_lock'][fid_dataset].acquire(timeout=0.1):
                # If complete is ongoing, there is no use in trying to trigger it
                if config['shared_context']['s3_complete_event'][fid_dataset].is_set():
                    return

            # ********* LOCK VARIABLES *********
            with config['shared_context']['s3_stream_lock']:
                if config['shared_context']['s3_stream_size'][fid_dataset] < config['flush_size']:
                    # *************** UNLOCK GLOBAL COMPLETE ***************
                    # If the batch doesn't need to be completed, leave.
                    config['shared_context']['s3_complete_lock'][fid_dataset].release()
                    return
            # ********* UNLOCK VARIABLES *********

            # ***************** LAUNCH COMPLETE *******************
            # **** Get all remaining data from other processes ****

            # protect local stream
            local_stream = _s3_stream[fid_dataset]

            # trigger complete event to launch _complete_request_listener
            # functions of each process
            config['shared_context']['s3_complete_event'][fid_dataset].set()

            # ensure to get the data from each process
            # may take a little while as data won't be
            # sent until their current file has been processed
            local_stream.clear()
            for _ in range(config['shared_context']['s3_stream_user'][fid_dataset]):
                local_stream.write(config['shared_context']['s3_complete_pipe'][fid_dataset].get())

            last_part_number = config['shared_context']['s3_stream_part'][fid_dataset]

        else:
            # For one process
            local_stream = _s3_stream[fid_dataset]

            _flush_mongo_buffer(function_id, dataset, local_stream.part_number)
            last_part_number = local_stream.part_number

            # wait for all upload threads to be done
            _s3_done_event[fid_dataset].wait()

        # **** Complete the tar file and send the last part to s3
        local_stream.complete_tar_file()
        part_pk = local_stream.send_buf(last_part_number)

        part_full_path = f'{_stream_cache}/{function_id}/{dataset}'
        os.makedirs(part_full_path, exist_ok=True)

        part_file_name = f'part-{last_part_number}.json'
        part_file_path = os.path.join(part_full_path, part_file_name)

        json.dump(part_pk, open(part_file_path, 'w'))
        # ****

        # Transfer stream and mongo caches contents to the complete cache
        dir_name = os.path.splitext(os.path.split(local_stream.key)[-1])[0]
        cache_path = f'{_stream_complete_cache}/{function_id}/{dataset}/{dir_name}'
        os.makedirs(cache_path, exist_ok=True)

        # Move mongo docs
        mongo_dest = f'{cache_path}/mongo/'
        os.makedirs(mongo_dest, exist_ok=True)
        for mongo_file in glob.iglob(f'{_mongo_cache}/{function_id}/{dataset}/*'):
            shutil.move(mongo_file, mongo_dest + os.path.basename(mongo_file))

        # Move parts files
        parts_dest = f'{cache_path}/parts/'
        os.makedirs(parts_dest, exist_ok=True)
        for mongo_file in glob.iglob(f'{_stream_cache}/{function_id}/{dataset}/*'):
            shutil.move(mongo_file, parts_dest + os.path.basename(mongo_file))

        # Move upload
        upload_dest = f'{cache_path}/upload.json'
        shutil.move(f'{_stream_cache}/{function_id}/{dataset}.json', upload_dest)

        # Now that every thing is safe in the complete cache
        # delete the stream and release the complete lock
        if 'shared_context' in config:
            # Delete the global stream
            del config['shared_context']['s3_stream'][fid_dataset]

            # *************** UNLOCK GLOBAL COMPLETE ***************
            config['shared_context']['s3_complete_lock'][fid_dataset].release()

        logger.info('Merging uploaded parts into batch file...')

        threading.Thread(target=_complete_upload, args=(cache_path,)).start()

    # Join the potential upload thread
    upload_thread.join()


# The purpose of this function, which must be launched
# in a separate thread for every new stream
# shared between processes, is to send the data
# that has not yet been sent to s3 to the responsible
# process when the stream is being closed.
def _complete_request_listener(function_id, dataset):
    fid_dataset = f'{function_id}/{dataset}'

    # This shared event object is triggered by the
    # process that decides to close the s3 stream.
    config['shared_context']['s3_complete_event'][fid_dataset].wait()

    with _s3_complete_lock[fid_dataset]:
        # wait for all upload threads to be done
        _s3_done_event[fid_dataset].wait()

        # collect the content of the local stream object buffer
        remaining_bytes = _s3_stream[fid_dataset].buf.getvalue()

        last_part_number = config['shared_context']['s3_stream_part'][fid_dataset]
        _flush_mongo_buffer(function_id, dataset, last_part_number)

        # send it to the closing process
        config['shared_context']['s3_complete_pipe']['dataset'].put(remaining_bytes)

        # If the processing closing the tar file is not this one,
        # delete
        del _s3_stream[fid_dataset]
