import os.path
import threading
import io
import struct
from tarfile import BLOCKSIZE
from loguru import logger
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from lib.config import config
from lib.s3 import _s3_client, _stream_cache
from lib.mongodb import mongo_connection, _write_document
from lib.config import _random_name
from lib.manager import manager


class ZeroPin:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.event = threading.Event()
        self.event.set()

    def wait(self):
        self.event.wait()

    def inc(self):
        with self.lock:
            self.count += 1
            self.event.clear()

    def dec(self):
        with self.lock:
            self.count -= 1
            if self.count == 0:
                self.event.set()


class MultipartUpload:
    def __init__(self, key, upload_id):
        self.key = key
        self.upload_id = upload_id

    def send(self, body, part_number):
        etag = _s3_client.upload_part(Body=body,
                                      Bucket=config['s3']['bucket'],
                                      UploadId=self.upload_id,
                                      Key=self.key,
                                      PartNumber=part_number,
                                      ContentLength=len(body))['ETag']
        return {'ETag': etag, 'PartNumber': part_number}

    @classmethod
    def open(cls, key):
        upload_id = _s3_client.create_multipart_upload(Bucket=config['s3']['bucket'],
                                                            Key=key)['UploadId']
        return cls(key, upload_id)

    def close(self, parts):
        try:
            parts.sort(key=lambda x: x['PartNumber'])

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

    def get_wrapped(self):
        return self.key, self.upload_id


class TarWriter:
    # Various constants from tarfile.py that we need
    NUL = b"\0"
    BLOCKSIZE = 512
    RECORDSIZE = BLOCKSIZE * 20
    MIN_SIZE = 5 * 1024 * 1024

    @classmethod
    def stn(cls, s, length, encoding, errors):
        s = s.encode(encoding, errors)
        return s[:length] + (length - len(s)) * cls.NUL

    @classmethod
    def itn(cls, n, digits=8):
        n = int(n)
        if 0 <= n < 8 ** (digits - 1):
            s = bytes("%0*o" % (digits - 1, n), "ascii") + cls.NUL
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

    @classmethod
    def calc_chksums(cls, buf):
        unsigned_chksum = 256 + sum(struct.unpack_from("148B8x356B", buf))
        signed_chksum = 256 + sum(struct.unpack_from("148b8x356b", buf))
        return unsigned_chksum, signed_chksum


    @classmethod
    def add_file_record(cls, output: io.BytesIO, filename, filesize):
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

        devmajor = cls.stn("", 8, encoding, errors)
        devminor = cls.stn("", 8, encoding, errors)

        parts = [
            cls.stn(tarinfo.get("name", ""), 100, encoding, errors),
            cls.itn(tarinfo.get("mode", 0) & 0o7777, 8),
            cls.itn(tarinfo.get("uid", 0), 8),
            cls.itn(tarinfo.get("gid", 0), 8),
            cls.itn(tarinfo.get("size", 0), 12),
            cls.itn(tarinfo.get("mtime", 0), 12),
            b"        ",  # checksum field
            tarinfo.get("type", REGTYPE),
            cls.stn(tarinfo.get("linkname", ""), 100, encoding, errors),
            tarinfo.get("magic", GNU_MAGIC),
            cls.stn(tarinfo.get("uname", ""), 32, encoding, errors),
            cls.stn(tarinfo.get("gname", ""), 32, encoding, errors),
            devmajor,
            devminor,
            cls.stn(tarinfo.get("prefix", ""), 155, encoding, errors)
        ]
        buf = struct.pack("%ds" % BLOCKSIZE, b"".join(parts))
        chksum = cls.calc_chksums(buf[-BLOCKSIZE:])[0]
        buf = buf[:-364] + bytes("%06o\0" % chksum, "ascii") + buf[-357:]

        output.write(buf)
        return len(buf)

    @classmethod
    def complete_file_record(cls, buf, size):
        # write the end-of-file marker
        blocks, remainder = divmod(size, BLOCKSIZE)
        if remainder > 0:
            buf.write(cls.NUL * (BLOCKSIZE - remainder))
            return BLOCKSIZE - remainder
        return 0

    @classmethod
    def complete_tar(cls, buf: io.BytesIO):
        buf.write(cls.NUL * (BLOCKSIZE * 2))
        blocks, remainder = divmod(buf.tell(), cls.RECORDSIZE)
        if remainder > 0:
            buf.write(cls.NUL * (BLOCKSIZE * 2))


class _Substream:
    def __init__(self, stream_group):
        function_id, dataset, field = stream_group.split('/')
        self.stream_group = stream_group
        self.field = field

        # Try to join a stream at the process level
        if manager._global and stream_group in manager._global.streams:
            stream_wrapped = manager._global.streams[stream_group]
            self.conn = MultipartUpload(*stream_wrapped)

        # Otherwise create a new stream from scratch
        else:
            # Create the new stream
            self.conn = MultipartUpload.open(f'{dataset}/{field}!{_random_name(10)}.tar')

            # Register at the process level
            # Get all shared object from the manager
            self.stream = None
            if manager._global:
                manager._global.streams[stream_group] = self.conn.get_wrapped()

            # Save it as a file for recovery
            json.dump(self.conn.get_wrapped(), open(f'{_stream_cache}/{function_id}!{dataset}!{field}.json', 'w'))

        if manager._global:
            self.stream = manager._global.get_stream_handle(self.conn.key)

        if self.stream:
            self.stream.user_count.value += 1
            threading.Thread(target=self.complete_listener_start).start()
        else:
            self.part_lock = threading.Lock()
            self.stream_part = 1

        self.upload_pin = ZeroPin()
        self.write_pin = ZeroPin()

        self.write_lock = threading.Lock()
        self.mongo_buffer = []
        self.mongo_tmp = []
        self.parts = []
        self.buffer = io.BytesIO()

        self.n_bytes = 0
        self.done = False

        if not self.stream:
            self.stream_pos = 0

    def write(self, doc, img):
        tmp_buffer = io.BytesIO()

        tmp_buffer.name = doc[self.field]
        img.save(tmp_buffer)
        img_bytes = tmp_buffer.getvalue()
        n_bytes = tmp_buffer.tell()

        with self.write_lock:
            doc[f'_tar_{self.field}'] = os.path.basename(self.conn.key)
            doc[f'_pos_{self.field}'] = self.buffer.tell()
            self.mongo_tmp.append(doc)

            n_bytes += TarWriter.add_file_record(self.buffer, doc[self.field], len(img_bytes))
            self.buffer.write(img_bytes)
            n_bytes += TarWriter.complete_file_record(self.buffer, len(img_bytes))

            if self.buffer.tell() > TarWriter.MIN_SIZE:
                to_send = self.buffer.getvalue()
                self.buffer = io.BytesIO()
                self.upload_pin.inc()
                threading.Thread(target=self.send, args=(to_send,)).start()

        if self.stream:
            with self.stream.lock:
                self.stream.stream_size.value += n_bytes
        else:
            self.n_bytes += n_bytes

    def unbind(self):
        self.done = True
        if self.stream:
            self.stream.done.value = True
            del manager._global.streams[self.stream_group]

    def is_done(self):
        return self.done or self.stream.done.value

    def complete(self):
        self.write_pin.wait()

        if self.stream:
            self.stream.complete_event.set()

            last_buffer = io.BytesIO()
            last_mongo_tmp = []
            all_parts = []
            all_mongo = []
            for _ in range(self.stream.user_count.value):
                parts, sub_buffer, mongo_buffer, mongo_tmp = self.stream.complete_pipe.get()
                for doc in mongo_tmp:
                    doc[f'_pos_{self.field}'] += last_buffer.tell()
                last_mongo_tmp.extend(mongo_tmp)
                all_parts.extend(parts)
                last_buffer.write(sub_buffer)
                all_mongo.extend(mongo_buffer)
            self.mongo_buffer = all_mongo
            self.mongo_tmp = last_mongo_tmp
            self.parts = all_parts
            TarWriter.complete_tar(last_buffer)
            self.send(last_buffer.getvalue())
        else:
            self.upload_pin.wait()
            TarWriter.complete_tar(self.buffer)
            self.send(self.buffer.getvalue())

        self.conn.close(self.parts)

        self.send_mongo()

    def send(self, to_send):
        part_size = len(to_send)

        # get stream part
        if self.stream:
            with self.stream.lock:
                stream_part = self.stream.stream_part.value
                stream_pos = self.stream.stream_pos.value
                self.stream.stream_part.value += 1
                self.stream.stream_pos.value += part_size
        else:
            with self.part_lock:
                stream_part = self.stream_part
                stream_pos = self.stream_pos
                self.stream_part += 1
                self.stream_pos += part_size
        for doc in self.mongo_tmp:
            doc[f'_pos_{self.field}'] += stream_pos
        self.mongo_buffer.extend(self.mongo_tmp)
        self.mongo_tmp = []

        self.parts.append(self.conn.send(to_send, stream_part))
        self.upload_pin.dec()

    def complete_listener_start(self):
        # wait for complete
        self.stream.complete_event.wait()
        self.write_pin.wait()
        self.upload_pin.wait()

        # send it to the closing process
        remaining_bytes = self.buffer.getvalue()
        self.stream.complete_pipe.put((self.parts, remaining_bytes, self.mongo_buffer, self.mongo_tmp))

    def send_mongo(self):
        # send all mongo docs
        with ThreadPoolExecutor(max_workers=250) as workers:
            for doc in self.mongo_buffer:
                workers.submit(_write_document, doc)

        _, dataset, field = self.stream_group.split('/')
        s3_doc = {
            "_id": os.path.basename(self.conn.key),
            "dataset": dataset,
            'field': field
        }
        mongo_connection[config['project_name'] + "-pipeline-s3"].insert_one(s3_doc)
        mongo_connection[dataset + "-dataset-s3"].insert_one(s3_doc)

    def get_size(self):
        if self.stream:
            return self.stream.stream_size.value
        else:
            return self.n_bytes


_streams: dict[str, _Substream] = {}
_group_lock = threading.Lock()


def _acquire_group():
    if manager._global:
        manager._global.group_lock.acquire()
    else:
        _group_lock.acquire()


def _release_group():
    if manager._global:
        manager._global.group_lock.release()
    else:
        _group_lock.release()


def _write_image_stream(document, image_file, function_id, field):
    if 'dataset' not in document:
        raise Exception(f'No dataset found for this document : {document}')

    dataset = document['dataset']
    stream_group = f'{function_id}/{dataset}/{field}'

    _acquire_group()

    if stream_group not in _streams or _streams[stream_group].is_done():
        _streams[stream_group] = _Substream(stream_group)
    stream = _streams[stream_group]

    stream.write_pin.inc()
    _release_group()

    stream.write(document, image_file)

    stream.write_pin.dec()

    _acquire_group()
    if not stream.is_done() and stream.get_size() > config['s3']['flush_size']:
        stream.unbind()
        threading.Thread(target=stream.complete).start()
        del _streams[stream_group]
    _release_group()

