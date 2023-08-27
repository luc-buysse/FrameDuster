import os.path
import threading
import io
import struct
from tarfile import BLOCKSIZE

from frameduster.config import config
from frameduster.s3 import _s3_client

def _process_tar_batch(object_list, target, to_do, event, last=False):
    ts = S3TarStream(target)

    for object in object_list:
        ts.copy_file(object['Key'], object['Size'], object['Name'])

    ts.complete_tar_file(last)

    to_do[0] -= 1
    if to_do[0] == 0 and to_do[1]:
        event.set()

def make_remote_tar(object_list, target):
    """
    Creates a tar file object on a s3 server from a list of objects.
    Object items must have attributes : 'Key', 'Size' (in bytes) and 'Name'
        the id of the object in the tar file.
    The target in the key of the tar file created.
    This function is not completely reliable, it may fail.

    Do not hesitate to parallelize it.
    """
    total_size_outer = sum([obj['Size'] + 1024 for obj in object_list])
    total_size_inner = sum([obj['Size'] for obj in object_list])
    min_batch_size = 5 * 1024 * 1024
    batch_size = min_batch_size

    tmp = int(total_size_outer / 10_000)
    if tmp > batch_size:
        batch_size = tmp

    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    wfile = "working-dir/" + target
    to_do = [0, False]
    done_event = Event()

    part_id = 0
    current_list = []
    current_size = 0
    processed_size = 0
    for i, obj in enumerate(object_list):
        current_list.append(obj)
        current_size += obj['Size']
        processed_size += obj['Size']

        if current_size > batch_size and total_size_inner - processed_size > min_batch_size:
            to_do[0] += 1

            threading.Thread(target=_process_tar_batch, args=(current_list, f'{wfile}-{part_id}', to_do, done_event)).start()

            part_id += 1
            current_list = []
            current_size = 0
    to_do[0] += 1
    threading.Thread(target=_process_tar_batch, args=(current_list, f'{wfile}-{part_id}', to_do, done_event, True)).start()
    to_do[1] = True

    done_event.wait()

    upload_id = _s3_client.create_multipart_upload(Bucket=config['s3']['bucket'],
                                                   Key=target)['UploadId']

    parts = []
    def add_part(i):
        nonlocal upload_id
        nonlocal target
        nonlocal parts

        etag = _s3_client.upload_part_copy(Bucket=config['s3']['bucket'],
                                           CopySource={'Bucket': config['s3']['bucket'], 'Key': f'{wfile}-{i}'},
                                           UploadId=upload_id,
                                           Key=target,
                                           PartNumber=i + 1)['CopyPartResult']['ETag']
        parts.append({"ETag": etag, "PartNumber": i + 1})

    with ThreadPoolExecutor() as workers:
        for i in range(part_id):
            workers.submit(add_part, i)

    parts.sort(key=lambda x:  x['PartNumber'])
    _s3_client.complete_multipart_upload(Bucket=config['s3']['bucket'],
                                         UploadId=upload_id,
                                         Key=target,
                                         MultipartUpload={'Parts': parts})


class S3TarStream:
    # Various constants from tarfile.py that we need
    GNU_FORMAT = 1
    NUL = b"\0"
    BLOCKSIZE = 512
    RECORDSIZE = BLOCKSIZE * 20
    MIN_SIZE = 5 * 1024 * 1024

    def stn(self, s, length, encoding, errors):
        # Convert a string to a null-terminated bytes object.
        s = s.encode(encoding, errors)
        return s[:length] + (length - len(s)) * self.NUL

    def itn(self, n, digits=8, format=GNU_FORMAT):
        # Convert a python number to a number field.
        # POSIX 1003.1-1988 requires numbers to be encoded as a string of
        # octal digits followed by a null-byte, this allows values up to
        # (8**(digits-1))-1. GNU tar allows storing numbers greater than
        # that if necessary. A leading 0o200 or 0o377 byte indicate this
        # particular encoding, the following digits-1 bytes are a big-endian
        # base-256 representation. This allows values up to (256**(digits-1))-1.
        # A 0o200 byte indicates a positive number, a 0o377 byte a negative
        # number.
        original_n = n
        n = int(n)
        if 0 <= n < 8 ** (digits - 1):
            s = bytes("%0*o" % (digits - 1, n), "ascii") + self.NUL
        elif format == self.GNU_FORMAT and -256 ** (digits - 1) <= n < 256 ** (digits - 1):
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
        """Calculate the checksum for a member's header by summing up all
        characters except for the chksum field which is treated as if
        it was filled with spaces. According to the GNU tar sources,
        some tars (Sun and NeXT) calculate chksum with signed char,
        which will be different if there are chars in the buffer with
        the high bit set. So we calculate two checksums, unsigned and
        signed.
        """
        unsigned_chksum = 256 + sum(struct.unpack_from("148B8x356B", buf))
        signed_chksum = 256 + sum(struct.unpack_from("148b8x356b", buf))
        return unsigned_chksum, signed_chksum

    def __init__(self, dest, upload_id=None):

        self.buf = io.BytesIO()
        self.buf_size = 0
        self.padded = False

        # setup the padding
        # must be the name of a remote file
        # of size 1024 * 1024 * 5 bytes
        # his content is of no importance
        self.padding = "padding"

        self.upload_id = upload_id
        self.key = dest
        self.part_number = 1
        self.parts = []

        if not self.upload_id:
            self.open()

    def addFileRecord(self, filename, filesize):
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

        buf = b""
        if len(tarinfo["name"].encode(encoding, errors)) > LENGTH_NAME:
            raise Exception("Filename is too long for tar file header.")

        devmajor = self.stn("", 8, encoding, errors)
        devminor = self.stn("", 8, encoding, errors)

        parts = [
            self.stn(tarinfo.get("name", ""), 100, encoding, errors),
            self.itn(tarinfo.get("mode", 0) & 0o7777, 8, self.GNU_FORMAT),
            self.itn(tarinfo.get("uid", 0), 8, self.GNU_FORMAT),
            self.itn(tarinfo.get("gid", 0), 8, self.GNU_FORMAT),
            self.itn(tarinfo.get("size", 0), 12, self.GNU_FORMAT),
            self.itn(tarinfo.get("mtime", 0), 12, self.GNU_FORMAT),
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

    def add_file(self, file_bytes, name, size):
        # record
        self.addFileRecord(name, size)

        # content
        self.write(file_bytes)

        # complete
        self.completeFileRecord(size)

    def copy_file(self, key, size=None, name=None):
        # record
        if not size:
            size = _s3_client.get_object(Bucket=config['s3']['bucket'], Key=key)['ContentLength']
        self.addFileRecord(name if name else os.path.basename(key), size)
        self.send()

        self.copy(key, size)

        # padding
        self.completeFileRecord(size)

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
        self.parts.append(part_pk)
        self.part_number += 1

        # reset
        self.buf = io.BytesIO()
        self.buf_size = 0

        return part_pk

    def add_padding(self):
        etag = _s3_client.upload_part_copy(Bucket=config['s3']['bucket'],
                                    CopySource={'Bucket': config['s3']['bucket'], 'Key': self.padding},
                                    UploadId=self.upload_id,
                                    Key=self.key,
                                    PartNumber=self.part_number)['CopyPartResult']['ETag']
        self.parts.append({'ETag': etag, 'PartNumber': self.part_number})
        self.part_number += 1

    def remove_padding(self):
        size = _s3_client.get_object(Bucket=config['s3']['bucket'], Key=self.key)['ContentLength']
        try:
            etag = _s3_client.upload_part_copy(Bucket=config['s3']['bucket'],
                                               CopySource={'Bucket': config['s3']['bucket'], 'Key': self.key},
                                               UploadId=self.upload_id,
                                               Key=self.key,
                                               CopySourceRange=f'bytes={self.MIN_SIZE}-{size-1}',
                                               PartNumber=self.part_number)['CopyPartResult']['ETag']
        except Exception as e:
            print(e)
            print("Bytes :", f'bytes={self.MIN_SIZE}-{size-1}')
        self.parts.append({'ETag': etag, 'PartNumber': self.part_number})
        self.part_number += 1

    def send(self):
        """
        Returns True if it has successfully sent the content of the buffer.
        """
        if self.buf_size < self.MIN_SIZE:
            if self.part_number == 1:
                self.padded = True
                self.add_padding()

            self.send_buf()
            self.merge()
        else:
            self.send_buf()

    def copy(self, key, size):
        if size < self.MIN_SIZE:
            if self.padded:
                self.copy_force(key)
                self.merge()
                return True
            else:
                return False
        else:
            self.copy_force(key)
            self.padded = True
            return True

    def copy_force(self, key):
        etag = _s3_client.upload_part_copy(Bucket=config['s3']['bucket'],
                                           CopySource={'Bucket': config['s3']['bucket'], 'Key': key},
                                           UploadId=self.upload_id,
                                           Key=self.key,
                                           PartNumber=self.part_number)['CopyPartResult']['ETag']
        self.parts.append({'ETag': etag, 'PartNumber': self.part_number})
        self.part_number += 1

    def completeFileRecord(self, size):
        # write the end-of-file marker
        blocks, remainder = divmod(size, BLOCKSIZE)
        if remainder > 0:
            self.write(self.NUL * (BLOCKSIZE - remainder))

    def open(self):
        self.upload_id = _s3_client.create_multipart_upload(Bucket=config['s3']['bucket'],
                                                            Key=self.key)['UploadId']
        self.parts = []
        self.part_number = 1

    def close(self):
        try:
            _s3_client.complete_multipart_upload(Bucket=config['s3']['bucket'],
                                                 UploadId=self.upload_id,
                                                 Key=self.key,
                                                 MultipartUpload={'Parts': self.parts})
        except Exception as e:
            print("Error while closing file : ", self.key, " : ", e)

    def merge(self):
        self.close()
        self.open()
        self.copy_force(self.key)

    def abort(self):
        _s3_client.abort_multipart_upload(Bucket=config['s3']['bucket'],
                                          UploadId=self.upload_id,
                                          Key=self.key)

    def complete_tar_file(self, end_tar=True):
        if end_tar:
            self.write(self.NUL * (BLOCKSIZE * 2))
            self.buf_size += (BLOCKSIZE * 2)
            blocks, remainder = divmod(self.buf_size, self.RECORDSIZE)
            if remainder > 0:
                self.write(self.NUL * (BLOCKSIZE * 2))

        self.send()
        self.close()

        if self.padded:
            self.open()
            self.remove_padding()

            try:
                self.close()
            except Exception as e:
                self.abort()
                raise Exception("Cannot complete file : parts altogether are too small")
