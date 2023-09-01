import threading
from loguru import logger

from lib.iterable import _generator
from lib.mongodb import _write_document, _ensure_index
from lib.s3 import _write_image, _recover_s3_cache
from lib.s3_stream import _write_image_stream
from lib.local import _write_local
from lib.config import _random_name
from lib.manager import manager


def _fix_item(item, file_name):
    if file_name not in item:
        item[file_name] = _random_name()
    elif '*' in item[file_name]:
        item[file_name] = item[file_name].replace('*', _random_name(), 1)
        item[file_name] = item[file_name].replace('*', '', -1)


def _Output(output_type,
            compartment='_id',
            buffer_size=1,
            max_threads=1,
            stream=False,

            function_id=None,
            dataset_name=None):
    _ensure_index([f'_tar_{compartment}'])

    if output_type in ('database', 's3', 'local'):
        shared_attr = {}
        if dataset_name:
            shared_attr['dataset'] = dataset_name

    # write in the console
    if output_type == "stdout":
        def transformer(func, sub_proc_wrapped):
            def wrapper(*args, **kwargs):
                for item in _generator(func, *args, **kwargs):
                    print(item)
                yield None

            return wrapper

        return transformer

    # upload to database
    elif output_type == "database":
        def transformer(func, sub_proc_wrapped):
            def wrapper(*args, **kwargs):
                sem = threading.Semaphore(max_threads * 2)

                def _write_thread(item, buffer_size):
                    _write_document(item, buffer_size)
                    sem.release()

                for item in _generator(func, *args, **kwargs):
                    # create _id if it doesn't have one
                    _fix_item(item, '_id')

                    # add function mark and potential dataset name
                    item.update(shared_attr)

                    # wait for a thread slot to be available
                    sem.acquire()
                    # upload it to the database
                    threading.Thread(target=_write_thread, args=(item, buffer_size)).start()

                yield None

            return wrapper

        return transformer

    # upload to s3
    elif output_type == "s3":
        def transformer(func, sub_proc_wrapped):
            def wrapper(*args, **kwargs):

                # try recovering cached data
                # (needs to be done here after the processes are launched)
                if not stream:
                    if manager._global:
                        with manager._global.recover_lock:
                            if not manager._global.recovered.value:
                                _recover_s3_cache(function_id)
                                manager._global.recovered.value = True
                    else:
                        _recover_s3_cache(function_id)

                # Setup write function for multithreading
                write_sem = threading.Semaphore(max_threads)
                if stream:
                    def write_s3_thread(*input_tuple):
                        _write_image_stream(*input_tuple, function_id, compartment)
                        write_sem.release()
                else:
                    def write_s3_thread(*input_tuple):
                        _write_image(*input_tuple, function_id, compartment)
                        write_sem.release()

                for in_tuple in _generator(func, *args, **kwargs):
                    # ignore none
                    if in_tuple is None:
                        continue

                    # mongo_doc, image
                    item, ws = in_tuple

                    # create _id if it doesn't have one
                    _fix_item(item, '_id')
                    if compartment != '_id':
                        _fix_item(item, compartment)

                    # write additional information to the document
                    item.update(shared_attr)

                    # upload
                    write_sem.acquire()
                    threading.Thread(target=write_s3_thread, args=(item, ws)).start()
                yield None

            return wrapper

        return transformer

    # save locally
    elif output_type == "local":
        # setup write function for multithreading
        write_sem = threading.Semaphore(max_threads)

        def write_local_thread(*input_tuple):
            _write_local(*input_tuple, compartment)
            write_sem.release()

        def transformer(func, sub_proc_wrapped):
            def wrapper(*args, **kwargs):

                for intuple in _generator(func, *args, **kwargs):

                    # ignore none
                    if intuple is None:
                        continue

                    # mongo_doc, image
                    item, ws = intuple

                    # create _id if it doesn't have one
                    _fix_item(item, '_id')
                    if compartment != '_id':
                        _fix_item(item, compartment)

                    # write additional information to the document
                    item.update(shared_attr)

                    # save
                    write_sem.acquire()
                    threading.Thread(target=write_local_thread, args=(item, ws)).start()
                yield None

            return wrapper

        return transformer

    elif output_type == "pipe":
        def transformer(func, sub_proc_wrapped):
            return func

        return transformer
