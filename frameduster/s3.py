import io
import json
import queue
import boto3
import os
import tarfile
import glob
import hashlib
import threading

import urllib3
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pymongo.errors
from pymongo.collection import ReturnDocument
from PIL import Image, UnidentifiedImageError
import shutil
from loguru import logger

from config import config
from mongodb import _write_document, mongo_connection
from config import _random_name
from manager import manager

_cache_size = 0
_cache_lock = threading.Lock()
_main_cache = os.path.join(config['app_path'], 'data/_cache')
_mongo_cache = os.path.join(config['app_path'], 'data/_mongo_cache')
_tar_cache = os.path.join(config['app_path'], 'data/_tar_cache')
_stream_cache = os.path.join(config['app_path'], 'data/_stream_cache')
_stream_complete_cache = os.path.join(config['app_path'], 'data/_stream_complete_cache')

os.makedirs(_stream_cache, exist_ok=True)
os.makedirs(_stream_complete_cache, exist_ok=True)


def _get_s3_client():
    client = boto3.client(
        's3',
        aws_access_key_id=config['s3']['access_key'],
        aws_secret_access_key=config['s3']['secret_access_key'],
        region_name=config['s3']['region'],
        endpoint_url=config['s3']['endpoint_url']
    )

    return client


_s3_client = _get_s3_client()


def _get_image(doc, field):
    if f'_tar_{field}' not in doc:
        raise Exception("Not batch seems to be associated with this document.")
    if f'_pos_{field}' not in doc:
        raise Exception("Not position was provided with the document.")
    if f'dataset' not in doc:
        raise Exception("Not dataset was provided with the document.")

    tar_name = doc[f'_tar_{field}']
    tar_pos = doc[f'_pos_{field}']

    http = urllib3.PoolManager()

    file_url = os.path.join(config['s3']['base_url'], config['s3']['bucket']) + f'/{doc["dataset"]}/{tar_name}'
    obj = http.request('get',
                       file_url,
                       preload_content=False,
                       headers={'Range': f'bytes={tar_pos}-'})

    header_reader = tarfile.TarFile(fileobj=obj, mode='r')
    if not header_reader.firstmember:
        logger.error(f"Cannot read the file in the compartment {field} associated with the document {doc}")

    return Image.open(io.BytesIO(obj.read(header_reader.firstmember.size)))


def _iterate_batch_s3(doc_s3,
                      function_id=None,
                      query=None,
                      projection=None,
                      slice=None,
                      get_bounds=False,
                      threads=50,
                      max_preload=20):
    collection_s3 = mongo_connection[config['project_name'] + "-pipeline-s3"]
    main_collection = mongo_connection[config['project_name'] + "-pipeline"]

    if not query:
        query = {}

    # assemble tar_name
    dataset = doc_s3['dataset']
    tar_origin = '/'.join([dataset, doc_s3['_id']])

    global_start_byte = 0
    if slice and len(slice) > 0:
        global_start_byte = slice[0]
    elif function_id:
        init_doc = collection_s3.find_one({'_id': doc_s3['_id']}, {function_id + "_sb": 1})

        if init_doc and function_id + "_sb" in init_doc:
            global_start_byte = init_doc[function_id + "_sb"]

    http = urllib3.PoolManager()
    file_url = os.path.join(config['s3']['base_url'], config['s3']['bucket']) + f'/{tar_origin}'

    global_end_byte = None
    if slice and len(slice) > 1:
        global_end_byte = slice[1]

    start_byte = global_start_byte

    # Setup checkout thread function
    sem = threading.Semaphore(threads)
    chk_running = 0
    out_queue = queue.Queue(maxsize=max_preload)

    def checkout(img, filename, bounds):
        nonlocal query
        nonlocal chk_running
        nonlocal out_queue
        doc_query = query.copy()

        # get document
        doc_query[doc_s3['field']] = filename
        query_args = [doc_query]

        # DEBUG
        if function_id in query:
            del query[function_id]

        if function_id:
            update = {'$set': {function_id: 1}}
            query_args.append(update)
            if projection:
                query_args.append(projection)
            file_doc = main_collection.find_one_and_update(
                *query_args,
                return_document=ReturnDocument.BEFORE
            )
        else:
            if projection:
                query_args.append(projection)
            file_doc = main_collection.find_one(*query_args)

        if not file_doc:
            logger.error(f"Cannot fetch the document for query : {query}")
            sem.release()
            chk_running -= 1
            return

        # update start_byte in the s3 collection
        if function_id:
            collection_s3.update_one({'_id': doc_s3['_id']}, {'$set': {function_id + "_sb": bounds[0]}})

        # Store the result
        if get_bounds:
            out_item = file_doc, img, bounds
        else:
            out_item = file_doc, img

        out_queue.put(out_item)
        sem.release()
        chk_running -= 1

    def get_obj(start):
        range_str = f'bytes={start}-'
        if global_end_byte:
            range_str += f'{global_end_byte}'

        return http.request('get',
                            file_url,
                            preload_content=False,
                            headers={'Range': range_str})

    loader_running = True

    def loader(from_byte):
        nonlocal start_byte
        nonlocal chk_running
        nonlocal loader_running

        obj = get_obj(from_byte)
        header_reader = tarfile.TarFile(fileobj=obj, mode='r')

        while True:
            try:
                next_file = header_reader.next()
            except tarfile.ReadError:
                break
            except io.UnsupportedOperation:
                # logger.error(f"Relaunching loader thread for : {doc_s3['_id']}")
                # threading.Thread(target=loader, args=(start_byte,)).start()
                break
            except BaseException:
                break
            if next_file is None:  # end of file, exit function
                break

            padded_size = next_file._block(next_file.size)
            img_file = obj.read(padded_size)
            try:
                img = Image.open(io.BytesIO(img_file))
            except:
                logger.error("Impossible de lire l'image...")
                continue

            # check for available worker
            sem.acquire()

            chk_running += 1
            end_byte = global_start_byte + obj.tell()
            threading.Thread(target=checkout, args=(img, next_file.name, (start_byte, end_byte))).start()
            start_byte = end_byte

            if global_end_byte and start_byte >= global_end_byte:
                break
        loader_running = False

    loader_thread = threading.Thread(target=loader, args=(global_start_byte,))
    loader_thread.start()

    # send out the content of the output queue
    while True:
        try:
            yield out_queue.get(timeout=0.5)
        except queue.Empty:
            if not loader_running and chk_running <= 0:
                break


def _iterate_s3(function_id, field, query, projection):
    # specify function_id, file_name, (dataset) in the s3 collection query
    query_s3 = {
        function_id: {'$exists': False},
        'field': field
    }
    if query and 'dataset' in query:
        query_s3['dataset'] = query['dataset']

    collection_s3 = mongo_connection[config['project_name'] + "-pipeline-s3"]
    while True:
        # yield every batch, (init start byte to 0)
        doc_s3 = collection_s3.find_one_and_update(query_s3,
                                                   {'$set': {function_id: False}},
                                                   {'_id': 1, 'dataset': 1, 'field': 1})

        if not doc_s3:
            return

        yield from _iterate_batch_s3(doc_s3, function_id, query, projection)

        # set the batch to done
        collection_s3.update_one({'_id': doc_s3['_id']},
                                 {'$set': {function_id: True}})


def _recover_s3_cache(function_id):
    # try to recover aborted uploads from cache
    logger.info("Recovering cache...")

    to_remove = glob.glob(f"{_main_cache}/{function_id}/**/*", recursive=True)
    to_remove = list(map(lambda x: os.path.normpath(x), filter(lambda x: os.path.isfile(x), to_remove)))

    mongo_glob = f"{_mongo_cache}/{function_id}/**/*"
    for doc_file in glob.iglob(mongo_glob, recursive=True):
        if not os.path.isfile(doc_file):
            continue
        dataset = os.path.basename(os.path.split(doc_file)[0])

        # get the path of the file associated with it
        img_file_name = os.path.basename(os.path.splitext(doc_file)[0])
        img_file_path = os.path.normpath(f"{_main_cache}/{function_id}/{dataset}/{img_file_name}")

        # recover the mongo document
        try:
            to_remove.remove(img_file_path)
            with open(doc_file, 'r') as json_file:
                doc = json.load(json_file)
            Image.open(img_file_path).close()
        except (FileNotFoundError, io.UnsupportedOperation, UnidentifiedImageError, ValueError) as e:
            # if something is wrong delete both files
            for rm_file in [doc_file, img_file_path]:
                try:
                    os.remove(rm_file)
                except FileNotFoundError:
                    pass
            continue

        # add its size to the cache size
        file_size = os.stat(img_file_path).st_size

        # multiprocessing
        if 'shared_context' in config:
            fid_dataset = f'{function_id}/{doc["dataset"]}'

            if fid_dataset not in manager._global.cache_size:
                manager._global.cache_size[fid_dataset] = 0

            manager._global.cache_size[fid_dataset] += file_size

        # single process
        else:
            global _cache_size
            if not _cache_size:
                _cache_size = 0
            _cache_size += file_size

    # delete all image files that have no document
    for rm_file in to_remove:
        try:
            os.remove(rm_file)
        except FileNotFoundError:
            pass

    tar_glob = glob.glob(f'{_tar_cache}/{function_id}/**/*.tar')
    logger.info(f"Recovering {len(tar_glob)} tar batches...")
    for tar_path in tar_glob:
        threading.Thread(target=_upload_tar_batch, args=(tar_path, True)).start()

    logger.success("Done recovering cache")


def _upload_tar_batch(tar_path, recovery=False):
    dataset = os.path.basename(os.path.split(tar_path)[0])
    tar_name = os.path.basename(tar_path)
    file_name_field = tar_name.split('!')[0]

    # upload tar file
    distant_key = "/".join([dataset, tar_name])
    logger.info(f'Uploading : {distant_key}')

    _s3_client.upload_file(tar_path, config['s3']['bucket'], distant_key)

    tar_docs_path = os.path.splitext(tar_path)[0]

    def clean_up(error=False):
        if error:
            logger.warning(f"Failed to recover batch tar file {tar_path}")
        try:
            os.remove(tar_path)
        except Exception:
            pass
        try:
            shutil.rmtree(tar_docs_path)
        except Exception:
            pass

    if recovery:
        # get the names of all the files stored in the archive
        try:
            loaded_tar = tarfile.open(tar_path)
            tar_names = loaded_tar.getnames()
            loaded_tar.close()
        except Exception:
            clean_up()
            return

        # check if they match the mongo documents
        for doc_path in glob.iglob(f'{tar_docs_path}/*'):
            img_name = os.path.splitext(doc_path)[0]

            try:
                tar_names.remove(img_name)
            except ValueError:
                clean_up()
                return

        if len(tar_names) > 0:
            clean_up()
            return

    pbar = tqdm(glob.glob(f'{tar_docs_path}/*'))
    pbar.set_description_str(f"Updating database for {distant_key}...")
    with ThreadPoolExecutor(max_workers=250) as workers:
        for doc_path in pbar:
            doc = json.load(open(doc_path, 'r'))
            workers.submit(_write_document, doc)

    # update the s3 pipeline collection
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

    logger.success(f'Done uploading : {distant_key}')
    clean_up()


def _write_image(document, image_file, function_id, file_name_field=None):
    if 'dataset' not in document:
        raise Exception('Every uploaded file must be linked to a dataset')

    dataset = document['dataset']

    # if no file name field is provided use _id
    if not file_name_field:
        file_name_field = '_id'
    if file_name_field not in document:
        document[file_name_field] = _random_name()
    file_name = document[file_name_field]

    # save file as data/_cache/<function_id>/<dataset>/<file_name>
    full_path = f'{_main_cache}/{function_id}/{dataset}'
    file_path = os.path.join(full_path, file_name)
    os.makedirs(full_path, exist_ok=True)

    try:
        image_file.save(file_path)
    except Exception as e:
        return

    # check if files exists
    try:
        file_size = os.path.getsize(file_path)
    except FileNotFoundError:
        return

    # save mongo update request in a file for easy recovery
    mongo_full_path = f'{_mongo_cache}/{function_id}/{dataset}'
    os.makedirs(mongo_full_path, exist_ok=True)

    mongo_file_name = file_name + '.json'
    mongo_file_path = os.path.join(mongo_full_path, mongo_file_name)
    json.dump(document, open(mongo_file_path, 'w'))

    # increment cache size
    # return if cache_size < flush_size
    # otherwise block access to cache and flush it
    if 'shared_context' in config:
        # lock shared cache_size
        manager._global.cache_lock.acquire()

        fid_dataset = f'{function_id}/{dataset}'
        if fid_dataset not in manager._global.cache_size:
            manager._global.cache_size[fid_dataset] = 0

        manager._global.cache_size[fid_dataset] += file_size

        if manager._global.cache_size[fid_dataset] < config['s3']['flush_size']:
            manager._global.cache_lock.release()
            return
    else:
        global _cache_size
        _cache_lock.acquire()
        if _cache_size is None:
            _cache_size = file_size
        else:
            _cache_size += file_size
            if _cache_size < config['s3']['flush_size']:
                _cache_lock.release()
                return

    # ---------- flush cache -------------

    logger.info('Flushing batch into s3...')

    # make a tarfile and send it to s3
    # update all concerned collections :
    # - dataset collection
    # - pipeline collection
    # - s3 pipeline collection

    # fill the tar file with name '<md5(_id)>.tar'
    # created in data/_output_tar
    # remove all transferred files

    # tar path
    tar_prefix = file_name_field + "!"
    tar_name = tar_prefix + hashlib.md5(document['_id'].encode()).hexdigest() + ".tar"
    tar_cache = f"{_tar_cache}/{function_id}/{dataset}"
    os.makedirs(tar_cache, exist_ok=True)
    tar_path = os.path.join(tar_cache, tar_name)

    # doc path
    docs_tar_path = os.path.splitext(tar_path)[0]
    os.makedirs(docs_tar_path, exist_ok=True)

    batch_file = tarfile.TarFile(tar_path, "w")
    for img_path in glob.glob(f"{_main_cache}/{function_id}/{dataset}/*"):
        img_name = os.path.basename(img_path)

        # get document file path
        doc_path = f"{_mongo_cache}/{function_id}/{dataset}/{img_name}.json"
        if not os.path.isfile(img_path) or not os.path.isfile(doc_path):
            continue

        # check if the files are still being written
        try:
            os.rename(img_path, img_path)
            os.rename(doc_path, doc_path)
        except OSError:
            continue

        # add the file
        mongo_doc = json.load(open(doc_path, 'r'))
        mongo_doc[f'_tar_{file_name_field}'] = tar_name
        mongo_doc[f'_pos_{file_name_field}'] = batch_file.offset
        json.dump(mongo_doc, open(f'{docs_tar_path}/{img_name}.json', 'w'))

        batch_file.add(img_path, img_name)
        os.remove(img_path)
    batch_file.close()

    # resume other processes work
    if manager._global:
        manager._global.cache_size[f'{function_id}/{dataset}'] = 0
        manager._global.cache_lock.release()
    else:
        _cache_size = 0
        _cache_lock.release()

    # upload the tar file on the side and resume work
    threading.Thread(target=_upload_tar_batch, args=(tar_path,)).start()


def _delete_s3_batch(s3_batch, sem, pbar, full):
    try:
        _s3_client.delete_object(Bucket=config['s3']['bucket'],
                                 Key=f'{s3_batch["dataset"]}/{s3_batch["_id"]}')
    except:
        pass

    # delete the tar name field
    unset_fields = {
        f'_tar_{s3_batch["field"]}': '',
        f'_pos_{s3_batch["field"]}': ''
    }
    # if the field is not _id, delete the file name field
    if s3_batch['field'] != '_id' and full:
        unset_fields[s3_batch['field']] = ''

    # dataset
    mongo_connection[f'{s3_batch["dataset"]}-dataset-s3'].delete_one({'_id': s3_batch['_id']})
    mongo_connection[f'{s3_batch["dataset"]}-dataset'].update_many(
        {f'_tar_{s3_batch["field"]}': s3_batch['_id']},
        {'$unset': unset_fields})

    # local
    mongo_connection[f'{config["project_name"]}-pipeline-s3'].delete_one({'_id': s3_batch['_id']})
    mongo_connection[f'{config["project_name"]}-pipeline'].update_many(
        {f'_tar_{s3_batch["field"]}': s3_batch['_id']},
        {'$unset': unset_fields})

    pbar.update()
    sem.release()


def _delete_field(dataset, field, full):
    # list all available batches related to this field in the dataset collection
    # and the project collection
    s3_batches = {}

    for batch in mongo_connection[f'{dataset}-dataset-s3'].find({'field': field}):
        s3_batches[batch['_id']] = batch

    for batch in mongo_connection[f'{config["project_name"]}-pipeline-s3'].find({'field': field, 'dataset': dataset}):
        s3_batches[batch['_id']] = batch

    sem = threading.Semaphore(100)
    pbar = tqdm(total=len(s3_batches), desc=f"Deleting {len(s3_batches)} batches on s3...")
    threads = []
    for s3_batch in s3_batches.values():
        sem.acquire()
        thread = threading.Thread(target=_delete_s3_batch, args=(s3_batch, sem, pbar, full))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
