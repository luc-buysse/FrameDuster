import re
import threading
import time

from loguru import logger
from pymongo import MongoClient
from pymongo import uri_parser, UpdateOne
from tqdm import tqdm
import math

from lib.config import config


def _get_mongo_connection():
    mongo_uri = config['mongo']['uri']
    parsed_uri = uri_parser.parse_uri(mongo_uri)
    database_name = parsed_uri["database"]
    client = MongoClient(mongo_uri)

    return client[database_name]

def get_sub_cursor(collection, query, total_vec, index_vec, doc_count, projection=None, batch_size=101):
    slice_size = doc_count
    pos = 0
    for total_slices, slice_index in zip(total_vec, index_vec):
        slice_size /= total_slices
        pos += slice_index * slice_size

    pos = math.floor(pos)
    slice_size = math.floor(slice_size)

    logger.info(f'Projection : {projection}')

    if projection:
        return collection.find(query, projection, batch_size=batch_size).skip(pos).limit(slice_size)
    else:
        return collection.find(query, batch_size=batch_size).skip(pos).limit(slice_size)


mongo_connection = _get_mongo_connection()
_bulks = {'__main__': []}


def _write_document(doc, bulk_size=1):
    to_unset = {}
    if '__delete__' in doc:
        to_unset = {
            field: '' for field in doc['__delete__']
        }

    if bulk_size > 1:
        if 'dataset' in doc:
            dataset = doc['dataset']

            if dataset not in _bulks:
                _bulks[dataset] = []

            dataset_doc = {}
            for key in doc:
                if key[0] != '_':
                    dataset_doc[key] = doc[key]
            dataset_doc['_id'] = doc['_id']

            # add to dataset bulk
            _bulks[dataset].append(UpdateOne({
                '_id': doc['_id']
            }, {
                '$set': dataset_doc,
                '$unset': to_unset
            }, upsert=True))

            # write the bulk if sufficiently large
            if len(_bulks[dataset]) > bulk_size:
                mongo_connection[dataset + "-dataset"].bulk_write(_bulks[dataset])

        # add to pipeline bulk
        _bulks['__main__'].append(UpdateOne({
            '_id': doc['_id']
        }, {
            '$set': doc,
            '$unset': to_unset
        }, upsert=True))

        # write the bulk if sufficiently large
        if len(_bulks['__main__']) > bulk_size:
            mongo_connection[config['project_name'] + "-pipeline"].bulk_write(_bulks['__main__'])
        return

    # update dataset collection
    if 'dataset' in doc:
        dataset_doc = {}
        for key in doc:
            if key[0] != '_' or re.match('(_tar_.*)|(_local_.*)', key):
                dataset_doc[key] = doc[key]
        dataset_doc['_id'] = doc['_id']
        mongo_connection[doc['dataset'] + "-dataset"].update_one({
            '_id': doc['_id']
        }, {
            '$set': dataset_doc,
            '$unset': to_unset
        }, upsert=True)

    # update pipeline collection
    mongo_connection[config['project_name'] + "-pipeline"].update_one({
        '_id': doc['_id']
    }, {
        '$set': doc,
        '$unset': to_unset
    }, upsert=True)


def _mark_done(doc_id, function_id):
    mongo_connection[config['project_name'] + '-pipeline'].update_one(
        {'_id': doc_id},
        {'$set': {function_id: 2}}
    )


def _reset_function(function_id, full=False, delete=False):
    # a "simple reset" only switches the function_id
    # marked False (being processed) back to 'non existing'
    # this way only the elements that were not successfully
    # or completely processed will be processed when the
    # function is called again

    # a simple reset also allows to process the elements of
    # batches that were once open but not completely read

    service_type = function_id[1:function_id[1:].find('_') + 1]
    service_name = function_id[function_id[1:].find('_') + 2:]
    if not full and not delete:
        logger.info(f"Reseting unfinished items of {service_type} {service_name}...")
    elif not delete:
        logger.info(f"Reseting all progress of {service_type} {service_name}")
    else:
        logger.info(f"Removing all tracks of {service_type} {service_name}")

    if delete:
        mongo_connection[config['project_name'] + '-pipeline'].update_many({function_id: {'$exists': True}},
                                                                           {'$unset': {function_id: ""}})
        mongo_connection[config['project_name'] + '-pipeline-s3'].update_many({function_id: {'$exists': True}},
                                                                           {'$unset': {function_id: ''}})
    else:

        mongo_connection[config['project_name'] + '-pipeline'].update_many({function_id: 1},
                                                                           {'$set': {function_id: 0}})
        mongo_connection[config['project_name'] + '-pipeline-s3'].update_many({function_id: False},
                                                                           {'$unset': {function_id: ''}})

        # for a full reset, all function ids are destroyed
        # as well as batches progresses
        if full:
            mongo_connection[config['project_name'] + '-pipeline'].update_many({function_id: 2},
                                                                               {'$set': {function_id: 0}})
            mongo_connection[config['project_name'] + '-pipeline-s3'].update_many({},
                                                                               {'$unset': {function_id: '',
                                                                                            function_id + '_sb': ""}})

    logger.success(f"Done reseting {service_type} {service_name}")


def _state_function(function_id):
    service_type = function_id[1:function_id[1:].find('_')+1]
    service_name = function_id[function_id[1:].find('_')+2:]
    logger.info(f"Computing state for : {service_type} {service_name}")

    collection = mongo_connection[config['project_name'] + '-pipeline']
    collection_s3 = mongo_connection[config['project_name'] + '-pipeline-s3']

    main_state = (
        collection.count_documents({function_id: 0}),
        collection.count_documents({function_id: 1}),
        collection.count_documents({function_id: 2})
    )
    s3_state = (
        collection_s3.count_documents({function_id: {'$exists': False}}),
        collection_s3.count_documents({function_id: False}),
        collection_s3.count_documents({function_id: True})
    )

    logger.info(f"Main : {main_state} s3 : {s3_state}")


_index_cache = set()
def _ensure_index(fields):
    """
    This function ensures that the index associated with 'fields' exists.
    """
    index_name = '_1_'.join(sorted(fields)) + '_1'

    # Check the local cache
    if index_name not in _index_cache:
        # If the index is not in the local thread cache, ask the index manager for a global check
        config['mixed_context']['index_request_queue'].put(fields)
        # Validate index
        _index_cache.add(index_name)


def _index_manager():
    main_collection = mongo_connection[f'{config["project_name"]}-pipeline']

    local_state = set(map(lambda x: x['name'], main_collection.list_indexes()))

    def add_index_thread(fields):
        logger.info(f"Adding index for {fields}...")
        main_collection.create_index(
            [(field, 1) for field in fields]
        )
        logger.info(f"Done adding index for {fields}")

    while True:
        fields = config['mixed_context']['index_request_queue'].get()

        # End of the program
        if not fields:
            return

        index_name = '_1_'.join(sorted(fields)) + '_1'

        if index_name in local_state:
            continue

        local_state.add(index_name)
        threading.Thread(target=add_index_thread, args=(fields,)).start()


def _local_pbar_thread(stream_type):
    pos = 1 if stream_type == 'input' else 3

    local_pbar = tqdm(position=pos, colour='00ff00')
    local_pbar.set_description_str(f'Progress {stream_type} ')
    while True:
        config['mixed_context'][f'pbar_{stream_type}_queue'].get()
        local_pbar.update()


def _global_pbar_thread():
    function_id, query = config['mixed_context']['pbar_query_queue'].get()

    global_pbar = tqdm(position=2)
    global_pbar.set_description_str("Loading global progress...")

    to_do_count = mongo_connection[f'{config["project_name"]}-pipeline'].count_documents(query)
    done_query = query.copy()
    done_query[function_id] = 2
    done_count = mongo_connection[f'{config["project_name"]}-pipeline'].count_documents(done_query)

    global_pbar.total = to_do_count + done_count
    global_pbar.close()
    global_pbar = tqdm(position=2,
                       desc='Global progress',
                       initial=done_count,
                       total=done_count+to_do_count)

    while True:
        # update the global pbar every 10 sec + time of the request
        time.sleep(config['pbar_refresh_rate'])
        new_done_count = mongo_connection[f'{config["project_name"]}-pipeline'].count_documents(done_query)

        global_pbar.update(new_done_count - done_count)
        done_count = new_done_count
