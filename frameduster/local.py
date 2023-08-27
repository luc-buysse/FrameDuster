import os
import threading

from PIL import Image
from loguru import logger
from tqdm import tqdm

from frameduster.config import config
from frameduster.mongodb import _write_document, mongo_connection, _ensure_index
from frameduster.config import _random_name

_local_count = {}
_local_collection = mongo_connection[f'{config["project_name"]}-pipeline-local']


def _iterate_local(function_id, field, query=None, projection=None):
    if not query:
        query = {}
    query[function_id] = {'$exists': False}
    query[f'_local/{config["machine_id"]}/{field}'] = True

    main_collection = mongo_connection[config['project_name'] + "-pipeline"]
    query_args = [query, {'$set': {function_id: 1}}]
    if projection:
        query_args.append(projection)
    while True:
        doc = main_collection.find_one_and_update(*query_args)
        if not doc:
            return
        file_path = f'{config["dataset_path"]}/{doc["dataset"]}/{field}/{doc[field]}'
        yield doc, Image.open(file_path)


def _get_image_local(doc, field):
    file_path = f'{config["dataset_path"]}/{doc["dataset"]}/{field}/{doc[field]}'
    yield doc, Image.open(file_path)


def _write_local(document, ws, file_name_field=None):
    global _local_count

    if 'dataset' not in document:
        raise Exception('Every saved file must be linked to a dataset')
    dataset = document['dataset']

    # create index for further operations on the local storage
    _ensure_index(['dataset', f'_local/{config["machine_id"]}/{file_name_field}'])

    # if no file name field is provided use _id
    if not file_name_field:
        file_name_field = '_id'
    if file_name_field not in document:
        document[file_name_field] = _random_name()

    # save the file in <local_storage_path>/<dataset>/<compartment>/<file_name>
    full_path = f'{config["dataset_path"]}/{document["dataset"]}/{file_name_field}'
    os.makedirs(full_path, exist_ok=True)
    file_path = f'{config["dataset_path"]}/{document["dataset"]}/{file_name_field}/{document[file_name_field]}'
    ws.save(file_path)

    # update the local counts and sync with mongoDB
    ds_fnf_field = f'field/{dataset}/{file_name_field}'
    if ds_fnf_field not in _local_count:
        storage_doc = _local_collection.find_one({'_id': config['machine_id']})
        if not storage_doc or ds_fnf_field not in storage_doc:
            _local_count[ds_fnf_field] = 0
        else:
            _local_count[ds_fnf_field] = storage_doc[ds_fnf_field]
    _local_count[ds_fnf_field] += 1

    # project collection
    _local_collection.update_one({'_id': config['machine_id']},
                                 {'$set': {ds_fnf_field: _local_count[ds_fnf_field]}},
                                 upsert=True)
    # dataset collection
    mongo_connection[f'{document["dataset"]}-dataset-local'].update_one({'_id': config['machine_id']},
                                                                        {'$set': {ds_fnf_field: _local_count[ds_fnf_field]}},
                                                                        upsert=True)

    document[f'_local/{config["machine_id"]}/{file_name_field}'] = True
    _write_document(document)


def _ensure_file_local(dataset, field, file_name, sem):
    mongo_connection[f'{config["project_name"]}-pipeline'].update_one({field: file_name},
                                                                      {'$set': {
                                                                          f'_local/{config["machine_id"]}/{field}': True
                                                                      }})
    mongo_connection[f'{dataset}-dataset'].update_one({field: file_name},
                                                          {'$set': {
                                                              f'_local/{config["machine_id"]}/{field}': True
                                                          }})
    sem.release()


def _update_local():
    sem = threading.Semaphore(1000)

    machine_id = config["machine_id"]

    ds_fnf_set = set()
    local_doc = mongo_connection[f'{config["project_name"]}-pipeline-local'].find_one({
        '_id': machine_id
    })
    if local_doc:
        for field in local_doc:
            if field == '_id':
                continue
            ds_fnf_set.add(field[6:])

    for dataset in os.listdir(config['dataset_path']):
        if dataset == '_machine_id':
            continue
        for file_name_field in os.listdir(f'{config["dataset_path"]}/{dataset}'):
            ds_fnf_set.add(f'{dataset}/{file_name_field}')

    if len(ds_fnf_set) == 0:
        logger.info(f"Nothing to update for {machine_id}")
        return

    logger.info(f"Updating : {','.join(ds_fnf_set)}")

    for ds_fnf in ds_fnf_set:
        dataset, file_name_field = ds_fnf.split('/')

        down_list = mongo_connection[f'{config["project_name"]}-pipeline'].find({
            f'_local/{machine_id}/{file_name_field}': True,
            'dataset': dataset
        }, {'_id': True, file_name_field: True})

        down_dict = {item[file_name_field]: item["_id"] for item in down_list}

        # run through local files, add to DB if needed
        added_count = 0
        file_list = []
        try:
            file_list = os.listdir(f'{config["dataset_path"]}/{dataset}/{file_name_field}')
        except FileNotFoundError:
            pass
        else:
            pbar = tqdm(file_list)
            pbar.set_description_str(f'Scanning local storage {dataset}/{file_name_field}...')
            for file_name in pbar:
                # remove from down list
                try:
                    del down_dict[file_name]
                except KeyError:
                    added_count += 1
                    # if the file is not on the DB, update it
                    sem.acquire()
                    threading.Thread(target=_ensure_file_local, args=(dataset, file_name_field, file_name, sem)).start()

        # Delete all files that were not found locally
        def delete_one(_id):
            mongo_connection[f'{config["project_name"]}-pipeline'].update_one(
                {'_id': _id},
                {'$unset': {f'_local/{machine_id}/{file_name_field}': ''}}
            )
            sem.release()

        deleted_count = len(down_dict)

        pbar = tqdm(down_dict.values())
        pbar.set_description_str(f'Cleaning database for {dataset}/{file_name_field}...')
        for _id in pbar:
            sem.acquire()
            threading.Thread(target=delete_one, args=(_id,)).start()

        # update file count in the local collection
        _local_collection.update_one({'_id': config['machine_id']},
                                     {'$set': {f'field/{dataset}/{file_name_field}': len(file_list)}},
                                     upsert=True)
        mongo_connection[f'{dataset}-dataset-local'].update_one({'_id': config['machine_id']},
                                     {'$set': {f'field/{dataset}/{file_name_field}': len(file_list)}},
                                    upsert=True)

        logger.info(f'Done updating {dataset}/{file_name_field}'
                       f'\n\tDeleted : {deleted_count}'
                       f'\n\tAdded : {added_count}'
                       f'\n\tCurrent : {len(file_list)}')


def _unbind_local(machine_id):
    logger.info(f'Unbinding local storage : {machine_id}')

    storage_doc = mongo_connection[f'{config["project_name"]}-pipeline-local'].find_one({'_id': machine_id})
    if not storage_doc:
        return

    # delete all element documents' fields related to this storage location
    datasets = []
    for dataset_field in storage_doc:
        if dataset_field == '_id':
            continue

        dataset, field = dataset_field.split('/')[1:]
        datasets.append(dataset)

        field_to_delete = f'_local/{machine_id}/{field}'

        logger.info(f"Deleting : {machine_id}/{field}")

        mongo_connection[f'{config["project_name"]}-pipeline'].update_many({field_to_delete: True},
                                                                                 {'$unset': {field_to_delete: ''}})
        mongo_connection[f'{dataset}-dataset'].update_many({field_to_delete: True},
                                                                {'$unset': {field_to_delete: ''}})

    # delete local state description documents in project collection and dataset collections
    mongo_connection[f'{config["project_name"]}-pipeline-local'].delete_one({'_id': machine_id})
    for dataset in datasets:
        mongo_connection[f'{dataset}-dataset-local'].delete_one({'_id': machine_id})




def _scan_local():
    for doc in mongo_connection[f'{config["project_name"]}-pipeline-local'].find({}):
        state_str = f'Result of scan for storage {doc["_id"]} :\n'
        for field in doc:
            if field == '_id':
                continue
            state_str += f'\t{field[6:]} : {doc[field]}\n'
        logger.info(state_str)


def _set_local(new_id):
    if new_id == config["machine_id"]:
        logger.info(f'Local storage ID is already {new_id}, nothing to update.')
        return

    # remove all trace on db
    _unbind_local(config['machine_id'])

    logger.info(f'Setting local storage name to : {new_id}')

    # change the content of the identification file
    mid_path = os.path.join(config['dataset_path'], '_machine_id')
    with open(mid_path, 'w') as f:
        f.write(new_id)

    # update with new id
    config["machine_id"] = new_id
    _update_local()
