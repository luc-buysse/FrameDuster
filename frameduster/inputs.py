from loguru import logger

import threading
import queue

from frameduster.iterable import _generator
from frameduster.mongodb import mongo_connection, _ensure_index, get_sub_cursor
from frameduster.s3 import _iterate_s3, _get_image
from frameduster.local import _iterate_local, _get_image_local
from frameduster.config import config, _local


class _mail_box:
    """
    Merge multiple iterators into one, running each one within a separate thread and gathering their content.
    """

    def __init__(self, iterators, preload):
        # a decent preload is necessary to benefit from the multiple threads
        preload = max(preload, len(iterators))

        self.it = iterators
        self.box = queue.Queue()
        self.go = threading.Semaphore(preload)

        for it in iterators:
            threading.Thread(target=self.grinder, args=(it,), daemon=True).start()

    def grinder(self, it):
        while True:
            self.go.acquire()
            try:
                item = next(it)
            except StopIteration:
                self.box.put(None)
                return
            self.box.put(item)

    def __iter__(self):
        while True:
            item = self.box.get()
            self.go.release()

            if not item:
                return
            else:
                yield item



def _Input(
        # user args
        input_type,
        compartment=None,
        query=None,
        projection=None,
        preload=10,
        pick=False,
        index_query=True,
        threads=1,
        free_runner=False,
        slice_index=0,
        slice_count=1,
        limit=None,
        mongo_buffer=101,

        # lib args
        dataset_name=None,
        function_id="__default__"):
    def transformer(func, sub_proc_wrapped):
        def wrapper(*args, **kwargs):

            nonlocal compartment
            nonlocal query
            nonlocal projection
            nonlocal pick

            # Setup for storage input types
            if input_type in ('s3', 'database', 'local'):
                # Ensure index for function_id
                _ensure_index([function_id])

                # Initialize function id with 'non-processed' (0) value
                threading.Thread(target=mongo_connection[f'{config["project_name"]}-pipeline'].update_many, args=(
                    {function_id: {'$exists': False}},
                    {'$set': {function_id: 0}}
                )).start()

                # Add required fields to the projection
                if projection:
                    projection['_id'] = 1
                    projection['dataset'] = 1

                # Add required fields to the query
                if not query:
                    query = {}

                if dataset_name:
                    query['dataset'] = dataset_name
                query[function_id] = 0

                if input_type == 's3':
                    query[f'_tar_{compartment}'] = {'$exists': True}
                    if pick:
                        query[f'_pos_{compartment}'] = {'$exists': True}

                if input_type == 'local':
                    query[f'_local/{config["machine_id"]}/{compartment}'] = True

                # Now that the query is full, setup the global progress bar
                if (config['mixed_context']['pbar_input_service'] == function_id and
                        config['mixed_context']['pbar']['input']):
                    logger.info(f'Query for progress : {query}')
                    config['mixed_context']['pbar_query_queue'].put(
                        (function_id, query))

                # ************* Setup Indexes *************

                # function_id : almost always necessary for decent speed
                index_fields = {function_id}

                # dataset for scrapers
                if dataset_name:
                    index_fields.add('dataset')

                # query if asked by the user
                if index_query:
                    for field in query:
                        index_fields.add(field)

                # _tar indicator for s3 pick
                if input_type == 's3':
                    index_fields.add(f'_tar_{compartment}')

                # _local indicator for local
                if input_type == 'local':
                    index_fields.add(f'_local/{config["machine_id"]}/{compartment}')

                _ensure_index(list(index_fields))

            # Free runner setup
            if free_runner and input_type in ('s3', 'database', 'local'):
                # Get the total number of documents to process (make it only in one thread/process)
                if limit is None:
                    with config['mixed_context']['count_docs_value']:
                        # Compute it if needed
                        if config['mixed_context']['count_docs_value'].value == -1:
                            logger.info("Computing total number of items to process...")
                            config['mixed_context']['count_docs_value'].value = (
                                mongo_connection[f'{config["project_name"]}-pipeline'].count_documents(query))

                        total_docs = config['mixed_context']['count_docs_value'].value
                else:
                    total_docs = limit

                # Make a subcursor from 3 possible levels :
                # - multimachine (provided with @Input args 'slice_count', 'slice_index')
                # - multithread/multiprocess from @Processes/@Threads
                # - local multithreading
                free_cursors = []
                total_vec = [slice_count, _local.total_slices, threads]
                index_vec = [slice_index, _local.process_slice, 0]
                for i in range(threads):
                    index_vec[2] = i
                    sub_cursor = get_sub_cursor(
                        mongo_connection[config['project_name'] + "-pipeline"],
                        query,
                        total_vec,
                        index_vec,
                        total_docs,
                        projection,
                        mongo_buffer,
                    )
                    free_cursors.append(sub_cursor)

            # Set the default compartment for file storages
            if input_type in ['s3', 'local']:
                # default compartment is _id
                if not compartment:
                    compartment = '_id'

            # **** create the input generator ****

            if input_type == "database":
                if not free_runner:
                    # update function_id
                    update = {'$set': {function_id: 1}}

                    query_args = [query, update]
                    if projection:
                        query_args.append(projection)

                    def gen_func():
                        # yield all documents
                        while True:
                            next_item = (mongo_connection[config['project_name'] + "-pipeline"]
                                         .find_one_and_update(*query_args))
                            if not next_item:
                                return
                            yield next_item

                    generators = [gen_func() for _ in range(threads)]
                    input_generator = _mail_box(generators, preload)
                else:
                    input_generator = _mail_box(free_cursors, preload)

            elif input_type == "s3":
                if pick:
                    # update function_id
                    update = {
                        '$set': {function_id: 1},
                    }

                    query_args = [query]
                    if not free_runner:
                        query_args.append(update)
                    if projection:
                        projection[f'_tar_{compartment}'] = True
                        projection[f'_pos_{compartment}'] = True
                        query_args.append(projection)

                    if not free_runner:
                        def gen_func():
                            # yield all documents
                            while True:
                                next_item = (mongo_connection[config['project_name'] + "-pipeline"]
                                             .find_one_and_update(*query_args))
                                if not next_item:
                                    return
                                yield next_item, _get_image(next_item, compartment)

                        generators = [gen_func() for _ in range(threads)]
                    else:
                        def gen_func(cursor):
                            for next_item in cursor:
                                yield next_item, _get_image(next_item, compartment)

                        generators = [gen_func(cursor) for cursor in free_cursors]

                    input_generator = _mail_box(generators, preload)
                else:
                    generators = [_iterate_s3(function_id, compartment, query, projection) for _ in range(threads)]
                    input_generator = _mail_box(generators, preload)

            elif input_type == 'local':
                if not free_runner:
                    generators = [_iterate_local(function_id, compartment, query, projection) for _ in range(threads)]
                else:
                    def gen_func(cursor):
                        for next_item in cursor:
                            yield next_item, _get_image_local(next_item, compartment)

                    generators = [gen_func(cursor) for cursor in free_cursors]
                input_generator = _mail_box(generators, preload)

            elif input_type == "feed":
                raw_func = sub_proc_wrapped[1]
                if not hasattr(raw_func, '__feed__'):
                    raise Exception('Input("feed") specified but no __feed__ method provided')
                input_generator = raw_func.__feed__()

            elif input_type == "pipe":
                yield from _generator(func, *args)
                return
            else:
                raise Exception("Invalid argument for Input")

            yield from _generator(func, input_generator)

        return wrapper
    return transformer
