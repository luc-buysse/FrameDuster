import inspect
import json
import multiprocessing
import os
import threading

from processes import _Processes
from threads import _Threads
from outputs import _Output
from inputs import _Input
from load import _Load
from train import _Train
from clerk import _Clerk

from mongodb import _index_manager, _global_pbar_thread, _local_pbar_thread
from config import config


def _topping_transformer(name, args, kwargs):
    def transformer(input):
        dec_layer = [(name, args, kwargs)]
        if inspect.isfunction(input) or inspect.isclass(input):
            return (json.dumps(dec_layer), input)
        else:
            loaded_decorators = json.loads(input[0])
            return (json.dumps(dec_layer + loaded_decorators), input[1])

    return transformer


def Output(output_type,
           compartment='_id',
           max_threads=1,
           buffer_size=1,
           stream=False,
           pbar=False):
    """
    Stores or transfers data yielded by the main function of the service.

    :param output_type: Among ('database', 's3', 'local', 'pipe'). Specifies the destination of the data.
    :param compartment: Specific to 's3' and 'local'. Compartment to store the files into.
    :param max_threads: The maximum number of threads to use for saving operations.
    :param buffer_size: Specific to 'database'. The size of the buffer to use for MongoDB bulk_write operations.
    :param stream: Specific to 's3'. Whether to stream data to s3 directly and patch up remotely or store it temporarily in a local cache.
    :param pbar: True to activate a progress bar (local output only).
    :return:
    """

    args = [output_type, compartment]
    kwargs = {
        'max_threads': max_threads,
        'buffer_size': buffer_size,
        'stream': stream,
        'pbar': pbar
    }
    return _topping_transformer("Output", args, kwargs)


def Load(data,
         batch_size,
         composition=None,
         split=None,
         target=None,
         num_workers=os.cpu_count() - 1,
         preprocess=False,
         source=None,
         query=None,
         pick=False,
         shuffle_size=1000,
         slice_size=50):
    """
    Loads data in the form of torch Dataloader(s).

    (M): Always mandatory.
    (T): Mandatory for training.

    :param data: The field of the data you want to load. (M)
    :param batch_size: The batch size of the Dataloader(s) (M)
    :param composition: The number of elements you want to load from each dataset. (M)
    :param split: The ratios (Train, Validation, Test) you want to use. (T)
    :param target: The field which data should be used as target. (T)
    :param num_workers: The number of workers in the Dataloader.
    :param preprocess: If you want to use a preprocess function.
    :param source: The source of the data. By default, the data is fetched from the database
        and provided as it is. Possible sources are : ('s3', 'local', 'database')
    :param query: A query to filter data. With source='s3', consider using pick=True.
    :param pick: Specific to source=s3. Set to True, each file will be fetched individually.
    :param shuffle_size: Specific to [source=s3, pick=False]. The number of files in the shuffle buffer.
    :param slice_size:  Specific to [source=s3, pick=False]. The maximum number of files in each slice.
    :return: A Dataloader or a 3-tuple of Dataloaders (Train, Validation, Test).
    """

    kwargs = {
        'data': data,
        'batch_size': batch_size,
        'composition': composition,
        'split': split,
        'target': target,
        'num_workers': num_workers,
        'preprocess': preprocess,
        'source': source,
        'query': query,
        'pick': pick,
        'shuffle_size': shuffle_size,
        'slice_size': slice_size
    }

    return _topping_transformer("Load", [], kwargs)


def Input(input_type,
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
          pbar=False):
    """
    Loads data into an iterator passed to the main function as first argument.

    :param input_type: Specifies the origin of the data. Can be 'database', 's3', 'local', 'pipe' or 'feed'.
    :param compartment: For s3 and local, compartment of the data.
    :param query: A MongoDB query to filter the data.
    :param projection: A MongoDB projection to limit the data.
    :param preload: The number of elements to preload. Smoothens the iterator but takes up RAM.
    :param pick: For s3, select individual files instead of batches. Can be useful in combination with a query.
    :param index_query: Whether to index the fields of the query.
    :param threads: Number of loader threads to run.
    :param free_runner: Set to True triggers turbo mode, but your pool of machines must be synchronized manually using slice_index and slice_count.
    :param slice_count: The number of machines in your pool.
    :param slice_index: The id of the current machine within : [0, slice_count[.
    :param limit: Optional. The maximum number of elements you want to load using your pool of free_runners.
    :param mongo_buffer: The size of the Mongo batches to use with your free_runners.
    :param pbar: True to activate a progress bar (local input and global progress).
    :return: An iterator on the data : documents for 'database', (document, file) tuples for 's3' and 'local', any custom form of data for 'pipe' and 'feed'.
    """
    args = [input_type, compartment]
    kwargs = {
        'query': query,
        'projection': projection,
        'preload': preload,
        'pick': pick,
        'index_query': index_query,
        'threads': threads,
        'free_runner': free_runner,
        'slice_count': slice_count,
        'slice_index': slice_index,
        'limit': limit,
        'mongo_buffer': mongo_buffer,
        'pbar': pbar
    }

    return _topping_transformer("Input", args, kwargs)


def Processes(count):
    """
    Executes the main function and all subdecorators within separate processes.

    :param count: Number of processes.
    :return: Nothing.
    """
    args = [count]
    return _topping_transformer("Processes", args, {})


def Threads(count):
    """
    Executes the main function and all subdecorators within separate processes.

    :param count: Number of threads.
    :return: Nothing.
    """
    args = [count]
    return _topping_transformer("Threads", args, {})


_decode_decorator = {
    "Threads": _Threads,
    "Processes": _Processes,
    "Output": _Output,
    "Input": _Input,
    "Load": _Load,
    "Train": _Train,
    "Clerk": _Clerk
}


def _setup_mixed_context():
    # Intialize mixed context
    config['mixed_context'] = {}

    # Launch the index manager (use a multiprocessing queue, speed is not required here)
    config['mixed_context']['index_request_queue'] = multiprocessing.Queue()
    threading.Thread(target=_index_manager, daemon=True).start()

    config['mixed_context']['pbar'] = {
        'input': False,
        'output': False
    }

    config['mixed_context']['count_docs_value'] = multiprocessing.Value('d', -1)


def _setup_pbar(type):
    config['mixed_context'][f'pbar_{type}_queue'] = multiprocessing.Queue()
    config['mixed_context']['pbar'][type] = True
    threading.Thread(target=_local_pbar_thread, args=(type,)).start()

    if type == 'input':
        config['mixed_context']['pbar_query_queue'] = multiprocessing.Queue()
        threading.Thread(target=_global_pbar_thread).start()


def _add_decorators(decorators_str, func):
    # Initialize if needed before launching the pipeline and potentially forking
    if inspect.isclass(func):
        func = func()

    return __add_decorators(decorators_str, func)


def __add_decorators(decorators_str, func):
    decorators = json.loads(decorators_str)

    if len(decorators) == 0:
        return func

    dec = _decode_decorator[decorators[0][0]]
    dec_args = decorators[0][1]
    dec_kwargs = decorators[0][2]

    if dec == 'Input' and dec_args[0] not in ('feed', 'pipe'):
        threading.Thread(target=_local_pbar_thread).start()
        threading.Thread(target=_global_pbar_thread).start()

    next_decorators_str = json.dumps(decorators[1:])

    sub_proc_tuple = (next_decorators_str, func)
    return dec(*dec_args, **dec_kwargs)(__add_decorators(*sub_proc_tuple), sub_proc_tuple)
