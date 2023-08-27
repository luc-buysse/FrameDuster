import inspect
import json
import multiprocessing
import threading

from frameduster.processes import _Processes
from frameduster.threads import _Threads
from frameduster.outputs import _Output
from frameduster.inputs import _Input
from frameduster.load import _Load
from frameduster.train import _Train
from frameduster.clerk import _Clerk

from frameduster.mongodb import _index_manager, _global_pbar_thread, _local_pbar_thread
from frameduster.config import config

def _topping_transformer(name, args, kwargs):
    def transformer(input):
        dec_layer = [(name, args, kwargs)]
        if inspect.isfunction(input) or inspect.isclass(input):
            return (json.dumps(dec_layer), input)
        else:
            loaded_decorators = json.loads(input[0])
            return (json.dumps(dec_layer + loaded_decorators), input[1])
    return transformer


def Output(*args, **kwargs):
    if args[0] == "s3" and len(args) > 2:
        kwargs['file_name'] = args[1]
        args = (args[0],)
    return _topping_transformer("Output", args, kwargs)


def Load(*args, **kwargs):
    return _topping_transformer("Load", args, kwargs)


def Input(*args, **kwargs):
    if args[0] == "s3" and len(args) > 2:
        kwargs['file_name'] = args[1]
        args = (args[0],)
    return _topping_transformer("Input", args, kwargs)


def Processes(*args, **kwargs):
    return _topping_transformer("Processes", args, kwargs)


def Threads(*args, **kwargs):
    return _topping_transformer("Threads", args, kwargs)


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

    print(_decode_decorator[decorators[0][0]])

    sub_proc_tuple = (next_decorators_str, func)
    return dec(*dec_args, **dec_kwargs)(__add_decorators(*sub_proc_tuple), sub_proc_tuple)