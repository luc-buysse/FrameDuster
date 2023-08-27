import inspect
import multiprocessing
import threading
from multiprocessing import connection
import json

from lib.iterable import (_generator, _flush, )
from lib.outputs import _Output
from lib.inputs import _Input
from lib.config import config, _local
from lib.threads import _Threads
from lib.clerk import _Clerk

_decode_decorator = {
    "Output": _Output,
    "Input": _Input,
    "Threads": _Threads,
    "Clerk": _Clerk
}


def _manager_listener(manager):
    """
    This function listens for requests to add manager proxy items into the shared context.
    """

    while True:
        fid_dataset, request = config['shared_context']['request_queue'].get()

        if request == "s3_complete_lock":
            config['shared_context']['s3_complete_lock'][fid_dataset] = manager.Lock()
        elif request == "s3_complete_comm":
            config['shared_context']['s3_complete_event'][fid_dataset] = manager.Event()
            config['shared_context']['s3_complete_pipe'][fid_dataset] = manager.Queue()

        # Mark as done
        config['shared_context']['ack_queue'].append((fid_dataset, request))

def _init_shared_context():
    manager = multiprocessing.Manager()

    shared_context = {}

    # cache control
    shared_context['cache_size'] = manager.dict()
    shared_context['cache_s3_lock'] = multiprocessing.Lock()
    shared_context['mongo_s3_lock'] = multiprocessing.Lock()

    # index control
    shared_context['to_index'] = manager.Queue()

    # recover
    shared_context['recovered'] = manager.list()
    shared_context['recover_lock'] = manager.Lock()

    # s3 streams
    shared_context['s3_stream'] = manager.dict()
    shared_context['s3_stream_size'] = manager.dict()
    shared_context['s3_stream_user'] = manager.dict()
    shared_context['s3_stream_part'] = manager.dict()
    shared_context['s3_stream_lock'] = multiprocessing.Lock()

    # need to store mutable data structures
    # so it's initialized by the manager_listener thread
    shared_context['s3_complete_event'] = manager.dict()
    shared_context['s3_complete_pipe'] = manager.dict()
    shared_context['s3_complete_lock'] = manager.dict()

    # handle for the manager listener
    shared_context['request_queue'] = multiprocessing.Queue()
    shared_context['ack_queue'] = multiprocessing.Queue()

    config['shared_context'] = shared_context

    # launch the manager listener
    threading.Thread(target=_manager_listener, args=(manager,)).start()

def _add_decorators(decorators_str, func):
    decorators = json.loads(decorators_str)

    if len(decorators) == 0:
        return func

    dec = _decode_decorator[decorators[0][0]]
    dec_args = decorators[0][1]
    dec_kwargs = decorators[0][2]

    next_decorators_str = json.dumps(decorators[1:])

    sub_proc_tuple = (next_decorators_str, func)
    return dec(*dec_args, **dec_kwargs)(_add_decorators(*sub_proc_tuple), sub_proc_tuple)

def _sub_feeder(worker_conn):
    while True:
        # ask for data
        worker_conn.send(None)

        # wait for the answer
        resp = worker_conn.recv()
        if resp is None:
            return

        # return it
        yield resp

def _sub_transmitter(func, worker_conn, *args, **kwargs):
    for item in _generator(func, *args, **kwargs):
        if item is not None and worker_conn is not None:
            worker_conn.send(item)


def _sub_process(sub_proc_wrapped, shared_context, mixed_context, worker_conn_tuple, slice_desc):
    _local.process_slice, _local.total_slices = slice_desc

    worker_conn, do_feed = worker_conn_tuple

    config['shared_context'] = shared_context
    config['mixed_context'] = mixed_context
    if not do_feed:
        _sub_transmitter(_add_decorators(*sub_proc_wrapped), worker_conn)
    else:
        _sub_transmitter(_add_decorators(*sub_proc_wrapped), worker_conn, _sub_feeder(worker_conn))

def _Processes(count, do_generate=False):
    def transformer(func, sub_proc_wrapped):
        def wrapper(*args, **kwargs):
            processes = []

            do_feed = len(args) > 0 and (inspect.isgenerator(args[0]) or
                                         hasattr(args[0], '__iter__') or
                                         hasattr(args[0], '__next__'))
            dispatch = do_feed or do_generate
            feeder_generator = None
            if do_feed:
                feeder_generator = args[0]

            # setup dispatch
            if dispatch:
                pipes = [multiprocessing.Pipe() for i in range(count)]
                dispatcher_conns = list(map(lambda x: x[0], pipes))
                worker_conns = list(map(lambda x: x[1], pipes))

            if 'shared_context' in config:
                raise Exception('Two multi-processed routines cannot run at the same time.')

            # initialize shared context
            _init_shared_context()

            # launch subprocesses
            sub_process_count = count if dispatch else count-1
            for i in range(sub_process_count):
                wc = None if not dispatch else worker_conns[i]

                process_slice = i + (1 if not dispatch else 0)

                process = multiprocessing.Process(target=_sub_process, args=(sub_proc_wrapped,
                                                                             config['shared_context'],
                                                                             config['mixed_context'],
                                                                             (wc, do_feed),
                                                                             (process_slice, count)))
                process.start()
                processes.append(process)

            # if dispatching is needed, launch the dispatching loop
            if dispatch:
                # sentinels to be informed if a subprocess is finished
                sentinels = [proc.sentinel for proc in processes]

                # while some dispatching connections are still up
                while len(dispatcher_conns) > 0:

                    # wait for an event (process done, or data sent)
                    up_conns = connection.wait(dispatcher_conns + sentinels)

                    for conn in up_conns:
                        # for a dispatching connection
                        if conn in dispatcher_conns:
                            recv_item = conn.recv()

                            # if the item is None, the worker needs input data
                            # otherwise pass down the item into the main stream
                            if recv_item is not None:
                                yield recv_item
                            else:
                                try:
                                    next_item = next(feeder_generator)
                                except StopIteration:
                                    conn.send(None)
                                else:
                                    conn.send(next_item)

                        # for a subprocess stopped
                        else:
                            # remove its sentinel and its connection
                            # from the lists
                            rmi = sentinels.index(conn)
                            dispatcher_conns.pop(rmi)
                            sentinels.pop(rmi)

            # otherwise just use the main process like every subprocess
            else:
                _flush(func)
                for process in processes:
                    process.join()

            yield None
        return wrapper
    return transformer