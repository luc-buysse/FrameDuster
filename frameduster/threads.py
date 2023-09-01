from config import _local
from iterable import _generator

import threading
from queue import Queue


def _thread_function(start_events, outputs, i, generator, slice_desc):
    _local.process_slice, _local.total_slices = slice_desc
    while True:
        start_events[i].wait()

        try:
            outputs.put((next(generator), i))
        except StopIteration:
            outputs.put(None)
            return
        start_events[i].clear()


def _thread_input(input_queue, input_sem):
    # while the queue is not empty and
    while True:
        # notify the dispatcher to add an item to the queue
        input_sem.release()

        # yield an item from the queue
        # if the item is None, the end of the input
        # iterator has been reached : return
        item = input_queue.get()
        if not item:
            return
        yield item


def _dispatch(out_queue, out_sem, input_gen, stop_count):
    while True:
        # wait for a thread to ask an input
        out_sem.acquire()
        try:
            # feed the queue
            out_queue.put(next(input_gen))
        except StopIteration:
            # notify all listening threads that it's over
            for _ in range(stop_count):
                out_queue.put(None)
            return


def _Threads(count):
    def transformer(input_func, sub_proc_wrapped):
        def wrapper(*args, **kwargs):
            threads = []
            start_events = [threading.Event() for i in range(count)]
            outputs = Queue()

            # if an input is needed, set up the dispatcher
            if len(args) > 0:
                inputs = Queue()
                input_sem = threading.Semaphore(count)

                # start dispatcher thread
                threading.Thread(target=_dispatch, args=(inputs, input_sem, args[0], count)).start()

            # launch the threads
            for i in range(count):
                local_args = args

                # if an input is needed, create a local
                # thread input plugged to the dispatcher
                if len(args) > 0:
                    local_args = (_thread_input(inputs, input_sem), *args[1:])

                gen = _generator(input_func, *local_args, **kwargs)
                thread = threading.Thread(target=_thread_function,
                                          args=(start_events, outputs, i, gen, (i, count)))

                thread.start()
                start_events[i].set()
                threads.append(thread)

            # number of running threads
            alive_count = count

            # return the data processed by the threads
            while True:
                output = outputs.get()

                # if output is None, the end of the generator has been reached by one of the thread
                # decrease the number of running threads and stop the loop if none is left
                if output is None:
                    alive_count -= 1
                    if alive_count == 0:
                        return
                    continue

                data, i = output

                # start the thread again
                start_events[i].set()

                yield data

        return wrapper

    return transformer
