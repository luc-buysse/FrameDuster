import threading

from lib.mongodb import _mark_done
from lib.iterable import _generator
from lib.config import config


def _Clerk(function_id):
    """
    Keeps track of the document processes by the user function.
    """
    def transformer(func, sub_proc_wrapped):
        def wrapper(input):

            def in_marker():
                sem = threading.Semaphore(500)
                def mark_done(_id, function_id):
                    _mark_done(_id, function_id)
                    sem.release()

                for item in input:
                    # Pass item to the service user_function
                    yield item

                    # If it's a (doc, file) tuple, get rid of the file
                    if isinstance(item, tuple):
                        item = item[0]

                    # Mark the item as processed
                    sem.acquire()
                    threading.Thread(target=mark_done, args=(item['_id'], function_id)).start()

                    # Update local progress bar
                    if config['mixed_context']['pbar']['input']:
                        config['mixed_context']['pbar_input_queue'].put(0)

            if config['mixed_context']['pbar']['output']:
                for item in _generator(func, in_marker()):
                    yield item
                    config['mixed_context']['pbar_output_queue'].put(0)
            else:
                yield from _generator(func, in_marker())

        return wrapper
    return transformer
