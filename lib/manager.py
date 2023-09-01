import multiprocessing
import threading



class _GlobalStreamHandle:
    def __init__(self, wrapped):
        self.lock: multiprocessing.Lock = None
        self.stream_part: multiprocessing.Value = None
        self.stream_pos: multiprocessing.Value = None
        self.stream_size: multiprocessing.Value = None
        self.user_count: multiprocessing.Value = None
        self.complete_event: multiprocessing.Event = None
        self.complete_pipe: multiprocessing.Queue = None
        self.complete_lock: multiprocessing.Lock = None
        self.send_mongo_event: multiprocessing.Event = None
        self.done: multiprocessing.Value = None

        self.__dict__ = wrapped

    @staticmethod
    def create_wrapped(manager: multiprocessing.Manager):
        return {
            'lock': manager.Lock(),
            'stream_part': manager.Value('d', 1),
            'stream_pos': manager.Value('d', 0),
            'user_count': manager.Value('d', 0),
            'stream_size': manager.Value('d', 0),
            'done': manager.Value('b', False),
            'complete_event': manager.Event(),
            'complete_pipe': manager.Queue(),
            'complete_lock': manager.Lock()
        }


class _Global:
    REQ_NEW_STREAM_HANDLE = 0

    def request(self, req):
        # Send the request
        self.req_queue.put(req)

        # Wait for acknowledgment
        while self.ack_queue.get() != req:
            pass

    def manager_listener(self, manager):
        """
        This function listens for requests to add manager proxy items into the shared context.
        """

        while True:
            req_bundle = self.req_queue.get()
            request_id, data = req_bundle

            if request_id == _Global.REQ_NEW_STREAM_HANDLE:
                stream_key = data
                self.substream_handles[stream_key] = _GlobalStreamHandle.create_wrapped(manager)

            # Mark as done
            self.ack_queue.put(req_bundle)

    def __init__(self):
        # Create manager
        manager = multiprocessing.Manager()

        # Request setup
        self.req_queue = multiprocessing.Queue()
        self.ack_queue = multiprocessing.Queue()
        self.substream_handles = manager.dict()

        # Regular S3
        self.cache_size = manager.dict()
        self.cache_lock: multiprocessing.Lock = multiprocessing.Lock()

        # Index
        self.to_index = multiprocessing.Queue()

        # Recover
        self.recovered = multiprocessing.Value('b', False)
        self.recover_lock = multiprocessing.Lock()

        # Local
        self.local_lock = multiprocessing.Lock()
        self.local_count = manager.dict()

        # ------ Stream S3

        # Group
        self.group_lock = multiprocessing.Lock()
        self.streams = manager.dict()

        # Launch manager_listener
        threading.Thread(target=self.manager_listener, args=(manager,)).start()

    def get_stream_handle(self, stream_key):
        if stream_key not in self.substream_handles:
            self.request((_Global.REQ_NEW_STREAM_HANDLE, stream_key))

        return _GlobalStreamHandle(self.substream_handles[stream_key])

class Manager:
    def __init__(self):
        self._global = None


# Used to store the global object in case of multiprocessing
manager: Manager = Manager()
