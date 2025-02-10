import logging
from multiprocessing import Queue

class QueueHandler(logging.Handler):
    """Logging handler that sends log messages to a queue."""
    def __init__(self, queue):
        super().__init__()
        self.queue = queue

    def emit(self, record):
        self.queue.put(self.format(record))

def setup_logging(queue):
    """Set up logging to write to a file or console."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    # Create a handler that writes log messages to a file
    handler = logging.FileHandler('mu_zero.log')
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

    queue_handler = QueueHandler(queue)
    root.addHandler(queue_handler)

def log_listener(queue):
    """Listen for log messages and write them to a file."""
    while True:
        message = queue.get()
        if message == "STOP":
            break
        print(message)  