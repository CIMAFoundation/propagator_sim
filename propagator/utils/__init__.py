from .utils import *
from .scheduler import Scheduler
#from propagator_scheduler_rs import Scheduler


import psutil
import os

def  get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB     
