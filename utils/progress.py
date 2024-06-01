import numpy as np
import shutil
import time
from functools import wraps

def timer(c):
    """
    Decorator to show the time spent on a function
    Input: -c: config object
    only processor c.pid_show in c.comm will show the timer message
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()
            if c.comm.Get_rank() == c.pid_show:
                print(f"timer: {func.__name__} took {t1 - t0} seconds\n")
            return result
        return wrapper
    return decorator


def progress_bar(task_id, ntask, width=None):
    """
    Generate a progress bar based on task_id and ntask

    Inputs:
    - task_id: int
      Current task index, from 0 to ntask-1

    - ntask: int
      Total number of tasks

    - width: int, optional
      The length of the progress bar (number of characters)

    Return:
    - pstr: str
      The progress bar msg to be shown
    """
    if ntask==0:
        progress = 1
    else:
        progress = (task_id+1) / ntask

    if width is None:
        console_width = shutil.get_terminal_size().columns
        width = int(0.5 * console_width)

    ##the progress bar looks like this: .....  | ??%
    pstr = '\r{:{}}| '.format('.'*int(np.ceil(progress * width)), width)

    ##add the percentage completed at the end
    pstr += '{:.0f}%'.format(100*progress)

    return pstr


def print_with_cache(msg):
    ##previous message is cached so that new message is displayed only
    ##when it's different from the previous one (avoid redundant output)
    if not hasattr(print_with_cache, 'prev_msg'):
        print_with_cache.prev_msg = ''

    ##only show at most nmsg messages
    if msg != print_with_cache.prev_msg:
        print(msg, flush=True, end="")
        print_with_cache.prev_msg = msg

