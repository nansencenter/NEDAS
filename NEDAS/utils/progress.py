import os
import subprocess
import time
from typing import Optional
import numpy as np
from functools import wraps

def timer(c=None):
    """
    Decorator to show the time spent on a function.

    Args:
        c (Config, optional): config object

    Once decorated, only processor with ID :code:`c.pid_show` in :code:`c.comm` will run the timer in the function.
    """
    def decorator(func):
        if c is not None and not getattr(c, 'timer', True):
            # if the config states timer=False, just return original func
            return func

        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                t1 = time.time()
                if c is None or (hasattr(c, 'comm') and c.comm.Get_rank() == getattr(c, 'pid_show', 0)):
                    print(f"timer: {func.__name__} took {t1 - t0} seconds", flush=True)
        return wrapper

    return decorator

def progress_bar(task_id: int, ntask: int, width: int=50):
    """
    Generate a progress bar based on task_id and ntask.

    Args:
        task_id (int): Current task index, from 0 to ntask-1
        ntask (int): Total number of tasks
        width (int): The length of the progress bar (number of characters)

    Returns:
        str: The progress bar msg to be shown.
            Will require the print command with end="" option so that new line updated is overwritting the old line.
    """
    if ntask==0:
        progress = 1
    else:
        progress = (task_id+1) / ntask

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

def watch_files(files, timeout=1000, check_dt=1):
    ##wait for file in files to appear, check every check_dt seconds
    ##if timeout seconds passed but still file not found, raise error
    if isinstance(files, list):
        file_list = files
    else:
        file_list = [files]
    elapsed_t = 0
    while file_list:
        file_list = [f for f in file_list if not os.path.exists(f)]
        time.sleep(check_dt)
        elapsed_t += check_dt
        if elapsed_t > timeout:
            raise RuntimeError(f"watch_files: timed out waiting for files {file_list}")

def watch_log(logfile, keyword, timeout=1000, check_dt=1):
    ##wait for keyword to appear in a logfile (indicating success in completion)
    ##check every check_dt seconds
    ##if logfile size grows (some active output is happening), reset the timer
    ##if timeout is reached, raise error
    elapsed_t = 0
    n0 = count_lines_in_file(logfile)
    while not find_keyword_in_file(logfile, keyword):
        time.sleep(check_dt)
        elapsed_t += check_dt
        n1 = count_lines_in_file(logfile)
        if n1 > n0:
            elapsed_t = 0
            n0 = n1
        if elapsed_t > timeout:
            raise RuntimeError(f"watch_log: {logfile} remain stagnant for {timeout} seconds, while waiting for keyword '{keyword}'")

def find_keyword_in_file(file, keyword):
    p = subprocess.run(f"grep '{keyword}' {file}", shell=True, capture_output=True, text=True)
    if p.stderr:
        raise RuntimeError(p.stderr)
    else:
        if p.stdout:
            return True
    return False

def count_lines_in_file(file):
    p = subprocess.run(f"wc -l {file}", shell=True, capture_output=True, text=True)
    if p.stderr:
        raise RuntimeError(p.stderr)
    else:
        n = int(p.stdout.split(' ')[0])
    return n

