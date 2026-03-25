import os
import subprocess
import time

def progress_bar(task_id: int, ntask: int, width: int=20) -> str:
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
    progress = (task_id + 1) / ntask if ntask > 0 else 1.0
    filled_width = int(progress * width)
    
    # ANSI Green for the bar, Reset for the track
    green = "\033[1;32m"
    reset = "\033[0m"
    dim   = "\033[2m"
    
    bar = f"{green}{'━' * filled_width}{reset}{dim}{'─' * (width - filled_width)}{reset}"
    return f"[{bar}] {100*progress:3.0f}%"

def print_with_cache(msg: str) -> None:
    ##previous message is cached so that new message is displayed only
    ##when it's different from the previous one (avoid redundant output)
    if not hasattr(print_with_cache, 'prev_msg'):
        setattr(print_with_cache, 'prev_msg', '')

    ##only show at most nmsg messages
    if msg != getattr(print_with_cache, 'prev_msg', None):
        print(msg, flush=True, end="")
        setattr(print_with_cache, 'prev_msg', msg)

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

def watch_log(logfile: str, keyword: str, timeout: int=1000, check_dt: int=1) -> None:
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

def find_keyword_in_file(file: str, keyword: str) -> bool:
    p = subprocess.run(f"grep '{keyword}' {file}", shell=True, capture_output=True, text=True)
    if p.stderr:
        raise RuntimeError(p.stderr)
    else:
        if p.stdout:
            return True
    return False

def count_lines_in_file(file: str) -> int:
    p = subprocess.run(f"wc -l {file}", shell=True, capture_output=True, text=True)
    if p.stderr:
        raise RuntimeError(p.stderr)
    else:
        n = int(p.stdout.split(' ')[0])
    return n

