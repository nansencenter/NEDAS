import os
import sys
import subprocess
import time
from typing import Any, Callable

def print_with_cache(msg: str) -> None:
    # previous message is cached so that new message is displayed only
    # when it's different from the previous one (avoid redundant output)
    if not hasattr(print_with_cache, 'prev_msg'):
        setattr(print_with_cache, 'prev_msg', '')

    # only show at most nmsg messages
    if msg != getattr(print_with_cache, 'prev_msg', None):
        print(msg, flush=True, end="")
        setattr(print_with_cache, 'prev_msg', msg)

def watch_files(files, timeout=1000, check_dt=1):
    # wait for file in files to appear, check every check_dt seconds
    # if timeout seconds passed but still file not found, raise error
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
    # wait for keyword to appear in a logfile (indicating success in completion)
    # check every check_dt seconds
    # if logfile size grows (some active output is happening), reset the timer
    # if timeout is reached, raise error
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

class Formatter:
    def __init__(self, anchor=50, tabspace=4, progress_bar_width=10) -> None:
        # some visual parameters
        self.anchor = anchor
        self.tabspace = tabspace
        self.progress_bar_width = progress_bar_width

        # check if runtime output is redirected to a file
        self.tty = os.isatty(sys.stdout.fileno())

        # ANSI Escape sequences
        self.reset = "\033[0m" if self.tty else ''
        self.dim = "\033[2m" if self.tty else ''
        self.red = "\033[1;31m" if self.tty else ''
        self.green = "\033[1;32m" if self.tty else ''
        self.yellow = "\033[1;33m" if self.tty else ''
        self.blue = "\033[1;34m" if self.tty else ''
        self.clear_line = "\r\033[K" if self.tty else '\n'

        self.stat_flag = {
            'waiting': f"🕒 {self.blue}WAITING{self.reset}",
            'running': f"⏳ {self.yellow}RUNNING{self.reset}",
            'done': f"✅ {self.green}DONE{self.reset}",
            'error': f"❌ {self.red}ERROR{self.reset}",
        }

        assert self.tabspace > 1, "tabspace should be greater than 1 to have visible pipes in indent"
        self.pipe = f"│{' '*(self.tabspace-1)}"
        self.branch = f"├{'─'*(self.tabspace-2)} "
        self.padder = '─'

    def dimmer(self, text):
        return f"{self.dim}{text}{self.reset}"

    def indent(self, level: int, pop: bool=True) -> str:
        indent_str = ""
        if level > 1:
            indent_str += self.pipe*(level-2)
        indent_str += self.pipe if pop else self.branch
        return indent_str
        # raw_len = (level - 1) * self.tabspace + len(name) + 1
        # num_padding = max(2, self.anchor - raw_len)

        # return f"{self.dim}{indent_str}{self.reset}{name} {self.dim}{self.padder*num_padding}{self.reset} "

    def progress_bar(self, task_id: int, ntask: int) -> str:
        """
        Generate a progress bar based on task_id and ntask.

        Args:
            task_id (int): Current task index, from 0 to ntask-1
            ntask (int): Total number of tasks
            width (int): The length of the progress bar (number of characters)

        Returns:
            str: The progress bar msg to be shown.

        Note: Will require the print command with end="" option so that new line updated is overwritting the old line.
        """
        width = self.progress_bar_width
        progress = (task_id + 1) / ntask if ntask > 0 else 1.0
        filled_width = int(progress * width)

        bar = f"{self.yellow}{'━' * filled_width}{self.reset}{self.dim}{'─' * (width - filled_width)}{self.reset}"
        return f"[{bar}] {100*progress:3.0f}%"

class Progress:
    debug: bool
    debug_message: str = ''
    call_stack: list[dict[str, Any]]
    call_stack_max_level: int|None
    formatter: Formatter

    def __init__(self, debug: bool=False,
                 call_stack: list[dict]|None=None,
                 call_stack_max_level: int|None=None) -> None:
        self.debug = debug
        self.call_stack = []
        if call_stack:
            self.call_stack = call_stack
        self.call_stack_max_level = call_stack_max_level
        self.formatter = Formatter()

    @property
    def current_func(self) -> dict:
        if not self.call_stack:
            node = {
                'name': '',
                'substeps': 0,
                'status_line': '',
                'flag': 'running',
                'current_task': None,
                'total_tasks': None,
                'message': None,
                'elapsed_time': None,
            }
            return node
        return self.call_stack[-1]

    def stat_flag(self, flag: str) -> str:
        return self.formatter.stat_flag[flag]

    @property
    def elapsed_time(self) -> float:
        return self.call_stack[-1]['elapsed_time']

    @elapsed_time.setter
    def elapsed_time(self, value: float):
        self.call_stack[-1]['elapsed_time'] = value

    @property
    def within_max_level(self) -> bool:
        if self.call_stack_max_level is None:
            return True
        return len(self.call_stack) < self.call_stack_max_level
    
    @property
    def call_stack_level(self) -> int:
        return len(self.call_stack)

    def call_stack_push(self, func_name: str):
        node = {
            'name': func_name,
            'substeps': 0,
            'status_line': '',
            'flag': 'running',
            'current_task': None,
            'total_tasks': None,
            'message': None,
            'elapsed_time': None,
        }
        self.call_stack.append(node)

        # if self.call_stack[lv-1]['substeps'] == 0:
        #    # if lv < self.call_stack_max_level:
        #    self.print_1p('\n')
        indent = self.formatter.indent(self.call_stack_level, False)
        status_line = f"{indent}{func_name}: "

        if self.call_stack_level > 1:
            # mark a substep in the parent node
            self.call_stack[-2]['substeps'] += 1
        
        # update the status line for current node
        self.call_stack[-1]['status_line'] = status_line

    def call_stack_pop(self):
        flag = self.formatter.stat_flag['done']
        elapsed_time = self.call_stack[-1]['elapsed_time']
        timer_msg = f"{elapsed_time:7.2f}s" if elapsed_time is not None else ""
        pop = (self.call_stack[-1]['substeps']>0)
        indent = self.formatter.indent(self.call_stack_level, pop)
        if pop:
            self.call_stack[-1]['status_line'] = f"\033[2m{indent}\033[0m{flag} {timer_msg}\n"
        else:
            self.call_stack[-1]['status_line'] = f"\r\033[K{self.status_line} {flag} {timer_msg}\n"
        self.call_stack.pop()
        if len(self.call_stack)==1:
            self.call_stack[-1]['status_line'] = self.formatter.dimmer(self.formatter.pipe)+'\n'

    def raise_error(self):
        self.call_stack[-1]['flag'] = 'error'

    @property
    def status_line(self) -> str:
        anchor = 50
        indent = self.formatter.indent(len(self.call_stack))
        name_len = len(self.current_func) + len(indent)
        ndots = anchor - name_len if name_len < anchor else 2
        dot_string = self.formatter.dimmer(self.formatter.padder*ndots)
        stat_str = f"\r"+self.formatter.dimmer(indent)+self.current_func['name']+' '+dot_string
        return stat_str

    def show_progress(self) -> None:
        """
        Show progress

        If debug=True, print self.debug_message with flush=True
        Otherwise, show a progress bar, indicating current task/total_ntask percentage.
        """
        if self.debug:
            return
        pbar = self.formatter.progress_bar(self.call_stack[-1]['current_task'], self.call_stack[-1]['total_tasks'])
        
        # self._print(f"{self.status_line} {self.formatter.stat_flag['running']} {pbar}")
