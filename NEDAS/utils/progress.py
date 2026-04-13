import os
import subprocess
import time
from typing import Any

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
    """
    Formatter of the progress display.

    Args:
        interactive (bool, optional): Whether the output is interactive (supports ansi escape code). Defaults to True.
        anchor (int, optional): Characters to anchor the left part of status line. Defaults to 50.
        tabspace (int, optional): Number of spaces for one call stack level indentation. Defaults to 4.
        progress_bar_width (int, optional): Width of the progress bar in characters. Defaults to 10.
    """
    def __init__(self, interactive: bool=True, anchor=50, tabspace=4, progress_bar_width=10) -> None:
        self.interactive = interactive

        # some visual parameters
        self.anchor = anchor
        self.tabspace = tabspace
        self.progress_bar_width = progress_bar_width

        # ANSI Escape sequences
        self.reset = "\033[0m" if self.interactive else ''
        self.dim = "\033[2m" if self.interactive else ''
        self.red = "\033[1;31m" if self.interactive else ''
        self.green = "\033[1;32m" if self.interactive else ''
        self.yellow = "\033[1;33m" if self.interactive else ''
        self.blue = "\033[1;34m" if self.interactive else ''
        self.clear_line = "\r\033[K" if self.interactive else '\n'

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

    def indent(self, level: int, branch: bool=True) -> str:
        """
        Generate the indent string to form call stack tree structure in log.

        Args:
            level (int): The current call stack level.
            branch (bool): Whether a branch is needed at the end.

        Returns:
            str: The indent string
        """
        if level <= 1:
            return ""
        indent_str = self.pipe*(level-2)
        indent_str += self.branch if branch else self.pipe
        return self.dimmer(indent_str)

    def padding(self, level: int, name: str) -> str:
        """
        Generate the padding string to align the status line.

        Args:
            level (int): The current call stack level.
            name (str): The name of the current function or task.

        Returns:
            str: The padding string
        """
        name_len = len(name) + (level-1)*self.tabspace
        n = self.anchor - name_len if name_len < self.anchor else 2
        return self.dimmer(self.padder*n)

    def dimmer(self, msg): 
        return self.dim+msg+self.reset

    def progress_bar(self, task_id: int, ntask: int) -> str:
        """
        Generate a progress bar based on task_id and ntask.

        Args:
            task_id (int): Current task index, from 0 to ntask-1
            ntask (int): Total number of tasks

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
    """
    Progress tracker and displayer. Used by Context.logger to show runtime progress.
    """
    interactive: bool
    debug: bool
    call_stack: list[dict[str, Any]]
    call_stack_max_level: int|None
    formatter: Formatter

    def __init__(self, interactive: bool=True,
                 debug: bool=False,
                 call_stack: list[dict]|None=None,
                 call_stack_max_level: int|None=None,
                 anchor: int=50,
                 tabspace: int=4,
                 progress_bar_width: int=10) -> None:
        self.interactive = interactive
        self.debug = debug
        self.call_stack = []
        if call_stack:
            self.call_stack = call_stack
        self.call_stack_max_level = call_stack_max_level # TODO: suppress level>max_level into single line
        self.fmt = Formatter(interactive, anchor, tabspace, progress_bar_width)

    def new_node(self, func_name: str|None=None) -> dict:
        node = {
            'name': func_name,
            'substeps': 0,
            'header': '',
            'flag': 'waiting',
            'current_task': 0,
            'total_tasks': 1,
            'message': '',
            'elapsed_time': None,
        }
        return node

    @property
    def node(self) -> dict:
        if not self.call_stack:
            return self.new_node('')
        return self.call_stack[-1]

    @property
    def within_max_level(self) -> bool:
        if self.call_stack_max_level is None:
            return True
        return self.level <= self.call_stack_max_level
    
    @property
    def level(self) -> int:
        return len(self.call_stack)

    def push(self, func_name: str):
        newline = ''
        if self.call_stack:
            parent = self.node
            if parent['substeps'] == 0:
                newline = '\n'
            parent['substeps'] += 1

        node = self.new_node(func_name)
        self.call_stack.append(node)

        indent = self.fmt.indent(self.level)
        self.node['header'] = f"{indent}{func_name}: "
        return f"{newline}{self.node['header']}"

    def pop(self):
        if not self.call_stack:
            return ''

        level = self.level
        node = self.node
        within_max_level = self.within_max_level
        self.call_stack.pop()

        stat_flag = self.fmt.stat_flag[node['flag']]
        elapsed_time = node['elapsed_time']
        timer_msg = f"{elapsed_time:7.2f}s" if elapsed_time is not None else ""
        message = f"({node['message']})" if node['message'] else ""
        indent = ''
        addline = ''

        if not self.interactive:
            if node['substeps'] > 0:
                indent = self.fmt.indent(level, branch=False)
                if within_max_level:
                    addline = f'{indent}\n'
            result = f"{stat_flag} {timer_msg} {message}"
            return f"{indent}{result}\n{addline}"

        if node['substeps'] > 0:
            newline = ''
            indent = self.fmt.indent(level, branch=False)
            result = f"{stat_flag} {timer_msg} {message}"
            if within_max_level:
                addline = f'{indent}\n'
        else:
            newline = self.fmt.clear_line
            indent = self.fmt.indent(level)
            name = node['name']
            padding = self.fmt.padding(level, name)
            result = f"{name} {padding} {stat_flag} {timer_msg} {message}"
            addline = ''
        return f"{newline}{indent}{result}\n{addline}"

    def flag(self, flag: str):
        self.node['flag'] = flag

    def update(self) -> str:
        node = self.node
        level = self.level
        current_task = node['current_task']
        total_tasks = node['total_tasks']

        if not self.interactive:
            prev_percent_bin = (100 * (current_task-1) // total_tasks) // 10
            curr_percent_bin = (100 * current_task // total_tasks) // 10
            if curr_percent_bin > prev_percent_bin:
                percent = 10 * curr_percent_bin
                return f"{percent}%..."
            return ''
        clear = self.fmt.clear_line
        indent = self.fmt.indent(level)
        func_name = node['name']
        padding = self.fmt.padding(level, func_name)
        stat_flag = self.fmt.stat_flag[node['flag']]
        pbar = ''
        if node['flag'] == 'running':
            pbar = self.fmt.progress_bar(current_task, total_tasks)
        message = f"({node['message']})" if node['message'] else ""
        return f"{clear}{indent}{node['name']} {padding} {stat_flag} {pbar} {message}"

    def log(self, msg: str) -> str:
        """
        Safely injects a global message without breaking the tree.
        """
        indent = self.fmt.indent(self.level+1, branch=False)
        addline = f"{indent}\n"
        if self.node['substeps'] == 0:
            return f"\n{addline}{msg}"
        return f"{addline}{msg}\n"
