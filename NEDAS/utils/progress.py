import re
import os
import shutil
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
    def __init__(self, interactive: bool=True, is_notebook: bool=False, cols=80, anchor=50, tabspace=4, progress_bar_width=10) -> None:
        self.interactive = interactive
        self.is_notebook = is_notebook
        self.cols = cols

        # some visual parameters
        self.anchor = anchor
        self.tabspace = tabspace
        self.progress_bar_width = progress_bar_width

        # ANSI Escape sequences
        self.reset = "\033[0m" if self.interactive else ''
        self.dim = "\033[38;5;244m" if self.interactive else ''
        self.red = "\033[1;31m" if self.interactive else ''
        self.green = "\033[1;32m" if self.interactive else ''
        self.yellow = "\033[1;33m" if self.interactive else ''
        self.blue = "\033[1;34m" if self.interactive else ''
        self.clear_line = "\r\033[K" if self.interactive else '\n'
        self.event_flag = {
            '': f"{self.dim}◈{self.reset}",
            'info': f"🔔 ",
            'warning': f"{self.red}!!{self.reset}",
            'stats': "🔎 ",
            'finish': "🏁 ",
        }
        self.stat_flag = {
            'waiting': f"{self.blue}⏰ {self.reset}",
            'running': f"{self.yellow}⏳ {self.reset}",
            'done': f"{self.green}✅ {self.reset}",
            'error': f"{self.red}❌ {self.reset}",
        }

        assert self.tabspace > 1, "tabspace should be greater than 1 to have visible pipes in indent"
        self.pipe = f"│{' '*(self.tabspace-1)}"
        self.branch = f"├{'─'*(self.tabspace-2)} "
        self.padder = '─'

    def strip_escape_code(self, text: str) -> str:
        re_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
        clean_text = re_escape.sub('', text).replace('\r', '')
        return clean_text

    def truncate(self, text: str) -> str:
        visible_text = self.strip_escape_code(text)
        visible_len = len(visible_text)

        # get real time cols if possible
        cols_in = self.cols
        if self.is_notebook:
            self.cols = 800
        else:
            self.cols, _ = shutil.get_terminal_size() #fallback=(self.cols,1))

        if visible_len <= self.cols:
           padding_needed = self.cols - visible_len - 1
           return f"{text}{' '*padding_needed}"

        result = ''
        count = 0
        max_visible = self.cols - 4
        parts = re.split(r'(\x1b\[[0-9;]*[a-zA-Z])', text)
        for part in parts:
            if part.startswith('\x1b'):
                result += part
            else:
                chars_left = max_visible - count
                if len(part) <= chars_left:
                    result += part
                    count += len(part)
                else:
                    result += part[:chars_left]
                    count += chars_left
                    break
        return result + "..." + self.reset # Always reset to be safe

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
        n = self.anchor - name_len - 2 if name_len < (self.anchor-2) else 2
        return self.dimmer(f" {self.padder*n} ")

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
                 is_notebook: bool=False,
                 cols: int=80,
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

        self.call_stack_max_level = call_stack_max_level
        if not self.interactive:
            self.call_stack_max_level = None

        self.fmt = Formatter(interactive, is_notebook, cols, anchor, tabspace, progress_bar_width)

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

    def within_max_level(self, level: int) -> bool:
        if self.call_stack_max_level is None:
            return True
        if level <= 2:
            return True
        return level <= self.call_stack_max_level

    def is_leaf(self, node) -> bool:
        if node['substeps'] == 0:
            if self.call_stack_max_level and self.call_stack_max_level < 2:
                return False
            return True 
        return False

    @property
    def level(self) -> int:
        return len(self.call_stack)

    def set_flag(self, flag: str):
        self.node['flag'] = flag

    def get_timer_msg(self, node):
        elapsed = node.get('elapsed_time')
        timer_msg = f"{elapsed:7.2f}s" if elapsed is not None else ""
        return timer_msg

    def _format_line(self, node: dict,
                     level: int,
                     include_indent: bool=True,
                     include_name: bool=True,
                     include_padding: bool=True,
                     is_branch: bool=True,
                     trailer: str="") -> str:
        """
        Maintains order: [Indent] [Name] [Padding] [Flag] [Trailer (PBar/Time)] [Message]
        """
        indent = self.fmt.indent(level, branch=is_branch) if include_indent else ""
        name = node['name'] if include_name else ""
        if self.within_max_level(level):
            header = indent+name
            padding = self.fmt.padding(level, name) if include_padding else ""
        else:
            header = node['header']
            padding = f"... "
        stat_flag = self.fmt.stat_flag.get(node['flag'], node['flag'].upper())
        message = f"({node['message']})" if node['message'] else ""
        return f"{header}{padding}{stat_flag} {trailer} {message}"

    def push(self, func_name: str):
        parent = self.node
        node = self.new_node(func_name)
        self.call_stack.append(node)

        if self.call_stack_max_level and self.call_stack_max_level < 2 and self.level > 1:
            return ""
        if self.debug:
            return f"\nENTERING: {func_name}\n"

        newline = ''
        if self.call_stack:
            if self.within_max_level(self.level):
                if parent['substeps'] == 0:
                    newline = '\n'
                parent['substeps'] += 1
            else:
                newline = self.fmt.clear_line

        if self.within_max_level(self.level):
            self.node['header'] = f"{self.fmt.indent(self.level)}{func_name}"
            return newline+self.fmt.truncate(f"{self.node['header']}: ")
        self.node['header'] = f"{parent['header']} > {func_name}"
        return newline+self.fmt.truncate(f"{self.node['header']}: ")

    def pop(self):
        if not self.call_stack:
            return ''
        node, level = self.node, self.level
        within_max_level = self.within_max_level(level)
        self.call_stack.pop()

        timer_msg = self.get_timer_msg(node)
        stat_flag = self.fmt.stat_flag.get(node['flag'], node['flag'].upper())

        if self.call_stack_max_level and self.call_stack_max_level < 2 and level > 1:
            return ""
        if self.debug:
            return f"EXITING: {node['name']} {stat_flag} {timer_msg}\n\n"

        # Handle the vertical branch line for parents
        addline = f"{self.fmt.indent(level, branch=False)}\n" if (node['substeps'] > 0 and within_max_level) else ""

        if not self.interactive:
            is_branch = (node['substeps'] == 0)
            res = self._format_line(node, level, include_name=False, include_padding=False, include_indent=not is_branch, is_branch=is_branch, trailer=timer_msg)
            return self.fmt.truncate(res)+f"\n{addline}"

        if self.is_leaf(node):
            # Leaf node: clear line, show full name + result
            res = self._format_line(node, level, include_name=True, is_branch=True, trailer=timer_msg)
            res = self.fmt.truncate(res)
            if within_max_level:
                res += f"\n{addline}"
            return f"{self.fmt.clear_line}{res}"
        else:
            # Parent node: don't clear, don't show name/padding
            res = self._format_line(node, level, include_name=False, include_padding=False, is_branch=False, trailer=timer_msg)
            return self.fmt.truncate(res)+f"\n{addline}"

    def update(self) -> str:
        if self.call_stack_max_level and self.call_stack_max_level < 2:
            return ""
        if self.debug:
            return ""

        node, level = self.node, self.level
        total, current = node['total_tasks'], node['current_task']

        if not self.interactive:
            prev_bin = (100 * (current - 1) // total) // 10
            curr_bin = (100 * current // total) // 10
            return f"{10 * curr_bin}%..." if curr_bin > prev_bin else ''

        # Prepare the 'trailer' (Progress Bar)
        pbar = self.fmt.progress_bar(current, total) if node['flag'] == 'running' else ""

        res = self._format_line(node, level, include_name=True, trailer=pbar)
        res = self.fmt.truncate(res)
        return f"{self.fmt.clear_line}{res}"

    def log(self, msg: str, flag: str) -> str:
        """
        Safely injects a global message without breaking the tree.
        """
        indent = ''
        if self.within_max_level(self.level):
            indent = self.fmt.indent(self.level+1, branch=False)
        else:
            if self.call_stack_max_level:
                indent = self.fmt.indent(self.call_stack_max_level, branch=False)

        event_pin = self.fmt.event_flag.get(flag, flag.upper()+':')

        if self.debug:
            return f"{event_pin} {msg}\n"

        if self.is_leaf(self.node):
            if self.node['flag'] == 'running':
                return f"\n{indent}\n{event_pin} {msg}\n"
            return f"\n{indent}\n{event_pin} {msg}"
        return f"{indent}\n{event_pin} {msg}\n"
