from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING
import os
import shutil
import glob
import errno
import subprocess
import numpy as np
# from NEDAS.job_submitters import get_job_submitter
from .types import IOMode
if TYPE_CHECKING:
    from .context import Context

class IOBackend(ABC):
    """
    Base class for handling runtime input/output for state and observation variables
    and OS-level operation such as file manipulation and running commands

    Attributes:
        io_mode (IOMode): 'offline' for file I/O and 'online' for persistent memory I/O
        debug (bool): If in debug mode or not
        tags (list[str]): List of names for copies of state/obs data

    """
    io_mode: IOMode = 'offline'
    debug: bool = False
    tags: list[str] = ['prior', 'prior_mean', 'post', 'post_mean', 'truth', 'z', 'z_mean']

    def __init__(self, c: Context) -> None:
        self.debug = c.config.debug

    def validate_tag(self, tag: str):
        if tag not in self.tags:
            raise ValueError(f"IOBackend: unknown tag '{tag}', supported: {self.tags}")

    def prepare_collective_io(self, c: Context, tag: str) -> None:
        """
        Prepare for collective io, needed typically for offline io modes.

        Creating files for writing if not existing yet. Creating file locks for parallel io, etc.
        """
        pass

    @abstractmethod
    def read_field(self, c: Context, tag: str, rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read a 2D field data from the state

        Args:
            c (Context): the runtime context
            tag (str): which copy of the state to read from
            rec_id (int): field record id
            mem_id (int): ensemble member index from 0 to nens-1

        Returns:
            np.ndarray: the 2D field data
        """
        ...

    @abstractmethod
    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec_id: int, mem_id: int) -> None:
        """
        Write a 2D field data to the state

        Args:
            fld (np.ndarray): the 2D field data
            c (Context): the runtime context
            tag (str): which copy of the state to write to
            rec_id (int): field record object
            mem_id (int): ensemble member index
        """
        ...

    @abstractmethod
    def call_method(self, c: Context, tag: str, method: Callable, *args, **kwargs) -> Any:
        """
        Call a method to perform some tasks.

        Args:
            c (Context): the runtime context
            tag (str): which copy of the model state to request io from: "prior", "posterior" or "truth"
            method (Callable): method name
            *args, **kwargs: will be passed to the method

        Returns:
            Any: whatever the method(**kwargs) returns
        """
        ...

    @abstractmethod
    def save_ndarray(self, c: Context, name: str, data: np.ndarray, path: str | None=None) -> None:
        """
        Save ndarray data

        Args:
            c (Context): the runtime context
            name (str): the name of the data
            data (np.ndarray): the data
            path (str, optional): system path to save the data to.
        """
        ...

    @abstractmethod
    def load_ndarray(self, c: Context, name: str, path: str | None=None) -> np.ndarray | None:
        """
        Load ndarray from saved data
        
        Args:
            c (Context): the runtime context
            name (str): the name of the data
            path (str, optional): system path to the saved data.

        Returns:
            np.ndarray: the data
        """
        ...

    def make_dir(self, dirname:str|None) -> None:
        """
        Create a directory if it does not exist.

        FileExistsError can happen if multiple processors are trying to make the same directory.
        This function will ignore this error and continue.

        Args:
            dirname (str|None): Directory name to be created.
        """
        if dirname is None:
            return
        try:
            os.makedirs(dirname, exist_ok=True)
        except FileExistsError:
            pass

    def copy_file(self, file1: str, file2: str) -> None:
        shutil.copy2(file1, file2, follow_symlinks=True)
        if self.debug:
            print(f"copied {file1} to {file2}", flush=True)

    def move_file(self, file1: str, file2: str) -> None:
        if os.path.exists(file2):
            os.replace(file1, file2)
        else:
            shutil.move(file1, file2)
        if self.debug:
            print(f"moved {file1} to {file2}", flush=True)

    def move_files_to_dir(self, files: str, dirname: str) -> None:
        # Find all matching files and move them
        for file_path in glob.glob(files):
            dest_path = os.path.join(dirname, os.path.basename(file_path))
            if os.path.exists(dest_path):
                os.replace(file_path, dest_path)
            else:
                shutil.move(file_path, dirname)
            if self.debug:
                print(f"renamed '{file_path}' -> '{os.path.join(dirname, os.path.basename(file_path))}'", flush=True)

    def remove_files(self, files: str) -> None:
        for file_path in glob.glob(files):
            try:
                os.remove(file_path)
            except OSError as e:
                # ignore if the file was already delected by other process
                if e.errno != errno.ENOENT:
                    raise
            if self.debug:
                print(f"removed {file_path}", flush=True)

    def remove_dir(self, dirname: str) -> None:
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            # ignore if the directory is already delected
            if e.errno != errno.ENOENT:
                raise
        if self.debug:
            print(f"removed {dirname}", flush=True)

    def save_debug_data(self, c: Context, name: str, data: dict, path: str | None=None) -> None:
        """
        Save debug data in npz format

        Args:
            c (Context): the runtime context
            name (str): the name of the data
            data (dict): the data
            path (str, optional): system path to save the data to.
        """
        if path is None:
            path = c.config.work_dir  # default path

        file = os.path.join(path, f"{name}.npz")

        # make sure directory exists
        self.make_dir(os.path.dirname(file))

        # save the data to file
        np.savez(file, **data)

        if self.debug:
            print(f"saved debug data '{file}'", flush=True)

    # def run_command(self, shell_cmd:str) -> None:
    #     """
    #     Run a shell command in a subprocess and check for errors.

    #     Args:
    #         shell_cmd (str): Shell command to be executed.

    #     Raises:
    #         RuntimeError: If the command returns a non-zero exit status.
    #     """
    #     if self.debug:
    #         print(shell_cmd, flush=True)
    #     p = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    #     if p.returncode != 0:
    #         raise RuntimeError(f"{p.stderr}")

    # def run_job(self, commands:str, **kwargs):
    #     """
    #     Run a shell command by submitting it to a job scheduler using JobSubmitter class.

    #     Args:
    #         commands (str): Shell command to be executed.
    #         **kwargs: Key-value pairs to passed to the job submitter run method.
    #     """
    #     ##get the right job submitter
    #     job_submitter = get_job_submitter(**kwargs)

    #     ##run job using the submitter
    #     job_submitter.run(commands)
