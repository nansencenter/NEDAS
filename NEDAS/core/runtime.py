from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING
import os
import shutil
import errno
import subprocess
import numpy as np
from NEDAS.job_submitters import get_job_submitter
from .types import IOMode
if TYPE_CHECKING:
    from .context import Context

class Runtime(ABC):
    """
    Base class for handling runtime input/output for state and observation variables
    and OS-level operation such as file manipulation and running commands

    Attributes:
        tags (list[str]): List of names for copies of state/obs data

    """
    io_mode: IOMode = 'offline'
    tags: list[str] = ['prior', 'prior_mean', 'post', 'post_mean', 'truth', 'z', 'z_mean']

    def __init__(self, c: Context) -> None:
        pass

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
    def call_io_method(self, c: Context, tag: str, method: Callable, *args, **kwargs) -> Any:
        """
        Call a method to perform some io tasks.

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

    def move_file(self, file1: str, file2: str) -> None:
        shutil.move(file1, file2)

    def remove_file(self, file: str) -> None:
        try:
            os.remove(file)
        except OSError as e:
            # ignore if the file was already delected by other process
            if e.errno != errno.ENOENT:
                raise

    def remove_dir(self, dirname: str) -> None:
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            # ignore if the directory is already delected
            if e.errno != errno.ENOENT:
                raise

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

    def run_command(self, shell_cmd:str) -> None:
        """
        Run a shell command in a subprocess and check for errors.

        Args:
            shell_cmd (str): Shell command to be executed.

        Raises:
            RuntimeError: If the command returns a non-zero exit status.
        """
        p = subprocess.run(shell_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if p.returncode != 0:
            raise RuntimeError(f"{p.stderr}")

    def run_job(self, commands:str, **kwargs):
        """
        Run a shell command by submitting it to a job scheduler using JobSubmitter class.

        Args:
            commands (str): Shell command to be executed.
            **kwargs: Key-value pairs to passed to the job submitter run method.
        """
        ##get the right job submitter
        job_submitter = get_job_submitter(**kwargs)

        ##run job using the submitter
        job_submitter.run(commands)
