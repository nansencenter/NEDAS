import os
import glob
import shutil
import errno
from pathlib import Path
from datetime import datetime

default_directories = {
  'cycle_dir': '{work_dir}/cycle/{time:%Y%m%d%H%M}',
  'forecast_dir': '{work_dir}/cycle/{time:%Y%m%d%H%M}/{model_name}',
  'analysis_dir': '{work_dir}/cycle/{time:%Y%m%d%H%M}/analysis/{iter}',
}

class FileSystem:
    """
    Manages runtime file system paths, name of files and directories
    """
    directories: dict[str, str]  # defines structure of working directories
    niter: int                   # number of iterations in assimilation algorithms
    debug: bool

    def __init__(self, work_dir: str|None=None, directories: dict[str, str]|None=None, niter: int=1, debug: bool=False) -> None:
        # the main working directory
        if work_dir is None:
            work_dir = os.getcwd()  # default to current directory
        self.work_dir = os.path.abspath(work_dir)

        # set up directory structure
        if directories is None:
            directories = default_directories
        self.directories = {}
        for key, value in directories.items():
            self.directories[key] = str(Path(value.replace('{work_dir}', self.work_dir)))

        # setup number of iterations
        self.niter = niter

        self.debug = debug

    def cycle_dir(self, time: datetime) -> str:
        """
        Directory path for an analysis cycle.

        Args:
            time (datetime): Time of the analysis cycle.

        Returns:
            str: Directory path for the analysis cycle.
        """
        return self.directories['cycle_dir'].format(time=time)

    def forecast_dir(self, time: datetime, model_name: str) -> str:
        """
        Directory path for a model forecast step.

        Args:
            time (datetime): Time of the analysis cycle.
            model_name (str): Name of the model.

        Returns:
            str: Directory path for the model forecast.
        """
        return self.directories['forecast_dir'].format(time=time, model_name=model_name)

    def analysis_dir(self, time: datetime, iter: int=0) -> str:
        """
        Directory path for an analysis step.

        Args:
            time (datetime): Time of the analysis cycle.
            iter (int): If niter > 1, an outer iteration loop exists, step is the index in the loop.

        Returns:
            str: Directory path for the analysis step.
        """
        if self.niter == 1:
            iter_dir= ''
        else:
            iter_dir = f"iter{iter}"
        return self.directories['analysis_dir'].format(time=time, iter=iter_dir)

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
