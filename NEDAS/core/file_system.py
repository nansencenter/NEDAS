import os
import glob
import shutil
import errno
from pathlib import Path
from datetime import datetime
from NEDAS.config import Config

class FileSystem:
    """
    Manages runtime file system paths, name of files and directories
    """
    config: Config

    def __init__(self, config: Config|None=None) -> None:
        if config is None:
            config = Config()  # default configuration
        self.config = config

        # make sure work_dir exists
        self.make_dir(config.work_dir)

    def cycle_dir(self, time: datetime) -> str:
        """
        Directory path for an analysis cycle.

        Args:
            time (datetime): Time of the analysis cycle.

        Returns:
            str: Directory path for the analysis cycle.
        """
        return self.config.directories['cycle_dir'].format(time=time)

    def forecast_dir(self, time: datetime, model_name: str) -> str:
        """
        Directory path for a model forecast step.

        Args:
            time (datetime): Time of the analysis cycle.
            model_name (str): Name of the model.

        Returns:
            str: Directory path for the model forecast.
        """
        return self.config.directories['forecast_dir'].format(time=time, model_name=model_name)

    def analysis_dir(self, time: datetime, iter: int=0) -> str:
        """
        Directory path for an analysis step.

        Args:
            time (datetime): Time of the analysis cycle.
            iter (int): If niter > 1, an outer iteration loop exists, step is the index in the loop.

        Returns:
            str: Directory path for the analysis step.
        """
        if self.config.niter == 1:
            iter_dir= ''
        else:
            iter_dir = f"iter{iter}"
        return self.config.directories['analysis_dir'].format(time=time, iter=iter_dir)

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
        if self.config.debug:
            print(f"copied {file1} to {file2}", flush=True)

    def move_file(self, file1: str, file2: str) -> None:
        if os.path.exists(file2):
            os.replace(file1, file2)
        else:
            shutil.move(file1, file2)
        if self.config.debug:
            print(f"moved {file1} to {file2}", flush=True)

    def move_files_to_dir(self, files: str, dirname: str) -> None:
        # Find all matching files and move them
        for file_path in glob.glob(files):
            dest_path = os.path.join(dirname, os.path.basename(file_path))
            if os.path.exists(dest_path):
                os.replace(file_path, dest_path)
            else:
                shutil.move(file_path, dirname)
            if self.config.debug:
                print(f"renamed '{file_path}' -> '{os.path.join(dirname, os.path.basename(file_path))}'", flush=True)

    def remove_files(self, files: str) -> None:
        for file_path in glob.glob(files):
            try:
                os.remove(file_path)
            except OSError as e:
                # ignore if the file was already delected by other process
                if e.errno != errno.ENOENT:
                    raise
            if self.config.debug:
                print(f"removed {file_path}", flush=True)

    def remove_dir(self, dirname: str) -> None:
        try:
            shutil.rmtree(dirname)
        except OSError as e:
            # ignore if the directory is already delected
            if e.errno != errno.ENOENT:
                raise
        if self.config.debug:
            print(f"removed {dirname}", flush=True)

    def link_file(self, src: str, dst: str) -> None:
            """
            Tries to create a symbolic link. If the filesystem doesn't support it 
            (like exFAT), falls back to a file copy.
            """
            # Ensure the source exists before trying anything
            if not os.path.exists(src):
                raise FileNotFoundError(f"Source file {src} not found.")

            # Remove destination if it exists to avoid 'File exists' errors
            if os.path.exists(dst):
                os.remove(dst)

            try:
                os.symlink(src, dst)
                if self.config.debug:
                    print(f"linked {src} -> {dst}", flush=True)
            except OSError as e:
                # Error 1: Operation not permitted (typical for exFAT/NTFS)
                # Error 95: Operation not supported
                if e.errno in (errno.EPERM, errno.EOPNOTSUPP):
                    if self.config.debug:
                        print(f"Symlink failed (FS limitation), copying instead: {src} -> {dst}", flush=True)
                    self.copy_file(src, dst)
                else:
                    raise