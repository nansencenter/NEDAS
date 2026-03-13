import os
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

    def __init__(self, work_dir: str|None=None, directories: dict[str, str]|None=None, niter: int=1) -> None:
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
