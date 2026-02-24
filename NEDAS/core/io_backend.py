from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
#from NEDAS.core import Coordinator

class IOBackend:
    """
    Base class for handling input/output for state and observation variables
    """
    def __init__(self, c):
        # self.analysis_dir = c.analysis_dir(c.time, c.iter)
        # self.prior_file = os.path.join(self.analysis_dir, 'prior_state.bin')
        # self.prior_mean_file = os.path.join(self.analysis_dir, 'prior_mean_state.bin')
        # self.post_file = os.path.join(self.analysis_dir, 'post_state.bin')
        # self.post_mean_file = os.path.join(self.analysis_dir, 'post_mean_state.bin')
        # self.z_coords_file = os.path.join(self.analysis_dir, 'z_coords.bin')
        self.c = c
        if self.c.config.directories:
            self.directories = self.c.config.directories
        else:
            raise ValueError(f"Config: directories must be defined for io_mode='offline'")

    def cycle_dir(self, time: datetime) -> str:
        """
        Directory path for an analysis cycle.

        Args:
            time (datetime): Time of the analysis cycle.

        Returns:
            str: Directory path for the analysis cycle.
        """
        return self.directories['cycle_dir'].format(time=time)

    def forecast_dir(self, time: datetime, model_name: str):
        """
        Directory path for a model forecast step.

        Args:
            time (datetime): Time of the analysis cycle.
            model_name (str): Name of the model.

        Returns:
            str: Directory path for the model forecast.
        """
        return self.directories['forecast_dir'].format(time=time, model_name=model_name)

    def analysis_dir(self, time: datetime, iter: int=0):
        """
        Directory path for an analysis step.

        Args:
            time (datetime): Time of the analysis cycle.
            iter (int): If niter > 1, an outer iteration loop exists, step is the index in the loop.

        Returns:
            str: Directory path for the analysis step.
        """
        if self.c.config.niter == 1:
            iter_dir= ''
        else:
            iter_dir = f"iter{iter}"
        return self.directories['analysis_dir'].format(time=time, step=iter_dir)

    @abstractmethod
    def read_field(self, rec, member: int) -> np.ndarray:
        """
        Read a 2D field from the State

        Args:
            rec (FieldRecord): field record object
            member (int): ensemble member index from 0 to nens-1

        Returns:
            np.ndarray: the 2D field data
        """
        pass

    @abstractmethod
    def write_field(self, fld: np.ndarray, rec, member: int) -> None:
        """
        Write a 2D field to the State

        Args:
            fld (np.ndarray): the 2D field data
            rec (FieldRecord): field record object
            member (int): ensemble member index
        """
        pass
