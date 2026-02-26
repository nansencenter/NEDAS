import os
import struct
import numpy as np
from datetime import datetime
from NEDAS.utils.conversion import type_dic, type_size
from NEDAS.core import Context, IOBackend
from NEDAS.core.types import FieldRecord

class FileIO(IOBackend):
    """
    IO Backend using restart files to hold model state (a pause-restart strategy)
    """
    directories: dict

    def __init__(self, c: Context):
        if c.config.directories is None:
            raise ValueError
        self.directories = c.config.directories

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
        if self.cf.niter == 1:
            iter_dir= ''
        else:
            iter_dir = f"iter{iter}"
        return self.directories['analysis_dir'].format(time=time, step=iter_dir)


    @property
    def binfile_name(self, tag):
        analysis_dir = '.'
        # analysis_dir = self.analysis_dir(self.c.time, self.c.iter)
        return os.path.join(analysis_dir, 'prior_state.bin')

    def read_field(self, c: Context, rec: FieldRecord, member: int) -> np.ndarray:
        """Read a field from a binary file"""

        nv = 2 if rec.is_vector else 1
        fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        fld_size = np.sum((~c.grid.mask).astype(int))

        with open(self.prior_file, 'rb') as f:
            f.seek(member*c.state.info.size + rec.pos)
            fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec.dtype]),
                            f.read(nv*fld_size*type_size[rec.dtype])))
            fld = np.full(fld_shape, np.nan)
            if rec.is_vector:
                fld[:, ~c.grid.mask] = fld_.reshape((2, -1))
            else:
                fld[~c.grid.mask] = fld_
            return fld

    def write_field(self, c: Context, fld: np.ndarray, rec: FieldRecord, member: int) -> None:
        """Write a field to a binary file"""
        fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

        if rec.is_vector:
            fld_ = fld[:, ~c.grid.mask].flatten()
        else:
            fld_ = fld[~c.grid.mask]

        with open(self.prior_file, 'r+b') as f:
            f.seek(member*c.state.info.size + rec.pos)
            f.write(struct.pack(fld_.size*type_dic[rec.dtype], *fld_))

    # def get_model_z_coords(self, c, model_name, **kwargs) -> np.ndarray:
    #     """
    #     Get z coordinates from model class using its z_coords method
    #     """
    #     path = c.forecast_dir(kwargs['time'], model_name)
    #     model = c.models[model_name]
    #     zfld = model.z_coords(path=path, **kwargs)
    #     return zfld
