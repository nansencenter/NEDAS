import os
import struct
from typing import Callable
import numpy as np
from datetime import datetime
from NEDAS.utils.conversion import type_dic, type_size
from NEDAS.core import Context, IOBackend
from NEDAS.core.types import FieldRecord

class FileIO(IOBackend):
    """
    IO Backend using restart files to hold model state (a pause-restart strategy)
    """
    dictionaries: dict
    niter: int

    def __init__(self, c: Context):
        if c.config.directories is None:
            raise ValueError
        self.directories = c.config.directories
        self.niter = c.config.niter

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
        if self.niter == 1:
            iter_dir= ''
        else:
            iter_dir = f"iter{iter}"
        return self.directories['analysis_dir'].format(time=time, step=iter_dir)

    def binfile_name(self, c: Context, tag: str) -> str:
        """
        Name of the binary file that stores the state data.

        Args:
            c (Context): the runtime context
            tag (str): which version of the state

        Returns:
            str: file name
        """
        analysis_dir = self.analysis_dir(c.time, c.iter)
        return os.path.join(analysis_dir, f'fields_{tag}.bin')

    def prepare_collective_io(self, c: Context, tag: str):
        binfile = self.binfile_name(c, tag)
        if c.pid == 0:
            ##if file doesn't exist, create the file
            open(binfile, 'wb')
            ##write state_info to the accompanying .dat file
            c.state.info.write_to_file(binfile)
        c.comm.Barrier()

    def read_field(self, c: Context, tag: str, rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read a field from cache or binary file
        """
        self.validate_tag(tag)
        ##check if it is available in cache
        if c.state and rec_id in c.state.rec_list[c.pid_rec] and mem_id in c.state.mem_list[c.pid_mem]:
            fields = getattr(c.state, f"fields_{tag}")
            return fields[mem_id, rec_id]

        ##otherwise, read it from binfile
        rec = c.state.info.fields[rec_id]
        nv = 2 if rec.is_vector else 1
        fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        fld_size = np.sum((~c.grid.mask).astype(int))

        binfile = self.binfile_name(c, tag)
        with open(binfile, 'rb') as f:
            f.seek(mem_id*c.state.info.size + rec.pos)
            fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec.dtype]),
                            f.read(nv*fld_size*type_size[rec.dtype])))
            fld = np.full(fld_shape, np.nan)
            if rec.is_vector:
                fld[:, ~c.grid.mask] = fld_.reshape((2, -1))
            else:
                fld[~c.grid.mask] = fld_
            return fld

    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec_id: int, mem_id: int) -> None:
        """
        Write a field to a binary file
        """
        self.validate_tag(tag)
        rec = c.state.info.fields[rec_id]
        fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

        if rec.is_vector:
            fld_ = fld[:, ~c.grid.mask].flatten()
        else:
            fld_ = fld[~c.grid.mask]

        binfile = self.binfile_name(c, tag)
        with open(binfile, 'r+b') as f:
            f.seek(mem_id*c.state.info.size + rec.pos)
            f.write(struct.pack(fld_.size*type_dic[rec.dtype], *fld_))

        c.comm.Barrier()

    def call_io_method(self, c: Context, tag: str, method: Callable, *args, **kwargs):
        self.validate_tag(tag)
        # form the path in kwargs 
        # additional info from input args to form the path prefix
        model_name = kwargs['model_src']
        model = c.models[model_name]
        time = kwargs['time']
        if tag == 'prior':
            path = self.forecast_dir(time, model_name)

        if tag == 'post':
            prev_time = time - c.config.cycle_period
            if prev_time < c.config.time_start:
                prev_time = c.config.time_start
            path = self.forecast_dir(prev_time, model_name)

        elif tag == 'truth':
            path = model.truth_dir

        else:
            raise ValueError(f"Unknown tag: '{tag}'")

        kwargs['path'] = path
        return method(*args, **kwargs)
