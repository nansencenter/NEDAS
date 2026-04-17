import os
import struct
from typing import Callable
import numpy as np
from NEDAS.utils.conversion import type_dic, type_size
from NEDAS.core.io_backend import IOBackend
from NEDAS.core.context import Context

class OfflineIO(IOBackend):
    """
    Offline IO backend using restart files to hold model state (a pause-restart strategy)
    """
    io_mode = 'offline'

    def binfile_name(self, c: Context, tag: str) -> str:
        """
        Name of the binary file that stores the state data.

        Args:
            c (Context): the runtime context
            tag (str): which version of the state

        Returns:
            str: file name
        """
        analysis_dir = c.fs.analysis_dir(c.time, c.iter)
        return os.path.join(analysis_dir, f'fields_{tag}.bin')

    def prepare_fields_storage(self, c: Context, tag: str):
        binfile = self.binfile_name(c, tag)
        if c.pid == 0:
            # create the .bin file
            with open(binfile, 'wb') as f:
                pass
            # write state_info to the accompanying .dat file
            c.state.info.write_to_file(binfile)
        c.comm.Barrier()

    def read_field(self, c: Context, tag: str, rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read a field from cache or binary file
        """
        self.validate_tag(tag)
        # check if it is available in cache
        if hasattr(c.state, f"fields_{tag}"):
            if c.state and rec_id in c.state.rec_list[c.pid_rec] and mem_id in c.mem_list[c.pid_mem]:
                fields = getattr(c.state, f"fields_{tag}")
                return fields[mem_id, rec_id]

        # otherwise, read it from binfile
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
        # only write to binfile if the field is owned by the pid_mem
        # for ensemble mean every pid_mem receives a copy from allreduce, but only root need to write it.
        if mem_id not in c.mem_list[c.pid_mem]:
            return

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

    def call_method(self, c: Context, tag: str, method: Callable, *args, **kwargs):
        self.validate_tag(tag)
        # form the path in kwargs
        # additional info from input args to form the path prefix
        model_name = kwargs['model_src']
        model = c.models[model_name]
        if tag in ['raw', 'current', 'post', 'z']:
            path = c.fs.forecast_dir(c.time, model_name)

        elif tag == 'prior':
            path = c.fs.forecast_dir(c.prev_time, model_name)

        elif tag == 'truth':
            path = model.truth_dir

        else:
            raise ValueError(f"tag '{tag}' not supported in io.call_method")
        # make sure path exists
        if path:
            os.makedirs(path, exist_ok=True)  # use shell_utils makedir

        kwargs['path'] = path
        return method(*args, **kwargs)

    def save_ndarray(self, c: Context, name: str, data: np.ndarray, path: str | None = None) -> None:
        if path is None:
            path = c.config.work_dir  # default path

        file = os.path.join(path, f"{name}.npy")

        # make sure directory exists
        os.makedirs(os.path.dirname(file), exist_ok=True)

        # save the data to file
        np.save(file, data)

    def load_ndarray(self, c: Context, name: str, path: str | None = None) -> np.ndarray | None:
        if path is None:
            path = c.config.work_dir # default path

        file = os.path.join(path, f"{name}.npy")

        # load data from file
        if os.path.exists(file):
            return np.load(file)
        else:
            return None
