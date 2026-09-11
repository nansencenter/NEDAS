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

    def analysis_dir(self, c: Context) -> str:
        return c.fs.analysis_dir(c.time, c.iter)

    def state_binfile_name(self, c: Context, tag: str) -> str:
        """
        Name of the binary file that stores the state data.
        """
        return os.path.join(self.analysis_dir(c), f'fields_{tag}.bin')

    def obs_binfile_name(self, c: Context, tag: str) -> str:
        """
        Name of the binary file that stores the observation data.
        """
        return os.path.join(self.analysis_dir(c), f'obs_{tag}.bin')

    def prepare_obs_storage(self, c: Context, tag: str) -> None:
        """Create obs_{tag}.bin and write obs_{tag}.dat before the parallel write loop."""
        binfile = self.obs_binfile_name(c, tag)
        if c.pid == 0:
            with open(binfile, 'wb') as f:
                pass
            c.obs.info.write_to_file(binfile)
        c.comm.Barrier()

    def write_obs(self, seq: np.ndarray, c: Context, tag: str, obs_rec_id: int, mem_id: int) -> None:
        if mem_id not in c.mem_list[c.pid_mem]:
            return
        # cache in memory so assimilator can access obs_prior during analysis
        getattr(c.obs, f'obs_{tag}')[mem_id, obs_rec_id] = seq
        # persist to binary file
        rec = c.obs.info.records[obs_rec_id]
        seq_ = seq.flatten() if rec.is_vector else seq
        with open(self.obs_binfile_name(c, tag), 'r+b') as f:
            f.seek(mem_id * c.obs.info.size + rec.pos)
            f.write(struct.pack(seq_.size * type_dic[rec.dtype], *seq_))

    def read_obs(self, c: Context, tag: str, obs_rec_id: int, mem_id: int) -> np.ndarray:
        # check memory cache first
        obs_store = getattr(c.obs, f'obs_{tag}')
        if (mem_id, obs_rec_id) in obs_store:
            return obs_store[mem_id, obs_rec_id]
        rec = c.obs.info.records[obs_rec_id]
        nv = 2 if rec.is_vector else 1
        with open(self.obs_binfile_name(c, tag), 'rb') as f:
            f.seek(mem_id * c.obs.info.size + rec.pos)
            raw = f.read(nv * rec.nobs * type_size[rec.dtype])
        seq_ = np.array(struct.unpack(nv * rec.nobs * type_dic[rec.dtype], raw))
        return seq_.reshape(2, rec.nobs) if rec.is_vector else seq_

    def prepare_fields_storage(self, c: Context, tag: str):
        binfile = self.state_binfile_name(c, tag)
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
                if (mem_id, rec_id) in fields:
                    return fields[mem_id, rec_id]

        # otherwise, read it from binfile
        rec = c.state.info.fields[rec_id]
        nv = 2 if rec.is_vector else 1
        fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        fld_size = np.sum((~c.grid.mask).astype(int))

        binfile = self.state_binfile_name(c, tag)
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

        binfile = self.state_binfile_name(c, tag)
        with open(binfile, 'r+b') as f:
            f.seek(mem_id*c.state.info.size + rec.pos)
            f.write(struct.pack(fld_.size*type_dic[rec.dtype], *fld_))

    def call_method(self, c: Context, tag: str, method: Callable, *args, **kwargs):
        self.validate_tag(tag)

        # static member (covariance_def.nens_static): a restart file in the bank static_dir, at the
        # time and source member listed for it in static_list; the time offset from the analysis
        # time (state/obs at multiple time steps) is kept
        if tag == 'static':
            static_time, static_member = c.covariance.static_members[kwargs['member']]
            kwargs['time'] = static_time + (kwargs['time'] - c.time)
            kwargs['member'] = static_member
            kwargs['path'] = c.covariance.static_dir
            return method(*args, **kwargs)

        # if path is already specified, directly call the method
        if 'path' in kwargs and kwargs['path'] is not None:
            return method(*args, **kwargs)

        # otherwise, use additional info from kwargs to form the path
        model_name = kwargs['model_src']
        model = c.models[model_name]
        if tag in ['raw', 'current', 'post', 'z']:
            path = c.fs.forecast_dir(c.time, model_name)
        elif tag == 'prior':
            if kwargs['time'] == c.time:
                if c.time == c.config.time_start and model.ens_init_dir is not None:
                    # very first cycle: c.prev_time collapses to c.time itself here (there is
                    # no earlier cycle), so forecast_dir(c.prev_time,...) would self-referentially
                    # resolve to the SAME directory as tag='current'/'post' -- reading it after
                    # this cycle's own analysis has run would silently return the POSTERIOR, not
                    # the prior. The true original prior for the first cycle is the pre-staged
                    # restart files in ens_init_dir instead (mirrors
                    # schemes/filter.py::Scheme.get_restart_dir(), which already resolves this
                    # exact case the same way for the preprocess/postprocess/ensemble_forecast
                    # steps -- found and fixed 2026-07-28, Yue, while adding a proper
                    # once-before-outer-loop path for prior inflation, RTPP in particular, which
                    # needs a genuine prior/post distinction even at the very first cycle).
                    path = model.ens_init_dir.format(time=c.time)
                else:
                    path = c.fs.forecast_dir(c.prev_time, model_name)
            else:
                path = c.fs.forecast_dir(c.time, model_name)
        elif tag == 'truth':
            path = model.truth_dir
        else:
            raise ValueError(f"tag '{tag}' not supported in io.call_method")

        # make sure path exists
        # if path:
        #     c.fs.make_dir(path)

        kwargs['path'] = path
        return method(*args, **kwargs)
