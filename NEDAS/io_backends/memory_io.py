from typing import Callable, Any
import numpy as np
from NEDAS.core import Context, IOBackend

class MemoryIO(IOBackend):
    """
    Memory IO backend. Keep data in the memory and avoid file I/O completely.

    Only works for single processor now, but this is convenient for long experiments and simple models
    """
    shared_data: dict[str, Any] = {}

    def __init__(self, c: Context):
        #assert c.config.nproc == 1, "currently only support serial programs: nproc=1"

        # allocate shared data in mpi environment
        #c.config.nproc > 1

        # allocate shared memory
        #self.memory = np.zeros((100, 100, 3))
        pass

    def read_field(self, c: Context, tag: str, rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read a field from memory
        """
        self.validate_tag(tag)

        ##check if it is available in state cache
        if c.state and rec_id in c.state.rec_list[c.pid_rec] and mem_id in c.state.mem_list[c.pid_mem]:
            fields = getattr(c.state, f"fields_{tag}")
            return fields[mem_id, rec_id]

        ##otherwise, get it from model.memory
        rec = c.state.info.fields[rec_id]
        model = c.models[rec.model_src]
        return model.read_var(tag=tag, member=mem_id, **rec.asdict())

    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec_id: int, mem_id: int) -> None:
        """
        Write a field to memory
        """
        self.validate_tag(tag)
        rec = c.state.info.fields[rec_id]
        model = c.models[rec.model_src]
        model.write_var(fld, tag=tag, member=mem_id, **rec.asdict())

    def call_io_method(self, c: Context, tag: str, method: Callable, *args, **kwargs):
        self.validate_tag(tag)

        # just append tag to the kwargs, online model classes will read this tag
        # and look for corresponding dict entries for cached data.
        kwargs['tag'] = tag

        return method(*args, **kwargs)

    def save_ndarray(self, c: Context, name: str, data: np.ndarray, path: str | None = None) -> None:
        # form the key in the data dict
        key = name
        if path is not None:
            key = f"{path}_{name}"

        # save data to dict
        self.shared_data[key] = data
    
    def load_ndarray(self, c: Context, name: str, path: str | None = None) -> np.ndarray | None:
        # form the key in the data dict
        key = name
        if path is not None:
            key = f"{path}_{name}"

        # read from dict to obtain data
        if key in self.shared_data:
            return self.shared_data[key]
        else:
            return None
 
    def output_snapshot(self, c: Context) -> None:
        """
        Output a snapshot of data stored in memory to npz files
        """
        ...
