from typing import Callable, Any
import numpy as np
from NEDAS.core.io_backend import IOBackend
from NEDAS.core.context import Context

class OnlineIO(IOBackend):
    """
    Online IO backend. Keep data in the memory and avoid file IO completely.

    Only works for single processor now, but this is convenient for long experiments and simple models
    """
    io_mode = 'online'
    shared_data: dict[str, Any] = {}

    def read_field(self, c: Context, tag: str, rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read a field from memory
        """
        self.validate_tag(tag)

        fields = getattr(c.state, f"fields_{tag}")
        return fields[mem_id, rec_id]

    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec_id: int, mem_id: int) -> None:
        """
        Write a field to memory
        """
        self.validate_tag(tag)

        if not hasattr(c.state, f"fields_{tag}"):
            setattr(c.state, f"fields_{tag}", {})
        fields = getattr(c.state, f"fields_{tag}")
        fields[mem_id, rec_id] = fld

    def read_obs(self, c: Context, tag: str, obs_rec_id: int, mem_id: int) -> np.ndarray:
        """
        Read an observation from memory
        """
        self.validate_tag(tag)
        obs_ens = getattr(c.obs, f'obs_{tag}')
        return obs_ens[mem_id, obs_rec_id]

    def write_obs(self, seq: np.ndarray, c: Context, tag: str, obs_rec_id: int, mem_id: int) -> None:
        """
        Write an observation to memory
        """
        self.validate_tag(tag)
        
        if not hasattr(c.obs, f'obs_{tag}'):
            setattr(c.obs, f'obs_{tag}', {})
        obs_ens = getattr(c.obs, f'obs_{tag}')
        obs_ens[mem_id, obs_rec_id] = seq

    def call_method(self, c: Context, tag: str, method: Callable, *args, **kwargs):
        self.validate_tag(tag)

        # 'post' is an alias for 'current' in online mode: the updator always writes
        # the posterior under 'current'; there is no separate 'post' memory slot.
        # In offline mode 'post' already routes to the same path as 'current'.
        kwargs['tag'] = 'current' if tag == 'post' else tag

        return method(*args, **kwargs)
