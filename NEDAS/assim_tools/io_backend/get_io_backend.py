from __future__ import annotations
from typing import Union, TYPE_CHECKING
from .file_io import FileIO
from .memory_io import MemoryIO
if TYPE_CHECKING:
    from NEDAS.config import Config
    from NEDAS.assim_tools.state.state import State

def get_io_backend(c: Config, state: State) -> Union[FileIO, MemoryIO]:
    if c.io_mode == 'offline':
        return FileIO(c, state)
    elif c.io_mode == 'online':
        return MemoryIO(c, state)
    else:
        raise ValueError(f"unknown io_mode {c.io_mode}")
