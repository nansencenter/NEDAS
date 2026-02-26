import os
import numpy as np
from NEDAS.core import Context, IOBackend
from NEDAS.core.types import FieldRecord

class MemoryIO(IOBackend):
    def __init__(self, c: Context):
        pass

    def read_field(self, c: Context, rec: FieldRecord, member: int) -> np.ndarray:
        fld = np.ones(1)
        return fld    

    def write_field(self, c: Context, fld: np.ndarray, rec: FieldRecord, member: int) -> None:
        pass