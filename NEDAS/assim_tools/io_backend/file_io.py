import os
import struct
import numpy as np
from datetime import datetime
from NEDAS.utils.conversion import type_dic, type_size
from NEDAS.core import Coordinator, IOBackend

class FileIO(IOBackend):
    """
    IO Backend using restart files to hold model state (a pause-restart strategy)
    """
    directories: dict



    @property
    def prior_file(self):
        analysis_dir = self.analysis_dir(self.c.time, self.c.iter)
        return os.path.join(analysis_dir, 'prior_state.bin')

    # def read_field(self, rec: FieldRecord, member: int) -> np.ndarray:
    #     """Read a field from a binary file"""

    #     nv = 2 if rec.is_vector else 1
    #     fld_shape = (2,)+self.state.info.shape if rec.is_vector else self.state.info.shape
    #     fld_size = np.sum((~self.c.grid.mask).astype(int))

    #     with open(self.prior_file, 'rb') as f:
    #         f.seek(member*self.state.info.size + rec.pos)
    #         fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec.dtype]),
    #                         f.read(nv*fld_size*type_size[rec.dtype])))
    #         fld = np.full(fld_shape, np.nan)
    #         if rec.is_vector:
    #             fld[:, ~self.c.grid.mask] = fld_.reshape((2, -1))
    #         else:
    #             fld[~self.c.grid.mask] = fld_
    #         return fld

    # def write_field(self, fld: np.ndarray, rec: FieldRecord, member: int) -> None:
    #     """Write a field to a binary file"""
    #     fld_shape = (2,)+self.state.info.shape if rec.is_vector else self.state.info.shape
    #     assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

    #     if rec.is_vector:
    #         fld_ = fld[:, ~self.c.grid.mask].flatten()
    #     else:
    #         fld_ = fld[~self.c.grid.mask]

    #     with open(self.prior_file, 'r+b') as f:
    #         f.seek(member*self.state.info.size + rec.pos)
    #         f.write(struct.pack(fld_.size*type_dic[rec.dtype], *fld_))

    # def get_model_z_coords(self, c, model_name, **kwargs) -> np.ndarray:
    #     """
    #     Get z coordinates from model class using its z_coords method
    #     """
    #     path = c.forecast_dir(kwargs['time'], model_name)
    #     model = c.models[model_name]
    #     zfld = model.z_coords(path=path, **kwargs)
    #     return zfld
