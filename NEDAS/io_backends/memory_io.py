import numpy as np
from NEDAS.core import Context, IOBackend
from NEDAS.core.types import FieldRecord

"""
Memory IO backend. Keep data in the memory and avoid file I/O completely.

Only works for single processor now, but this is convenient for long experiments and simple models
"""

class MemoryIO(IOBackend):
    nens: int
    memory: dict[str, dict] = {}

    def __init__(self, c: Context):
        assert c.config.nproc == 1, "currently only support serial programs: nproc=1"

        self.nens = c.config.nens

        # allocate memory
        for tag in self.tags:
            self.memory[tag] = {}
            # for mem_id in range(self.nens):
            #     for rec_id 
            #         self.memory[tag][mem_id, rec_id] = c.models[model_name].memory[] ##don't, we don't know about this

    def read_field(self, c: Context, tag: str, rec: FieldRecord, member: int) -> np.ndarray:
        """
        Read a field from memory
        """
        # nv = 2 if rec.is_vector else 1
        # fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        # fld_size = np.sum((~c.grid.mask).astype(int))

        # binfile = self.binfile_name(c, tag)
        # with open(binfile, 'rb') as f:
        #     f.seek(member*c.state.info.size + rec.pos)
        #     fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec.dtype]),
        #                     f.read(nv*fld_size*type_size[rec.dtype])))
        #     fld = np.full(fld_shape, np.nan)
        #     if rec.is_vector:
        #         fld[:, ~c.grid.mask] = fld_.reshape((2, -1))
        #     else:
        #         fld[~c.grid.mask] = fld_
        rec_id = ...
        return self.memory[tag][member, rec_id, ...]

    def write_field(self, fld: np.ndarray, c: Context, tag: str, rec: FieldRecord, member: int) -> None:
        """
        Write a field to memory
        """
        # fld_shape = (2,)+c.state.info.shape if rec.is_vector else c.state.info.shape
        # assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

        # if rec.is_vector:
        #     fld_ = fld[:, ~c.grid.mask].flatten()
        # else:
        #     fld_ = fld[~c.grid.mask]

        # binfile = self.binfile_name(c, tag)
        # with open(binfile, 'r+b') as f:
        #     f.seek(member*c.state.info.size + rec.pos)
        #     f.write(struct.pack(fld_.size*type_dic[rec.dtype], *fld_))

    def call_model_io(self, c: Context, model_name: str, method: str, **kwargs):
        # obtain method
        model = c.models[model_name]
        func = getattr(model, method)

        return func(**kwargs)