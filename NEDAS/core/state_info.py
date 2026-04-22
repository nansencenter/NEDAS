import numpy as np
from NEDAS.utils.conversion import type_size, t2h, h2t, dt1h, ensure_list
from NEDAS.core.context import Context
from NEDAS.core.types import FieldRecord

class StateInfo:
    """
    Manages the metadata, indexing, and memory offsets for the model state.

    Attributes:
        shape (tuple): domain dimension(s) for the fields
        fields (dict[int, FieldRecord]): dictionary containing field ids and the corresponding field records
        size (int): total size of the complete state (bytes), for one member
        variables (set[str]): set of unique variables in the state
        err_types (set[str]): set of unique error models in the state
    """
    shape: tuple
    mask: np.ndarray
    fields: dict[int, FieldRecord]
    size: int
    variables: list[str]
    err_types: list[str]

    def __init__(self, c: Context):
        """Parse the configuration to generate the state info object"""
        self.shape = c.grid.x.shape
        self.mask = c.grid.mask
        self.fields = {}
        # self.scalars: Dict[int, ScalarRecord] = {}
        self.size = 0
        variables = set()
        err_types = set()
        self.pos = 0  # seek position for rec

        # loop through variables in state_def
        for vrec in ensure_list(c.config.state_def):
            vname = vrec['name']
            variables.add(vname)

            vtype = vrec['var_type']
            err_types.add(vrec['err_type'])

            if vtype == 'field':
                self.add_fields_for_variable(c, vrec)

            elif vtype == 'scalar':
                pass

            else:
                raise NotImplementedError(f"{vtype} is not supported in the state vector.")

        # convert set to list, for indexing later
        self.variables = list(variables)
        self.err_types = list(err_types)

    def add_fields_for_variable(self, c: Context, vrec: dict) -> None:
        """
        Add fields for a variable in the state. The state variable has dimensions t, z, y, x
        while the 'field' is the 2D part with y, x dimensions.

        Args:
            c (Context): the runtime context object
            vrec (dict): the variable record defining its properties
        """
        vname = vrec['name']
        model_name = vrec['model_src']
        model = c.models[model_name]
        if vname not in model.variables:
            raise RuntimeError(f"variable '{vname}' not defined in {model_name} model.variables")

        #now go through time (t) and zlevels (k) to form a uniq field record
        time_steps = c.time + np.array(c.config.state_time_steps)*dt1h
        rec_id = len(self.fields)
        for time in time_steps:
            for k in model.variables[vname].levels:
                rec = FieldRecord(
                    name=vname,
                    model_src=vrec['model_src'],
                    dtype=model.variables[vname].dtype,
                    is_vector=model.variables[vname].is_vector,
                    units=model.variables[vname].units,
                    err_type=vrec['err_type'],
                    time=time,
                    dt=c.config.state_time_scale,
                    k=k,
                    pos=self.pos,
                )
                self.fields[rec_id] = rec

                # update seek position
                nv = 2 if rec.is_vector else 1
                fld_size = np.sum((~self.mask).astype(int))  # size of this 2D field
                self.pos += nv * fld_size * type_size[rec.dtype]
                rec_id += 1

        #update total size
        self.size = self.pos

    def __repr__(self):
        return (f"StateInfo(nfld={len(self.fields)}, "
                f"size={self.size} bytes, "
                f"variables={list(self.variables)})")

    def write_to_file(self, binfile: str):
        """
        Write the info to a .dat file accompanying the .bin file

        Args:
            binfile (str): File path for the .bin file
        """
        with open(binfile.replace('.bin','.dat'), 'wt') as f:
            # first line: grid dimension
            if len(self.shape) == 1:
                f.write(f"{self.shape[0]}\n")
            else:
                f.write(f"{self.shape[0]} {self.shape[1]}\n")

            # second line: total size of the state
            f.write(f"{self.size}\n")

            # followed by nfield lines: each for a field record
            for i, rec in self.fields.items():
                f.write(f"{rec.name} {rec.model_src} {rec.dtype} {int(rec.is_vector)} {rec.units} {rec.err_type} {t2h(rec.time)} {rec.dt} {rec.k} {rec.pos}\n")

    def read_from_file(self, binfile: str):
        """
        Read .dat file accompanying the .bin file and updates state_info

        Args:
            binfile (str): File path for the .bin file
        """
        with open(binfile.replace('.bin','.dat'), 'r') as f:
            lines = f.readlines()

            ss = lines[0].split()
            if len(ss)==1:
                self.shape = (int(ss[0]),)
            else:
                self.shape = (int(ss[0]), int(ss[1]))

            self.size = int(lines[1])

            # records for uniq fields
            self.fields = {}
            rec_id = 0
            for lin in lines[2:]:
                ss = lin.split()
                self.fields[rec_id] = FieldRecord(
                    name=ss[0],
                    model_src=ss[1],
                    dtype=ss[2],
                    is_vector=bool(int(ss[3])),
                    units=ss[4],
                    err_type=ss[5],
                    time=h2t(float(ss[6])),
                    dt=float(ss[7]),
                    k=int(ss[8]),
                    pos=int(ss[9])
                    )
                rec_id += 1
