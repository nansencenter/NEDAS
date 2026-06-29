import numpy as np
from NEDAS.utils.conversion import dt1h, ensure_list, type_size, t2h, h2t
from .types import ErrorModel, ObsRecord
from .context import Context

class ObsInfo:
    """
    Manages the metadata, indexing and memory allocation for the observation sequences

    Attributes:
        records (dict[int], ObsRecord]): dictionary containing obs_rec_id and the corresponding obs record
        variables (set[str]): set of unique variables in the observations
        err_types (set[str]): set of unique error models used in the observations
    """
    records: dict[int, ObsRecord]
    variables: list[str]
    err_types: list[str]

    def __init__(self, c: Context):
        """
        Parse the configuration to generate the observation info object.

        Args:
            c (Context): the runtime context object.

        Returns:
            dict: A dictionary with some dimensions and list of unique obs records
        """
        self.records = {}
        variables = set()
        err_types = set()

        # loop through variables in obs_def
        for vrec in ensure_list(c.config.obs_def):
            vname = vrec['name']
            variables.add(vname)

            if 'err' not in vrec or vrec['err'] is None:
                vrec['err'] = {}
            assert isinstance(vrec.get('err'), dict), f"obs_def: {vname}: expect 'err' to be a dictionary"
            err_types.add(vrec['err'].get('type', 'normal'))

            self.add_obs_record(c, vrec)

        # convert set to list, for later indexing
        self.variables = list(variables)
        self.err_types = list(err_types)

        self.complete_err_cross_corr_matrix()

        c.debug_message = f"number of unique observation records = {len(self.records)}"
        c.debug_message = f"observation variables: {self.variables}"

    def add_obs_record(self, c: Context, vrec: dict):
        """
        Add observation record

        Args:
            c (Context): the runtime context object
            vrec (dict): the observation record defining its properties
        """
        vname = vrec['name']
        dataset = c.datasets[vrec['dataset_src']]
        variables = dataset.variables
        assert vname in variables, 'variable '+vname+' not defined in '+vrec['dataset_src']+'.dataset.variables'

        # parse impact of obs on each state variable, default is 1.0 on all variables unless set by obs_def record
        impact_on_state = {}
        for state_name in c.state.info.variables:
            impact_on_state[state_name] = 1.0
        if 'impact_on_state' in vrec and vrec['impact_on_state'] is not None:
            for state_name, impact_fac in vrec['impact_on_state'].items():
                impact_on_state[state_name] = impact_fac

        # loop through time steps in obs window
        time_steps = c.time + np.array(c.config.obs_time_steps)*dt1h
        rec_id = len(self.records)
        for time in time_steps:
            err_opts = vrec['err']
            err = ErrorModel(
                type=err_opts.get('type', 'normal'),
                std=err_opts.get('std', 1.),
                hcorr=err_opts.get('hcorr',0.),
                vcorr=err_opts.get('vcorr',0.),
                tcorr=err_opts.get('tcorr',0.),
                cross_corr=err_opts.get('cross_corr',{}),
            )
            rec = ObsRecord(
                name=vname,
                dataset_src=vrec['dataset_src'],
                model_src=vrec['model_src'],
                nobs=vrec.get('nobs', 0),  # for synthetic observation use only, real obs will count nobs later in prepare_obs
                obs_window_min=vrec.get('obs_window_min', dataset.obs_window_min),
                obs_window_max=vrec.get('obs_window_max', dataset.obs_window_max),
                dtype=variables[vname].dtype,
                is_vector=variables[vname].is_vector,
                units=variables[vname].units,
                z_units=variables[vname].z_units,
                time=time,
                dt=0,
                err=err,
                hroi=vrec['hroi'] * c.config.localize_scale_fac[c.iter],
                vroi=vrec['vroi'],
                troi=vrec['troi'],
                impact_on_state=impact_on_state,
            )
            self.records[rec_id] = rec

    def complete_err_cross_corr_matrix(self):
        """Go through the obs error cross correlation matrix again to fill in the default values"""
        for obs_rec_id, obs_rec in self.records.items():
            if not isinstance(obs_rec.err.cross_corr, dict):
                raise TypeError(f"obs_def: {obs_rec.name} has err.cross_corr defined as {obs_rec.err.cross_corr}, expecting a dictionary")
            for vname in self.variables:
                if vname not in obs_rec.err.cross_corr:
                    if vname == obs_rec.name:
                        obs_rec.err.cross_corr[vname] = 1.0
                    else:
                        obs_rec.err.cross_corr[vname] = 0.0
                else:
                    if not isinstance(obs_rec.err.cross_corr[vname], float):
                        raise TypeError(f"obs_def: {obs_rec.name} has err.cross_corr.{vname} defined as {obs_rec.err.cross_corr[vname]}, expecting a float")

    def finalize_pos(self):
        """Compute byte offsets and total size once rec.nobs is known (after prepare_obs)."""
        offset = 0
        for rec in self.records.values():
            nv = 2 if rec.is_vector else 1
            rec.pos = offset
            offset += nv * rec.nobs * type_size[rec.dtype]
        self.size = offset

    def write_to_file(self, binfile: str) -> None:
        """
        Write binary-reading metadata to the .dat file.
        
        Columns: name dataset_src model_src dtype is_vector units z_units
                 time dt obs_window_min obs_window_max nobs pos
        """
        with open(binfile.replace('.bin', '.dat'), 'wt') as f:
            f.write(f"{len(self.records)}\n")
            f.write(f"{self.size}\n")
            for rec in self.records.values():
                f.write(
                    f"{rec.name} {rec.dataset_src} {rec.model_src} "
                    f"{rec.dtype} {int(rec.is_vector)} "
                    f"{rec.units} {rec.z_units} "
                    f"{t2h(rec.time)} {rec.dt} "
                    f"{rec.obs_window_min} {rec.obs_window_max} "
                    f"{rec.nobs} {rec.pos}\n"
                )

    def read_from_file(self, binfile: str) -> None:
        """
        Read .dat file; updates existing records or reconstructs from scratch.

        Column layout (0-based):
          0:name 1:dataset_src 2:model_src 3:dtype 4:is_vector 5:units 6:z_units
          7:time 8:dt 9:obs_window_min 10:obs_window_max 11:nobs 12:pos
        """
        from .types import ErrorModel, ObsRecord
        with open(binfile.replace('.bin', '.dat'), 'r') as f:
            lines = f.readlines()
        nrec = int(lines[0])
        self.size = int(lines[1])
        for rec_id, line in enumerate(lines[2:2 + nrec]):
            ss = line.split()
            if rec_id in self.records:
                rec = self.records[rec_id]
            else:
                rec = ObsRecord(
                    name=ss[0], dataset_src=ss[1], model_src=ss[2],
                    dtype=ss[3], is_vector=bool(int(ss[4])),
                    units=ss[5], z_units=ss[6],
                    err=ErrorModel(type='normal', std=1., hcorr=0., vcorr=0., tcorr=0., cross_corr={}),
                    time=h2t(float(ss[7])), dt=float(ss[8]),
                    obs_window_min=int(ss[9]), obs_window_max=int(ss[10]),
                    hroi=0., vroi=0., troi=0.,
                    nobs=int(ss[11]), pos=int(ss[12]),
                    impact_on_state={},
                )
                self.records[rec_id] = rec
            rec.nobs = int(ss[11])
            rec.pos  = int(ss[12])
