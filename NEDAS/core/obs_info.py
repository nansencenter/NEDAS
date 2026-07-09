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
        obs_def_list = ensure_list(c.config.obs_def)
        variables = set()
        err_types = set()

        # first pass: collect the full set of obs variables/err types before
        # constructing any records, so add_obs_record (second pass, below)
        # can build each record's cross_corr tuple already ordered against
        # the final self.variables list, rather than needing a separate
        # after-the-fact completion step over per-record dicts.
        for vrec in obs_def_list:
            vname = vrec['name']
            variables.add(vname)
            if 'err' not in vrec or vrec['err'] is None:
                vrec['err'] = {}
            assert isinstance(vrec.get('err'), dict), f"obs_def: {vname}: expect 'err' to be a dictionary"
            err_types.add(vrec['err'].get('type', 'normal'))

        # convert set to list, for later indexing
        self.variables = list(variables)
        self.err_types = list(err_types)

        # second pass: now self.variables is final, build records (and their
        # cross_corr/impact_on_variable tuples) against it
        for vrec in obs_def_list:
            self.add_obs_record(c, vrec)

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

        # cross-variable error correlation, positional against self.variables
        # (already finalized by the time add_obs_record runs -- see
        # ObsInfo.__init__'s two-pass construction). Default: 1.0 with self,
        # 0.0 with everything else, same as the old per-record dict default.
        cross_corr_opts = vrec['err'].get('cross_corr', {}) or {}
        if not isinstance(cross_corr_opts, dict):
            raise TypeError(f"obs_def: {vname} has err.cross_corr defined as {cross_corr_opts}, expecting a dictionary")
        cross_corr = []
        for vname2 in self.variables:
            if vname2 in cross_corr_opts:
                val = cross_corr_opts[vname2]
                if not isinstance(val, float):
                    raise TypeError(f"obs_def: {vname} has err.cross_corr.{vname2} defined as {val}, expecting a float")
                cross_corr.append(val)
            else:
                cross_corr.append(1.0 if vname2 == vname else 0.0)
        cross_corr = tuple(cross_corr)

        # impact of this obs on each state variable, positional against
        # state.info.variables; user specifies overrides, default is 1.0.
        # c.state may not exist yet for standalone/diagnostic-only Obs(c)
        # construction (e.g. diag/plot/observations.py) -- impact tuple is
        # simply empty in that case, since nothing on that path performs a
        # multivariate state update anyway.
        if hasattr(c, 'state') and hasattr(c.state, 'info'):
            state_variables = c.state.info.variables
        else:
            state_variables = []
        impact_opts = vrec.get('impact_on_variable') or {}
        impact_on_variable = tuple(impact_opts.get(svname, 1.0) for svname in state_variables)

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
                cross_corr=cross_corr,
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
                impact_on_variable=impact_on_variable,
            )
            self.records[rec_id] = rec

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
                    err=ErrorModel(type='normal', std=1., hcorr=0., vcorr=0., tcorr=0., cross_corr=()),
                    time=h2t(float(ss[7])), dt=float(ss[8]),
                    obs_window_min=int(ss[9]), obs_window_max=int(ss[10]),
                    hroi=0., vroi=0., troi=0.,
                    nobs=int(ss[11]), pos=int(ss[12]),
                    impact_on_variable=(),
                )
                self.records[rec_id] = rec
            rec.nobs = int(ss[11])
            rec.pos  = int(ss[12])
