import numpy as np
from NEDAS.utils.conversion import dt1h, ensure_list
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

        ##loop through variables in obs_def
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

        if c.config.debug:
            print(f"number of unique observation records = {len(self.records)}", flush=True)
            print(f"observation variables: {self.variables}", flush=True)

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

        ##parse impact of obs on each state variable, default is 1.0 on all variables unless set by obs_def record
        impact_on_state = {}
        for state_name in c.state.info.variables:
            impact_on_state[state_name] = 1.0
        if 'impact_on_state' in vrec and vrec['impact_on_state'] is not None:
            for state_name, impact_fac in vrec['impact_on_state'].items():
                impact_on_state[state_name] = impact_fac

        ##loop through time steps in obs window
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
                nobs=vrec.get('nobs', 0),  ##for synthetic observation use only, real obs will count nobs later in prepare_obs
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

    # def write_obs_info(self, binfile):
    #     with open(binfile.replace('.bin','.dat'), 'wt') as f:
    #         f.write('{} {}\n'.format(self.info['nobs'], self.info['nens']))
    #         for rec in self.info['obs_seq'].values():
    #             f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(rec['name'], rec['dataset_src'], rec['model_src'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['z_units'], rec['x'], rec['y'], rec['z'], t2h(rec['time']), rec['pos']))

    # ##read obs_info from the dat file
    # def read_obs_info(self, binfile):
    #     with open(binfile.replace('.bin','.dat'), 'r') as f:
    #         lines = f.readlines()

    #         ss = lines[0].split()
    #         self.info = {'nobs':int(ss[0]), 'nens':int(ss[1]), 'obs_seq':{}}

    #         ##following lines of obs records
    #         obs_id = 0
    #         for lin in lines[1:]:
    #             ss = lin.split()
    #             rec = {'name': ss[0],
    #                 'dataset_src': ss[1],
    #                 'model_src': ss[2],
    #                 'dtype': ss[3],
    #                 'is_vector': bool(int(ss[4])),
    #                 'units': ss[5],
    #                 'z_units':ss[6],
    #                 'err_type': ss[7],
    #                 'err': float(ss[8]),
    #                 'x': float(ss[9]),
    #                 'y': float(ss[10]),
    #                 'z': float(ss[11]),
    #                 'time': h2t(float(ss[12])),
    #                 'pos': int(ss[13]), }
    #             self.info['obs_seq'][obs_id] = rec
    #             obs_id += 1
