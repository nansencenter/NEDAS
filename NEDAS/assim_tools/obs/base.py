import os
import numpy as np
from NEDAS.utils.conversion import t2h, h2t, dt1h, ensure_list
from NEDAS.utils.progress import progress_bar
from NEDAS.utils.parallel import bcast_by_root, distribute_tasks

class Obs:
    """
    Class for handling observations.
    
    The observation has dimensions: variable, time, z, y, x
    Since the observation network is typically irregular, we store the obs record
    for each variable in a 1d sequence, with coordinates (t,z,y,x), and size nobs

    To parallelize workload, we distribute each obs record over all the processors

    for batch assimilation mode, each pid stores the list of local obs within the
    hroi of its tiles, with size nlobs (number of local obs)
    
    for serial mode, each pid stores a non-overlapping subset of the obs list,
    here 'local' obs (in storage sense) is broadcast to all pid before computing
    its update to the state/obs near that obs.

    The hroi is separately defined for each obs record.
    For very large hroi, the serial mode is more parallel efficient option, since
    in batch mode the same obs may need to be stored in multiple pids

    To compare to the observation, obs_prior simulated by the model needs to be
    computed, they have dimension [nens, nlobs], indexed by (mem_id, obs_id)
    """
    def __init__(self, c, state):
        self.analysis_dir = c.analysis_dir(c.time, c.iter)

        self.info = bcast_by_root(c.comm)(self.parse_obs_info)(c, state)

        self.obs_rec_list = bcast_by_root(c.comm)(self.distribute_obs_tasks)(c)

        self.obs_inds = {}         ##will be created by assimilator.assign_obs()
        self.obs_seq = {}          ##will be created by prepare_obs()
        self.obs_prior_seq = {}    ##will be created by prepare_obs_from_state()
        self.lobs = {}             ##will be created by transpose_to_ensemble_complete()
        self.lobs_prior = {}
        self.lobs_post = {}        ##will be created by assimilator.assimilate()
        self.obs_post_seq = {}     ##will be created by transpose_to_field_complete()
        self.data = {}             ##will be created by pack_obs_data, for use in assimilate()

    def parse_obs_info(self, c, state):
        """
        Parse info for the observation records defined in config.

        Args:
            c (Config): Configuration object.
            state (State): State object.

        Returns:
            dict: A dictionary with some dimensions and list of unique obs records
        """
        info = {'size':0, 'records':{}}
        obs_variables = set()
        obs_err_types = set()
        obs_rec_id = 0  ##record id for an obs sequence
        pos = 0         ##seek position for rec

        ##loop through obs variables defined in obs_def
        for vrec in ensure_list(c.obs_def):
            vname = vrec['name']
            if 'err' not in vrec or vrec['err'] is None:
                vrec['err'] = {}
            assert isinstance(vrec.get('err'), dict), f"obs_def: {vname}: expect 'err' to be a dictionary"
            obs_err_type = vrec['err'].get('type', 'normal')

            ##some properties of the variable is defined in its source module
            dataset = c.datasets[vrec['dataset_src']]
            variables = dataset.variables
            assert vname in variables, 'variable '+vname+' not defined in '+vrec['dataset_src']+'.dataset.variables'

            ##parse impact of obs on each state variable, default is 1.0 on all variables unless set by obs_def record
            impact_on_state = {}
            for state_name in state.info['variables']:
                impact_on_state[state_name] = 1.0
            if 'impact_on_state' in vrec and vrec['impact_on_state'] is not None:
                for state_name, impact_fac in vrec['impact_on_state'].items():
                    impact_on_state[state_name] = impact_fac

            ##loop through time steps in obs window
            for time in c.time + np.array(c.obs_time_steps)*dt1h:
                obs_rec = {'name': vname,
                        'dataset_src': vrec['dataset_src'],
                        'model_src': vrec['model_src'],
                        'nobs': vrec.get('nobs', 0),
                        'obs_window_min': vrec.get('obs_window_min', dataset.obs_window_min),
                        'obs_window_max': vrec.get('obs_window_max', dataset.obs_window_max),
                        'dtype': variables[vname]['dtype'],
                        'is_vector': variables[vname]['is_vector'],
                        'units': variables[vname]['units'],
                        'z_units': variables[vname]['z_units'],
                        'time': time,
                        'dt': 0,
                        'pos': pos,
                        'err':{'type': obs_err_type,
                               'std': vrec['err'].get('std', 1.),  ##for synthetic obs perturb, real obs will have std from dataset
                               'hcorr': vrec['err'].get('hcorr',0.),
                               'vcorr': vrec['err'].get('vcorr',0.),
                               'tcorr': vrec['err'].get('tcorr',0.),
                               'cross_corr': vrec['err'].get('cross_corr',{}),
                               },
                        'hroi': vrec['hroi'] * c.localize_scale_fac[c.iter],
                        'vroi': vrec['vroi'],
                        'troi': vrec['troi'],
                        'impact_on_state': impact_on_state,
                        }
                obs_variables.add(vname)
                obs_err_types.add(obs_err_type)
                info['records'][obs_rec_id] = obs_rec

                ##update obs_rec_id
                obs_rec_id += 1

                ##we don't know the size of obs_seq yet
                ##will wait for prepare_obs to update the seek position

        info['variables'] = list(obs_variables)
        info['err_types'] = list(obs_err_types)

        ##go through the obs_rec again to fill in the default err.cross_corr
        for obs_rec_id, obs_rec in info['records'].items():
            assert isinstance(obs_rec['err']['cross_corr'], dict), f"obs_def: {obs_rec['name']} has err.cross_corr defined as {obs_rec['err']['cross_corr']}, expecting a dictionary"
            for vname in info['variables']:
                if vname not in obs_rec['err']['cross_corr']:
                    if vname == obs_rec['name']:
                        obs_rec['err']['cross_corr'][vname] = 1.0
                    else:
                        obs_rec['err']['cross_corr'][vname] = 0.0
                else:
                    assert isinstance(obs_rec['err']['cross_corr'][vname], float), f"obs_def: {obs_rec['name']} has err.cross_corr.{vname} defined as {obs_rec['err']['cross_corr'][vname]}, expecting a float"

        return info

    def distribute_obs_tasks(self, c):
        """
        Distribute obs_rec_id across processors

        Args:
            c (Config): Configuration object.

        Returns:
            dict: Dictionary {pid_rec (int): list[obs_rec_id (int)]}
        """
        obs_rec_list_full = [i for i in self.info['records'].keys()]
        obs_rec_size = np.array([2 if r['is_vector'] else 1 for i,r in self.info['records'].items()])
        obs_rec_list = distribute_tasks(c.comm_rec, obs_rec_list_full, obs_rec_size)

        return obs_rec_list

    def read_mean_z_coords(self, c, state, time):
        """
        Read the ensemble-mean z coords from z_file at obs time

        Inputs:
            c (Config): Configuration object.
            state (State): State object.
            time (datetime): observation time.

        Returns:
            np.ndarray: Z-coordinate fields of shape (nz, c.grid.x.shape) for all unique levels defined in state.info
        """
        ##first, get a list of indices k
        k_list = list(set([r['k'] for i,r in state.info['fields'].items() if r['time']==time]))

        ##get z coords for each level
        z = np.zeros((len(k_list),)+state.info['shape'])
        for k in range(len(k_list)):

            ##the rec_id in z_file corresponding to this level
            ##there can be multiple records (different state variables)
            ##we take the first one
            rec_id = [i for i,r in state.info['fields'].items() if r['time']==time and r['k']==k_list[k]][0]

            ##read the z field with (mem_id=0, rec_id) from z_file
            z_fld = state.read_field(state.z_coords_file, c.grid.mask, 0, rec_id)

            ##assign the coordinates to z(k)
            if state.info['fields'][rec_id]['is_vector']:
                z[k, ...] = z_fld[0, ...]
            else:
                z[k, ...] = z_fld

        return z

    def state_to_obs(self, c, state, flag, **kwargs):
        """
        Compute the corresponding obs value given the state variable(s), namely the "obs_prior"
        This function includes several ways to compute the obs_prior:

        1, If obs_name is one of the variables provided by the model_src module, then
        model_src.read_var shall be able to provide the obs field defined on model native grid.
        Then we convert the obs field to the analysis grid and do vertical interpolation.

        Note that there are options to obtain the obs fields to speed things up:
        
        1.1, If the obs field can be found in the locally stored fields[mem_id, rec_id],
                we can just access it from memory (quickest)
        
        1.2, If the obs field is not in local memory, but still one of the state variables,
                we can get it through read_field() from the state_file
        
        1.3, If the obs is not one of the state variable, we get it through model_src.read_var()

        2, If obs_name is one of the variables provided by obs.obs_operator, we call it to
        obtain the obs seq. Typically the obs_operator performs more complex computation, such
        as path integration, radiative transfer model, etc. (slowest)

        When using synthetic observations in experiments, this function generates the
        unperturbed obs from the true states (nature runs). Setting "member=None" in the inputs
        indicates that we are trying to generate synthetic observations. Since the state variables
        won't include the true state, the only options above are 1.3 or 2 for this case.

        Args:
            c (Config): config object
            state (State): state object
            flag (str): 'prior' or 'posterior'
            **kwargs: Additional parameters
                - member: int, member index; or None if dealing with synthetic obs
                - name: str, obs variable name
                - time: datetime obj, time of the obs window
                - is_vector: bool, if True the obs is a vector measurement
                - dataset_src: str, dataset source module name providing the obs
                - model_src: str, model source module name providing the state
                - x, y, z, t: np.array, coordinates from obs_seq

        Returns:
            np.ndarray: Values corresponding to the obs_seq but from the state identified by kwargs
        """
        mem_id = kwargs['member']
        synthetic = True if mem_id is None else False
        time = kwargs['time']
        obs_name = kwargs['name']
        is_vector = kwargs['is_vector']

        ##obs dataset source module
        dataset = c.datasets[kwargs['dataset_src']]
        ##model source module
        model = c.models[kwargs['model_src']]

        ##option 1  TODO: update README and comments
        ## if dataset module provides an obs_operator, use it to compute obs
        if hasattr(dataset, 'obs_operator') and obs_name in dataset.obs_operator:
            if synthetic:
                path = model.truth_dir
            else:
                path = c.forecast_dir(time, kwargs['model_src'])

            operator = dataset.obs_operator[kwargs['name']]

            ##get the obs seq from operator
            seq = operator(path=path, model=model, grid=c.grid, mask=c.grid.mask, **kwargs)

        ##option 2:
        ## if obs variable is one of the state variable, or can be computed by the model,
        ## then we just need to collect the 3D variable and interpolate in x,y,z
        elif obs_name in model.variables:

            obs_x = np.array(kwargs['x'])
            obs_y = np.array(kwargs['y'])
            obs_z = np.array(kwargs['z'])
            nobs = len(obs_x)
            if is_vector:
                seq = np.full((2, nobs), np.nan)
            else:
                seq = np.full(nobs, np.nan)

            levels = model.variables[obs_name]['levels']
            for k in range(len(levels)):
                if obs_name in [r['name'] for r in ensure_list(c.state_def)] and not synthetic:
                    ##the obs is one of the state variables
                    ##find its corresponding rec_id
                    rec_id = [i for i,r in state.info['fields'].items() if r['name']==obs_name and r['k']==levels[k]][0]

                    ##option 1.1: if the current pid stores this field, just read it
                    ##TODO: can merge the two into state property getters
                    if rec_id in state.rec_list[c.pid_rec] and mem_id in state.mem_list[c.pid_mem]:
                        z = state.z_fields[mem_id, rec_id]
                        if flag == 'prior':
                            fld = state.fields_prior[mem_id, rec_id]
                        elif flag == 'posterior':
                            fld = state.fields_post[mem_id, rec_id]
                        else:
                            raise ValueError

                    else:  ##option 1.2: read field from state binfile
                        z = state.read_field(state.z_coords_file, c.grid.mask, 0, rec_id)
                        if flag == 'prior':
                            fld = state.read_field(state.prior_file, c.grid.mask, mem_id, rec_id)
                        elif flag == 'posterior':
                            fld = state.read_field(state.post_file, c.grid.mask, mem_id, rec_id)
                        else:
                            raise ValueError

                else:  ##option 1.3: get the field from model.read_var
                    if synthetic:
                        path = model.truth_dir
                    else:
                        path = c.forecast_dir(time, kwargs['model_src'])

                    if k == 0:  ##initialize grid obj for conversion
                        model.read_grid(path=path, **kwargs)
                        model.grid.set_destination_grid(c.grid)

                    model_z = model.z_coords(path=path, member=kwargs['member'], time=kwargs['time'], k=levels[k])
                    z_ = model.grid.convert(model_z, method='nearest')
                    z = np.array([z_, z_]) if is_vector else z_

                    model_fld = model.read_var(path=path, name=kwargs['name'], member=kwargs['member'], time=kwargs['time'], k=levels[k])
                    fld = model.grid.convert(model_fld, is_vector=is_vector, method='nearest')

                ##horizontal interp field to obs_x,y, for current layer k
                if is_vector:
                    z = c.grid.interp(z[0, ...], obs_x, obs_y, method='nearest')
                    zc = np.array([z, z])
                    v1 = c.grid.interp(fld[0, ...], obs_x, obs_y, method='nearest')
                    v2 = c.grid.interp(fld[1, ...], obs_x, obs_y, method='nearest')
                    vc = np.array([v1, v2])
                else:
                    zc = c.grid.interp(z, obs_x, obs_y, method='nearest')
                    vc = c.grid.interp(fld, obs_x, obs_y, method='nearest')

                ##vertical interp to obs_z, take ocean depth as example:
                ##    -------------------------------------------
                ##k-1 -  -  -  -  v[k-1], vp  }dzp  prevous layer
                ##    ----------  z[k-1], zp --------------------
                ##k   -  -  -  -  v[k],   vc  }dzc  current layer
                ##    ----------  z[k],   zc --------------------
                ##k+1 -  -  -  -  v[k+1]
                ##
                ##z at current level k denoted as zc, previous level as zp
                ##variables are considered layer averages, so they are at layer centers
                if k == 0:
                    dzc = zc  ##layer thickness for first level

                    ##the first level: constant vc from z=0 to dzc/2
                    inds = (obs_z >= np.minimum(0, 0.5*dzc)) & (obs_z <= np.maximum(0, 0.5*dzc))
                    seq[..., inds] = vc[..., inds]

                if k > 0:
                    dzc = zc - zp  ##layer thickness for level k

                    ##in between levels: linear interp between vp and vc
                    z_vp = zp - 0.5*dzp
                    z_vc = zp + 0.5*dzc
                    inds = (obs_z >= np.minimum(z_vp, z_vc)) & (obs_z <= np.maximum(z_vp, z_vc))
                    ##there can be collapsed layers if z_vc=z_vp
                    zdiff = z_vc - z_vp
                    collapsed = (zdiff == 0)
                    zdiff = np.where(collapsed, 1, zdiff)
                    vi = ((z_vc - obs_z)*vc + (obs_z - z_vp)*vp) / zdiff
                    vi = np.where(collapsed, vp, vi)

                    ##TODO: still got some warnings here
                    #with np.errstate(divide='ignore'):
                    #    vi = np.where(z_vp==z_vc,
                    #                vp,   ##for collapsed layers just use previous value
                    #                ((z_vc - obs_z)*vc + (obs_z - z_vp)*vp)/(z_vc - z_vp) )  ##otherwise linear interp between layer
                    seq[..., inds] = vi[..., inds]

                if k == len(levels)-1:
                    ##the last level: constant vc from z=zc-dzc/2 to zc
                    inds = (obs_z >= np.minimum(zc-0.5*dzc, zc)) & (obs_z <= np.maximum(zc-0.5*dzc, zc))
                    seq[..., inds] = vc[..., inds]

                if k < len(levels)-1:
                    ##make a copy of current layer as 'previous' for next k
                    zp = zc.copy()
                    vp = vc.copy()
                    dzp = dzc.copy()

        else:
            raise ValueError('unable to obtain obs prior for '+obs_name)

        return seq

    def collect_obs_seq(self, c, state):
        """
        Process the obs in parallel, read dataset files and convert to obs_seq
        which contains obs value, coordinates and other info

        Since this is the actual obs (1 copy), only 1 processor needs to do the work

        Argss:
            c (Config): Configuration object.
            state (State): State object.

        Returns:
            dict: observation sequence. Dictionary {obs_rec_id (int): record}
                where each record is a dictionary {key: np.ndarray}, the mandatory keys are
                'obs' the observed values (measurements)
                'x', 'y', 'z', 't' the coordinates for each measurement
                'err_std' the uncertainties for each measurement
                there can be other optional keys provided by read_obs() but we don't use them
        """
        if c.pid == 0:
            print('>>> read observation sequence from datasets', flush=True)

        ##get obs_seq from dataset module, each pid_rec gets its own workload as a subset of obs_rec_list
        obs_seq = {}
        for obs_rec_id in self.obs_rec_list[c.pid_rec]:
            obs_rec = self.info['records'][obs_rec_id]

            ##load the dataset module
            dataset = c.datasets[obs_rec['dataset_src']]
            assert obs_rec['name'] in dataset.variables, 'variable '+obs_rec['name']+' not defined in dataset.'+obs_rec['dataset_src']+'.variables'

            model = c.models[obs_rec['model_src']]
            ##read ens-mean z coords from z_file for this obs network
            ##typically model.z_coords can compute the z coords as well, but it is more efficient
            ##to just store the ensemble mean z here
            model.z = self.read_mean_z_coords(c, state, obs_rec['time'])

            if c.use_synthetic_obs:
                ##generate synthetic obs network
                seq = dataset.random_network(model=model, grid=c.grid, mask=c.grid.mask, **obs_rec)

                ##compute obs values
                seq['obs'] = self.state_to_obs(c, state, 'prior', member=None, **obs_rec, **seq)

                ##perturb with obs err
                seq['obs'] += np.random.normal(0, 1, seq['obs'].shape) * obs_rec['err']['std']

            else:
                ##read dataset files and obtain obs sequence
                seq = dataset.read_obs(model=model, grid=c.grid, mask=c.grid.mask, **obs_rec)

            if c.pid_mem == 0:
                print('number of '+obs_rec['name']+' obs from '+obs_rec['dataset_src']+': {}'.format(seq['obs'].shape[-1]), flush=True)

            ##misc. transform here
            for transform_func in c.transform_funcs:
                seq = transform_func.forward_obs(c, obs_rec, seq)

            obs_seq[obs_rec_id] = seq
            obs_rec['nobs'] = seq['obs'].shape[-1]  ##update nobs

        ##output obs sequence
        if c.pid_mem == 0:
            for obs_rec_id, rec in obs_seq.items():
                file = os.path.join(self.analysis_dir, f'obs_seq.rec{obs_rec_id}.npy')
                np.save(file, rec)

        return obs_seq

    def prepare_obs(self, c, state):
        self.obs_seq = bcast_by_root(c.comm_mem)(self.collect_obs_seq)(c, state)

    def prepare_obs_from_state(self, c, state, flag):
        """
        Compute the obs priors in parallel, run state_to_obs to obtain obs_prior_seq

        Args:
            c (Config): config object
            state (State): state object
            flag (str): 'prior' or 'posterior'

        Returns:
            dict: obs_prior_seq, Dictionary {(mem_id, obs_rec_id): seq}
                where seq is np.array with values corresponding to obs_seq['obs']
        """
        pid_mem_show = [p for p,lst in state.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show
        c.print_1p('>>> compute observation priors\n')

        ##process the obs, each proc gets its own workload as a subset of
        ##all proc goes through their own task list simultaneously
        nr = len(self.obs_rec_list[c.pid_rec])
        nm = len(state.mem_list[c.pid_mem])
        for m, mem_id in enumerate(state.mem_list[c.pid_mem]):
            for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):
                ##this is the obs record to process
                obs_rec = self.info['records'][obs_rec_id]

                if c.debug:
                    print(f"PID {c.pid:4}: obs_prior mem{mem_id+1:03} {obs_rec['name']:20}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nr*nm))

                seq = {}
                ##need the coordinates for transform later
                for key in ['x', 'y', 'z', 't', 'err_std']:
                    seq[key] = self.obs_seq[obs_rec_id][key]
                ##obtain obs_prior values from model state
                seq['obs'] = self.state_to_obs(c, state, flag, member=mem_id, **obs_rec, **self.obs_seq[obs_rec_id])

                ##misc. transform here
                for transform_func in c.transform_funcs:
                    seq = transform_func.forward_obs(c, obs_rec, seq)

                ##collect obs priors together
                if flag == 'prior':
                    self.obs_prior_seq[mem_id, obs_rec_id] = seq['obs']
                elif flag == 'posterior':
                    self.obs_post_seq[mem_id, obs_rec_id] = seq['obs']
                else:
                    raise ValueError
        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##output obs_prior sequeneces
        for key, seq in self.obs_prior_seq.items():
            mem_id, obs_rec_id = key
            file = os.path.join(self.analysis_dir, f'obs_prior_seq.rec{obs_rec_id}.mem{mem_id:03}.npy')
            np.save(file, seq)

    def global_obs_list(self, c):
        ##form the global list of obs (in serial mode the main loop is over this list)
        n_obs_rec = len(self.info['records'])

        i = {}  ##location in full obs vector on owner pid
        for owner_pid in range(c.nproc_mem):
            i[owner_pid] = 0

        obs_list = []
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info['records'][obs_rec_id]
            v_list = [0, 1] if obs_rec['is_vector'] else [None]
            for owner_pid in self.obs_inds[obs_rec_id].keys():
                for _ in self.obs_inds[obs_rec_id][owner_pid]:
                    for v in v_list:
                        obs_list.append((obs_rec_id, v, owner_pid, i[owner_pid]))
                        i[owner_pid] += 1

        if getattr(c, 'shuffle_obs', False):
            np.random.shuffle(obs_list)  ##randomize the order of obs (this is optional)

        return obs_list

    def write_obs_info(self, binfile):
        with open(binfile.replace('.bin','.dat'), 'wt') as f:
            f.write('{} {}\n'.format(self.info['nobs'], self.info['nens']))
            for rec in self.info['obs_seq'].values():
                f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(rec['name'], rec['dataset_src'], rec['model_src'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['z_units'], rec['x'], rec['y'], rec['z'], t2h(rec['time']), rec['pos']))

    ##read obs_info from the dat file
    def read_obs_info(self, binfile):
        with open(binfile.replace('.bin','.dat'), 'r') as f:
            lines = f.readlines()

            ss = lines[0].split()
            self.info = {'nobs':int(ss[0]), 'nens':int(ss[1]), 'obs_seq':{}}

            ##following lines of obs records
            obs_id = 0
            for lin in lines[1:]:
                ss = lin.split()
                rec = {'name': ss[0],
                    'dataset_src': ss[1],
                    'model_src': ss[2],
                    'dtype': ss[3],
                    'is_vector': bool(int(ss[4])),
                    'units': ss[5],
                    'z_units':ss[6],
                    'err_type': ss[7],
                    'err': np.float32(ss[8]),
                    'x': np.float32(ss[9]),
                    'y': np.float32(ss[10]),
                    'z': np.float32(ss[11]),
                    'time': h2t(np.float32(ss[12])),
                    'pos': int(ss[13]), }
                self.info['obs_seq'][obs_id] = rec
                obs_id += 1

    def transpose_to_ensemble_complete(self, c, state, input_obs, ensemble=False):
        """
        Transpose obs from field-complete to ensemble-complete

        Step 1, Within comm_mem, send the subset of input_obs with mem_id and par_id
        from the source proc (src_pid) to the destination proc (dst_pid), store the
        result in tmp_obs with all the mem_id (ensemble-complete)

        Step 2, Gather all obs_rec_id within comm_rec, so that each pid_rec will have the
        entire obs record for assimilation

        Args:
            c (Config): config obj
            input_obs (dict): obs_seq from process_all_obs() or obs_prior_seq from process_all_obs_priors()
            ensemble (bool): If True, the input_obs is the obs_prior_seq, dict[(mem_id, obs_rec_id), np.array];
                If False (default), the input_obs is the obs_seq, dict[obs_rec_id, dict[key, np.array]]     

        Returns,
            dict: output_obs
                If ensemble, returns the lobs_prior dict[(mem_id, obs_rec_id), dict[par_id, np.array]]
                If not ensemble, returns the lobs dict[obs_rec_id, dict[par_id, dict[key, np.array]]], key = 'obs','x','y','z','t'...
        """
        pid_mem_show = [p for p,lst in state.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show

        if ensemble:
            c.print_1p('obs prior sequences: ')
        else:
            c.print_1p('obs sequences: ')
        c.print_1p('transpose obs to local obs\n')

        ##Step 1: transpose to ensemble-complete by exchanging mem_id, par_id in comm_mem
        ##        input_obs -> tmp_obs
        tmp_obs = {}  ##local obs at intermediate stage

        nr = len(self.obs_rec_list[c.pid_rec])
        for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in state.mem_list.items()])
            for m in range(nm_max):
                if c.debug:
                    if m < len(state.mem_list[c.pid_mem]):
                        mem_id = state.mem_list[c.pid_mem][m]
                        print(f"PID {c.pid:4}: transposing obs: mem{mem_id+1:03} obs_rec{obs_rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing obs: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare the obs seq for sending if not at the end of mem_list
                if m < len(state.mem_list[c.pid_mem]):
                    mem_id = state.mem_list[c.pid_mem][m]
                    if ensemble:  ##this is the obs prior seq
                        seq = input_obs[mem_id, obs_rec_id].copy()
                    else:
                        if mem_id == 0:  ##this is the obs seq, just let mem_id=0 send it
                            seq = input_obs[obs_rec_id].copy()

                ##the collective send/recv follows the same idea under state.transpose_field_to_state
                ##1) receive lobs_seq from src_pid, for src_pid<pid first
                for src_pid in np.arange(0, c.pid_mem):
                    if m < len(state.mem_list[src_pid]):
                        src_mem_id = state.mem_list[src_pid][m]
                        if ensemble:
                            tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                        else:
                            if src_mem_id == 0:
                                tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

                ##2) send my obs chunk to a list of dst_pid, send to dst_pid>=pid first
                ##   then cycle back to send to dst_pid<pid. i.e. the dst_pid sequence is
                ##   [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
                if m < len(state.mem_list[c.pid_mem]):
                    for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                        if ensemble:
                            ##this is the obs prior seq for mem_id, obs_rec_id
                            ##for each par_id, assemble the subset lobs_seq using obs_inds
                            lobs_seq = {}
                            for par_id in state.par_list[dst_pid]:
                                inds = self.obs_inds[obs_rec_id][par_id]
                                lobs_seq[par_id] = seq[..., inds]

                            if dst_pid == c.pid_mem:
                                ##pid already stores the lobs_seq, just copy
                                tmp_obs[mem_id, obs_rec_id] = lobs_seq
                            else:
                                ##send lobs_seq to dst_pid
                                c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

                        else:
                            if mem_id == 0:
                                ##this is the obs seq with keys 'obs','err_std','x','y','z','t'
                                ##assemble the lobs_seq dict with same keys but subset obs_inds
                                ##do this for each par_id to get the full lobs_seq
                                lobs_seq = {}
                                for par_id in state.par_list[dst_pid]:
                                    lobs_seq[par_id] = {}
                                    inds = self.obs_inds[obs_rec_id][par_id]
                                    for key in ('obs', 'err_std', 'x', 'y', 'z', 't'):
                                        lobs_seq[par_id][key] = seq[key][..., inds]

                                if dst_pid == c.pid_mem:
                                    ##pid already stores the lobs_seq, just copy
                                    tmp_obs[obs_rec_id] = lobs_seq
                                else:
                                    ##send lobs_seq to dst_pid's lobs
                                    c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

                ##3) finish receiving lobs_seq from src_pid, for src_pid>pid now
                for src_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                    if m < len(state.mem_list[src_pid]):
                        src_mem_id = state.mem_list[src_pid][m]
                        if ensemble:
                            tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                        else:
                            if src_mem_id == 0:
                                tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##Step 2: collect all obs records (all obs_rec_ids) on pid_rec
        ##        tmp_obs -> output_obs
        output_obs = {}
        for entry in c.comm_rec.allgather(tmp_obs):
            for key, data in entry.items():
                output_obs[key] = data
        c.comm.Barrier()

        return output_obs

    def transpose_to_field_complete(self, c, state, lobs):
        """
        Transpose obs from ensemble-complete to field-complete

        Args:
            c (Config): config obj
            state (State): state obj
            lobs (dict): local obs seq

        Returns:
            dict: obs_seq, dictionary {(mem_id, obs_rec_id): np.ndarray}
        """
        pid_mem_show = [p for p,lst in state.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show

        c.print_1p('obs post sequences: ')
        c.print_1p('transpose local obs to obs\n')

        obs_seq = {}
        nr = len(self.obs_rec_list[c.pid_rec])
        for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in state.mem_list.items()])
            for m in range(nm_max):
                if c.debug:
                    if m < len(state.mem_list[c.pid_mem]):
                        mem_id = state.mem_list[c.pid_mem][m]
                        print(f"PID {c.pid:4}: transposing obs: mem{mem_id+1:03} obs_rec{obs_rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing obs: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare an empty obs_seq for receiving if not at the end of mem_list
                if m < len(state.mem_list[c.pid_mem]):
                    mem_id = state.mem_list[c.pid_mem][m]
                    rec = self.info['records'][obs_rec_id]
                    if rec['is_vector']:
                        seq = np.full((2, rec['nobs']), np.nan)
                    else:
                        seq = np.full((rec['nobs'],), np.nan)

                ##this is just the reverse of transpose_obs_to_lobs
                ## we take the exact steps, but swap send and recv operations here
                ##
                ## 1) send my lobs to dst_pid, for dst_pid<pid first
                for dst_pid in np.arange(0, c.pid_mem):
                    if m < len(state.mem_list[dst_pid]):
                        dst_mem_id = state.mem_list[dst_pid][m]
                        c.comm_mem.send(lobs[dst_mem_id, obs_rec_id], dest=dst_pid, tag=m)

                ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
                ##    because they wait to send stuff before able to receive themselves,
                ##    cycle back to receive from src_pid<pid then.
                if m < len(state.mem_list[c.pid_mem]):
                    for src_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):

                        if src_pid == c.pid_mem:
                            ##pid already stores the lobs_seq, just copy
                            lobs_seq = lobs[mem_id, obs_rec_id].copy()
                        else:
                            ##send lobs_seq to dst_pid
                            lobs_seq = c.comm_mem.recv(source=src_pid, tag=m)

                        ##unpack the lobs_seq to form a complete seq
                        for par_id in state.par_list[src_pid]:
                            inds = self.obs_inds[obs_rec_id][par_id]
                            seq[..., inds] = lobs_seq[par_id]

                        obs_seq[mem_id, obs_rec_id] = seq

                ## 3) finish sending lobs_seq to dst_pid, for dst_pid>pid now
                for dst_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                    if m < len(state.mem_list[dst_pid]):
                        dst_mem_id = state.mem_list[dst_pid][m]
                        c.comm_mem.send(lobs[dst_mem_id, obs_rec_id], dest=dst_pid, tag=m)
        c.comm.Barrier()
        c.print_1p(' done.\n')
        return obs_seq

    def pack_local_obs_data(self, c, state, par_id, lobs, lobs_prior):
        """pack lobs and lobs_prior into arrays for the jitted functions"""
        data = {}

        ##number of local obs on partition
        nlobs = np.sum([lobs[r][par_id]['obs'].size for r in self.info['records'].keys()])
        n_obs_rec = len(self.info['records'])        ##number of obs records
        n_state_var = len(state.info['variables'])   ##number of state variable names

        data['obs_rec_id'] = np.zeros(nlobs, dtype=int)
        data['obs'] = np.full(nlobs, np.nan)
        data['x'] = np.full(nlobs, np.nan)
        data['y'] = np.full(nlobs, np.nan)
        data['z'] = np.full(nlobs, np.nan)
        data['t'] = np.full(nlobs, np.nan)
        data['err_std'] = np.full(nlobs, np.nan)
        data['obs_prior'] = np.full((c.nens, nlobs), np.nan)
        data['used'] = np.full(nlobs, False)
        data['hroi'] = np.ones(n_obs_rec)
        data['vroi'] = np.ones(n_obs_rec)
        data['troi'] = np.ones(n_obs_rec)
        data['impact_on_state'] = np.ones((n_obs_rec, n_state_var))

        i = 0
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info['records'][obs_rec_id]

            data['hroi'][obs_rec_id] = obs_rec['hroi']
            data['vroi'][obs_rec_id] = obs_rec['vroi']
            data['troi'][obs_rec_id] = obs_rec['troi']
            for state_var_id in range(len(state.info['variables'])):
                state_vname = state.info['variables'][state_var_id]
                data['impact_on_state'][obs_rec_id, state_var_id] = obs_rec['impact_on_state'][state_vname]

            local_inds = self.obs_inds[obs_rec_id][par_id]
            d = len(local_inds)
            v_list = [0, 1] if obs_rec['is_vector'] else [None]
            for v in v_list:
                data['obs_rec_id'][i:i+d] = obs_rec_id
                data['obs'][i:i+d] = np.squeeze(lobs[obs_rec_id][par_id]['obs'][v, :])
                data['x'][i:i+d] = lobs[obs_rec_id][par_id]['x']
                data['y'][i:i+d] = lobs[obs_rec_id][par_id]['y']
                data['z'][i:i+d] = lobs[obs_rec_id][par_id]['z'].astype(np.float32)
                data['t'][i:i+d] = np.array([t2h(t) for t in lobs[obs_rec_id][par_id]['t']])
                data['err_std'][i:i+d] = lobs[obs_rec_id][par_id]['err_std']
                for m in range(c.nens):
                    data['obs_prior'][m, i:i+d] = np.squeeze(lobs_prior[m, obs_rec_id][par_id][v, :].copy())
                i += d

        return data

    def unpack_local_obs_data(self, c, state, par_id, lobs, lobs_prior, data):
        """unpack data and write back to the original lobs_prior dict"""
        n_obs_rec = len(self.info['records'])
        i = 0
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info['records'][obs_rec_id]

            local_inds = self.obs_inds[obs_rec_id][par_id]
            d = len(local_inds)
            v_list = [0, 1] if obs_rec['is_vector'] else [None]
            for v in v_list:
                for m in range(c.nens):
                    lobs_prior[m, obs_rec_id][par_id][v, :] = data['obs_prior'][m, i:i+d]
                i += d
