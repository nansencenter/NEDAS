import numpy as np
import os
import struct
from grid import Grid
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t, dt1h, ensure_list
from utils.progress import print_with_cache, progress_bar
from utils.parallel import by_rank, bcast_by_root, distribute_tasks
from utils.multiscale import get_scale_component, get_error_scale_factor
from utils.dir_def import forecast_dir
from .state import read_field

"""
Note: The observation has dimensions: variable, time, z, y, x
Since the observation network is typically irregular, we store the obs record
for each variable in a 1d sequence, with coordinates (t,z,y,x), and size nobs

To parallelize workload, we distribute each obs record over all the processors
- for batch assimilation mode, each pid stores the list of local obs within the
  hroi of its tiles, with size nlobs (number of local obs)
- for serial mode, each pid stores a non-overlapping subset of the obs list,
  here 'local' obs (in storage sense) is broadcast to all pid before computing
  its update to the state/obs near that obs.

The hroi is separately defined for each obs record.
For very large hroi, the serial mode is more parallel efficient option, since
in batch mode the same obs may need to be stored in multiple pids

To compare to the observation, obs_prior simulated by the model needs to be
computed, they have dimension [nens, nlobs], indexed by (mem_id, obs_id)
"""
def parse_obs_info(c):
    """
    Parse info for the observation records defined in config.

    Input:
    - c: config module with the environment variables

    Return:
    - info: dict
      A dictionary with some dimensions and list of unique obs records
    """
    obs_info = {'size':0, 'records':{}}
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
        dataset = c.dataset_config[vrec['dataset_src']]
        variables = dataset.variables
        assert vname in variables, 'variable '+vname+' not defined in '+vrec['dataset_src']+'.dataset.variables'

        ##parse impact of obs on each state variable, default is 1.0 on all variables unless set by obs_def record
        impact_on_state = {}
        for state_name in c.state_info['variables']:
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
                       'obs_window_min': vrec.get('obs_window_min', 0),
                       'obs_window_max': vrec.get('obs_window_max', 0),
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
                       'hroi': vrec['hroi'],
                       'vroi': vrec['vroi'],
                       'troi': vrec['troi'],
                       'impact_on_state': impact_on_state,
                       }
            obs_variables.add(vname)
            obs_err_types.add(obs_err_type)
            obs_info['records'][obs_rec_id] = obs_rec

            ##update obs_rec_id
            obs_rec_id += 1

            ##we don't know the size of obs_seq yet
            ##will wait for prepare_obs to update the seek position
        
    obs_info['variables'] = list(obs_variables)
    obs_info['err_types'] = list(obs_err_types)

    ##go through the obs_rec again to fill in the default err.cross_corr
    for obs_rec_id, obs_rec in obs_info['records'].items():
        assert isinstance(obs_rec['err']['cross_corr'], dict), f"obs_def: {obs_rec['name']} has err.cross_corr defined as {obs_rec['err']['cross_corr']}, expecting a dictionary"
        for vname in obs_info['variables']:
            if vname not in obs_rec['err']['cross_corr']:
                if vname == obs_rec['name']:
                    obs_rec['err']['cross_corr'][vname] = 1.0
                else:
                    obs_rec['err']['cross_corr'][vname] = 0.0
            else:
                assert isinstance(obs_rec['err']['cross_corr'][vname], float), f"obs_def: {obs_rec['name']} has err.cross_corr.{vname} defined as {obs_rec['err']['cross_corr'][vname]}, expecting a float"

    return obs_info

def distribute_obs_tasks(c):
    """
    Distribute obs_rec_id across processors

    Inputs: c: config module

    Returns: obs_rec_list: dict[pid_rec, list[obs_rec_id]]
    """
    obs_rec_list_full = [i for i in c.obs_info['records'].keys()]
    obs_rec_size = np.array([2 if r['is_vector'] else 1 for i,r in c.obs_info['records'].items()])
    obs_rec_list = distribute_tasks(c.comm_rec, obs_rec_list_full, obs_rec_size)

    return obs_rec_list

def read_mean_z_coords(c, time):
    """
    Read the ensemble-mean z coords from z_file at obs time

    Inputs:
    - c: config module
    - time: datetime obj

    Return:
    - z: np.array[nz, grid.x.shape]
      z coordinate fields for all unique level k defined in state_info
    """
    ##first, get a list of indices k
    z_file = os.path.join(c.analysis_dir, 'z_coords.bin')
    k_list = list(set([r['k'] for i,r in c.state_info['fields'].items() if r['time']==time]))

    ##get z coords for each level
    z = np.zeros((len(k_list),)+c.state_info['shape'])
    for k in range(len(k_list)):

        ##the rec_id in z_file corresponding to this level
        ##there can be multiple records (different state variables)
        ##we take the first one
        rec_id = [i for i,r in c.state_info['fields'].items() if r['time']==time and r['k']==k_list[k]][0]

        ##read the z field with (mem_id=0, rec_id) from z_file
        z_fld = read_field(z_file, c.state_info, c.mask, 0, rec_id)

        ##assign the coordinates to z(k)
        if c.state_info['fields'][rec_id]['is_vector']:
            z[k, ...] = z_fld[0, ...]
        else:
            z[k, ...] = z_fld

    return z

def assign_obs(c, obs_seq):
    """
    Assign the observation sequence to each partition par_id

    Inputs:
    - c: config module
    - obs_seq: from prepare_obs()

    Returns:
    - obs_inds: dict[obs_rec_id, dict[par_id, inds]]
      where inds is np.array with indices in the full obs_seq, for the subset of obs
      that belongs to partition par_id
    """

    ##each pid_rec has a subset of obs_rec_list
    obs_inds_pid = {}
    for obs_rec_id in c.obs_rec_list[c.pid_rec]:
        full_inds = np.arange(obs_seq[obs_rec_id]['obs'].shape[-1])
        obs_inds_pid[obs_rec_id] = {}

        if c.assim_mode == 'batch':
            ##screen horizontally for obs inside hroi of each partition
            obs_inds_pid[obs_rec_id] = assign_obs_to_tiles(c, obs_rec_id, obs_seq)

        elif c.assim_mode == 'serial':
            ##locality doesn't matter, we just divide obs_rec into nproc_mem parts
            inds = distribute_tasks(c.comm_mem, full_inds)
            for par_id in range(c.nproc_mem):
                obs_inds_pid[obs_rec_id][par_id] = inds[par_id]

    ##now each pid_rec has figured out obs_inds for its own list of obs_rec_ids, we
    ##gather all obs_rec_id from different pid_rec to form the complete obs_inds dict
    obs_inds = {}
    for entry in c.comm_rec.allgather(obs_inds_pid):
        for obs_rec_id, data in entry.items():
            obs_inds[obs_rec_id] = data

    return obs_inds

def assign_obs_to_tiles(c, obs_rec_id, obs_seq):
    hroi = c.obs_info['records'][obs_rec_id]['hroi']

    xo = np.array(obs_seq[obs_rec_id]['x'])  ##obs x,y
    yo = np.array(obs_seq[obs_rec_id]['y'])

    ##loop over partitions with par_id
    obs_inds = {}
    for par_id in range(len(c.partitions)):
        ##find bounding box for this partition
        if len(c.grid.x.shape)==2:
            ist,ied,di,jst,jed,dj = c.partitions[par_id]
            xmin, xmax, ymin, ymax = c.grid.x[0,ist], c.grid.x[0,ied-1], c.grid.y[jst,0], c.grid.y[jed-1,0]
        else:
            inds = c.partitions[par_id]
            x = c.grid.x[inds]
            y = c.grid.y[inds]
            xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        Dx = 0.5 * (xmax - xmin)
        Dy = 0.5 * (ymax - ymin)
        xc = xmin + Dx
        yc = ymin + Dy

        ##observations within the bounding box + halo region of width hroi will be assigned to
        ##this partition. Although this will include some observations near the corner that are
        ##not within hroi of any grid points, this is favorable for the efficiency in finding subset
        obs_inds[par_id] = np.where(np.logical_and(c.grid.distance_in_x(xc, xo) <= Dx+hroi,
                                                   c.grid.distance_in_y(yc, yo) <= Dy+hroi))[0]
    return obs_inds

def distribute_partitions(c):
    """
    Distribute par_id across processors according to the work load on each partition
    """
    par_list_full = np.arange(len(c.partitions))

    if c.assim_mode == 'batch':
        ##distribute the list of par_id according to workload to each pid
        ##number of unmasked grid points in each tile
        if len(c.grid.x.shape)==2:
            nlpts_loc = np.array([np.sum((~c.mask[jst:jed:dj, ist:ied:di]).astype(int))
                                for ist,ied,di,jst,jed,dj in c.partitions] )
        else:
            nlpts_loc = np.array([np.sum((~c.mask[inds]).astype(int))
                                for inds in c.partitions] )

        ##number of observations within the hroi of each tile, at loc,
        ##sum over the len of obs_inds for obs_rec_id over all obs_rec_ids
        nlobs_loc = np.array([np.sum([len(c.obs_inds[r][p])
                                      for r in c.obs_info['records'].keys()])
                              for p in par_list_full] )

        workload = np.maximum(nlpts_loc, 1) * np.maximum(nlobs_loc, 1)
        par_list = distribute_tasks(c.comm_mem, par_list_full, workload)

    if c.assim_mode == 'serial':
        ##just assign each partition to each pid, pid==par_id
        par_list = {p:np.array([p]) for p in range(c.nproc_mem)}

    return par_list

def global_obs_list(c):
    ##form the global list of obs (in serial mode the main loop is over this list)
    n_obs_rec = len(c.obs_info['records'])

    i = {}  ##location in full obs vector on owner pid
    for owner_pid in range(c.nproc_mem):
        i[owner_pid] = 0

    obs_list = []
    for obs_rec_id in range(n_obs_rec):
        obs_rec = c.obs_info['records'][obs_rec_id]
        v_list = [0, 1] if obs_rec['is_vector'] else [None]
        for owner_pid in c.obs_inds[obs_rec_id].keys():
            for _ in c.obs_inds[obs_rec_id][owner_pid]:
                for v in v_list:
                    obs_list.append((obs_rec_id, v, owner_pid, i[owner_pid]))
                    i[owner_pid] += 1

    np.random.shuffle(obs_list)  ##randomize the order of obs (this is optional)
    return obs_list

##TODO: some of these funcs are not ready
##write obs_info to a .dat file accompanying the obs_seq bin file
def write_obs_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        f.write('{} {}\n'.format(info['nobs'], info['nens']))
        for rec in info['obs_seq'].values():
            f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(rec['name'], rec['dataset_src'], rec['model_src'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['z_units'], rec['x'], rec['y'], rec['z'], t2h(rec['time']), rec['pos']))

##read obs_info from the dat file
def read_obs_info(binfile):
    with open(binfile.replace('.bin','.dat'), 'r') as f:
        lines = f.readlines()

        ss = lines[0].split()
        info = {'nobs':int(ss[0]), 'nens':int(ss[1]), 'obs_seq':{}}

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
            info['obs_seq'][obs_id] = rec
            obs_id += 1
        return info

##output an obs values to the binfile for a member (obs_prior), if member=None it is the actual obs
def write_obs_seq(binfile, info, obs_seq, member=None):
    nens = info['nens']

    with open(binfile, 'r+b') as f:
        for rec in obs_seq:
            nv = 2 if rec['is_vector'] else 1
            if member is None:
                m = 0
            else:
                assert member<nens, f'member = {member} exceeds the ensemble size {nens}'
                m = member+1

            f.seek(rec['pos'] + nv*type_size[rec['dtype']]*m)
            f.write(struct.pack((nv*type_dic[rec['dtype']]), *np.atleast_1d(rec['value'])))

def read_obs_seq(binfile, info, obs_seq, member=None):
    nens = info['nens']
    obs_seq_out = []
    with open(binfile, 'rb') as f:
        for rec in obs_seq:
            nv = 2 if rec['is_vector'] else 1
            if member is None:
                m = 0
            else:
                assert member<nens, f'member = {member} exceeds the ensemble size {nens}'
                m = member+1
            f.seek(rec['pos'] + nv*type_size[rec['dtype']]*m)
            rec['value'] = np.array(struct.unpack((nv*type_dic[rec['dtype']]), f.read(nv*type_size[rec['dtype']])))
            obs_seq_out.append(rec)
    return obs_seq_out

def state_to_obs(c, **kwargs):
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

    Inputs:
    - c: config module

    Some kwargs:
    - member: int, member index; or None if dealing with synthetic obs
    - model_fld: fields from prepare_state()
    - model_z: z_coords from prepare_state()
    - name: str, obs variable name
    - time: datetime obj, time of the obs window
    - is_vector: bool, if True the obs is a vector measurement
    - dataset_src: str, dataset source module name providing the obs
    - model_src: str, model source module name providing the state
    - x, y, z, t: np.array, coordinates from obs_seq

    Return:
    - seq: np.array
      values corresponding to the obs_seq but from the state identified by kwargs
    """

    mem_id = kwargs['member']
    synthetic = True if mem_id is None else False

    time = kwargs['time']
    obs_name = kwargs['name']
    is_vector = kwargs['is_vector']

    obs_x = np.array(kwargs['x'])
    obs_y = np.array(kwargs['y'])
    obs_z = np.array(kwargs['z'])
    nobs = len(obs_x)

    if is_vector:
        seq = np.full((2, nobs), np.nan)
    else:
        seq = np.full(nobs, np.nan)

    ##obs dataset source module
    dataset = c.dataset_config[kwargs['dataset_src']]
    ##model source module
    model = c.model_config[kwargs['model_src']]

    ##option 1  TODO: update README and comments
    ## if dataset module provides an obs_operator, use it to compute obs
    if hasattr(dataset, 'obs_operator') and kwargs['name'] in dataset.obs_operator:
        if synthetic:
            path = model.truth_dir
        else:
            path = forecast_dir(c, time, kwargs['model_src'])

        operator = dataset.obs_operator[kwargs['name']]

        ##get the obs seq from operator
        seq = operator(path=path, model=model, grid=c.grid, mask=c.mask, **kwargs)

    ##option 2:
    ## if obs variable is one of the state variable, or can be computed by the model,
    ## then we just need to collect the 3D variable and interpolate in x,y,z
    elif obs_name in model.variables:

        levels = model.variables[obs_name]['levels']
        for k in range(len(levels)):
            if obs_name in [r['name'] for r in ensure_list(c.state_def)] and not synthetic:
                ##the obs is one of the state variables
                ##find its corresponding rec_id
                rec_id = [i for i,r in c.state_info['fields'].items() if r['name']==obs_name and r['k']==levels[k]][0]

                ##option 1.1: if the current pid stores this field, just read it
                if rec_id in c.rec_list[c.pid_rec] and mem_id in c.mem_list[c.pid_mem] and 'model_fld' in kwargs and 'model_z' in kwargs:
                    z = kwargs['model_z'][mem_id, rec_id]
                    fld = kwargs['model_fld'][mem_id, rec_id]

                else:  ##option 1.2: read field from state binfile
                    z = read_field(os.path.join(c.analysis_dir,'z_coords.bin'), c.state_info, c.mask, 0, rec_id)
                    fld = read_field(os.path.join(c.analysis_dir,'prior_state.bin'), c.state_info, c.mask, mem_id, rec_id)

            else:  ##option 1.3: get the field from model.read_var
                if synthetic:
                    path = model.truth_dir
                else:
                    path = forecast_dir(c, time, kwargs['model_src'])

                if k == 0:  ##initialize grid obj for conversion
                    model.read_grid(path=path, **kwargs)
                    model.grid.set_destination_grid(c.grid)

                model_z = model.z_coords(path=path, member=kwargs['member'], time=kwargs['time'], k=levels[k])
                z_ = model.grid.convert(model_z)
                z = np.array([z_, z_]) if is_vector else z_

                model_fld = model.read_var(path=path, name=kwargs['name'], member=kwargs['member'], time=kwargs['time'], k=levels[k])
                fld = model.grid.convert(model_fld, is_vector=is_vector)

            ##horizontal interp field to obs_x,y, for current layer k
            if is_vector:
                z = c.grid.interp(z[0, ...], obs_x, obs_y)
                zc = np.array([z, z])
                v1 = c.grid.interp(fld[0, ...], obs_x, obs_y)
                v2 = c.grid.interp(fld[1, ...], obs_x, obs_y)
                vc = np.array([v1, v2])
            else:
                zc = c.grid.interp(z, obs_x, obs_y)
                vc = c.grid.interp(fld, obs_x, obs_y)

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
                inds = np.where(np.logical_and(obs_z >= np.minimum(0, 0.5*dzc),
                                               obs_z <= np.maximum(0, 0.5*dzc)) )
                seq[..., inds] = vc[..., inds]

            if k > 0:
                dzc = zc - zp  ##layer thickness for level k

                ##in between levels: linear interp between vp and vc
                z_vp = zp - 0.5*dzp
                z_vc = zp + 0.5*dzc
                inds = np.where(np.logical_and(obs_z >= np.minimum(z_vp, z_vc),
                                               obs_z <= np.maximum(z_vp, z_vc)) )
                ##there can be collapsed layers if z_vc=z_vp
                ##TODO: still got some warnings here
                with np.errstate(divide='ignore'):
                    vi = np.where(z_vp==z_vc,
                                  vp,   ##for collapsed layers just use previous value
                                  ((z_vc - obs_z)*vc + (obs_z - z_vp)*vp)/(z_vc - z_vp) )  ##otherwise linear interp between layer
                seq[..., inds] = vi[..., inds]

            if k == len(levels)-1:
                ##the last level: constant vc from z=zc-dzc/2 to zc
                inds = np.where(np.logical_and(obs_z >= np.minimum(zc-0.5*dzc, zc),
                                               obs_z <= np.maximum(zc-0.5*dzc, zc)) )
                seq[..., inds] = vc[..., inds]

            if k < len(levels)-1:
                ##make a copy of current layer as 'previous' for next k
                zp = zc.copy()
                vp = vc.copy()
                dzp = dzc.copy()

    else:
        raise ValueError('unable to obtain obs prior for '+obs_name)

    return seq

def prepare_obs(c):
    """
    Process the obs in parallel, read dataset files and convert to obs_seq
    which contains obs value, coordinates and other info

    Since this is the actual obs (1 copy), only 1 processor needs to do the work

    Inputs:
    - c: config module

    Return:
    - obs_seq: dict[obs_rec_id, record]
      where each record is dict[key, np.array], the mandatory keys are
        'obs' the observed values (measurements)
        'x', 'y', 'z', 't' the coordinates for each measurement
        'err_std' the uncertainties for each measurement
        there can be other optional keys provided by read_obs() but we don't use them
    - c.obs_info with updated nobs
    """
    by_rank(c.comm,0)(print_with_cache)('>>> read observation sequence from datasets\n')

    ##get obs_seq from dataset module, each pid_rec gets its own workload as a subset of obs_rec_list
    obs_seq = {}
    for obs_rec_id in c.obs_rec_list[c.pid_rec]:
        obs_rec = c.obs_info['records'][obs_rec_id]

        ##load the dataset module
        dataset = c.dataset_config[obs_rec['dataset_src']]
        assert obs_rec['name'] in dataset.variables, 'variable '+obs_rec['name']+' not defined in dataset.'+obs_rec['dataset_src']+'.variables'

        model = c.model_config[obs_rec['model_src']]
        ##read ens-mean z coords from z_file for this obs network
        ##typically model.z_coords can compute the z coords as well, but it is more efficient
        ##to just store the ensemble mean z here
        model.z = read_mean_z_coords(c, obs_rec['time'])

        if c.use_synthetic_obs:
            ##generate synthetic obs network
            seq = dataset.random_network(model=model, grid=c.grid, mask=c.mask, **obs_rec)

            ##compute obs values
            seq['obs'] = state_to_obs(c, member=None, **obs_rec, **seq)

            ##perturb with obs err
            seq['obs'] += np.random.normal(0, 1, seq['obs'].shape) * obs_rec['err']['std']

        else:
            ##read dataset files and obtain obs sequence
            seq = dataset.read_obs(model=model, grid=c.grid, mask=c.mask, **obs_rec)

        by_rank(c.comm_rec, c.pid_rec)(print_with_cache)('number of '+obs_rec['name']+' obs from '+obs_rec['dataset_src']+': {}\n'.format(seq['obs'].shape[-1]))

        ##misc. transform here
        ##e.g., multiscale approach:
        if c.nscale > 1 and c.decompose_obs:
            obs_grid = Grid(c.grid.proj, seq['x'], seq['y'], regular=False)
            seq['obs'] = get_scale_component(obs_grid, seq['obs'], c.character_length, c.scale_id)
            seq['err_std'] *= get_error_scale_factor(obs_grid, c.character_length, c.scale_id)

        obs_seq[obs_rec_id] = seq
        obs_rec['nobs'] = seq['obs'].shape[-1]  ##update nobs

    ##additional output for debugging
    if c.debug:
        # if c.pid == 0:
        #     np.save(os.path.join(c.analysis_dir, 'rec_list.npy'), c.rec_list)
        #     np.save(os.path.join(c.analysis_dir, 'mem_list.npy'), c.mem_list)
        #     np.save(os.path.join(c.analysis_dir, 'obs_rec_list.npy'), c.obs_rec_list)
        #     np.save(os.path.join(c.analysis_dir, 'obs_inds.npy'), c.obs_inds)
        #     np.save(os.path.join(c.analysis_dir, 'partitions.npy'), c.partitions)
        #     np.save(os.path.join(c.analysis_dir, 'par_list.npy'), c.par_list)
        if c.pid_mem == 0:
            #np.save(os.path.join(c.analysis_dir, f'obs_seq.{c.pid_rec}.npy'), obs_seq)
            for obs_rec_id, rec in obs_seq.items():
                file = os.path.join(c.analysis_dir, f'obs_seq.rec{obs_rec_id}.npy')
                np.save(file, rec)

    return c.obs_info, obs_seq

def prepare_obs_from_state(c, obs_seq, fields, z_fields):
    """
    Compute the obs priors in parallel, run state_to_obs to obtain obs_prior_seq

    Inputs:
    - c: config module
    - obs_seq: from prepare_obs()
    - fields, z_fiels: from prepare_state()

    Return:
    - obs_prior_seq: dict[(mem_id, obs_rec_id), seq]
      where seq is np.array with values corresponding to obs_seq['obs']
    """

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.obs_rec_list.items() if len(lst)>0][0]
    c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show

    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    print_1p('>>> compute observation priors\n')
    obs_prior_seq = {}

    ##process the obs, each proc gets its own workload as a subset of
    ##all proc goes through their own task list simultaneously
    nr = len(c.obs_rec_list[c.pid_rec])
    nm = len(c.mem_list[c.pid_mem])
    for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
        for r, obs_rec_id in enumerate(c.obs_rec_list[c.pid_rec]):
            ##this is the obs record to process
            obs_rec = c.obs_info['records'][obs_rec_id]

            if c.debug:
                print(f"PID {c.pid:4}: obs_prior mem{mem_id+1:03} {obs_rec['name']:20}", flush=True)
            else:
                print_1p(progress_bar(m*nr+r, nr*nm))

            seq = state_to_obs(c, member=mem_id,
                               model_fld=fields, model_z=z_fields,
                               **obs_rec, **obs_seq[obs_rec_id])

            ##misc. transform here
            ##e.g., multiscale approach:
            if c.nscale > 1 and c.decompose_obs:
                obs_grid = Grid(c.grid.proj, obs_seq[obs_rec_id]['x'], obs_seq[obs_rec_id]['y'], regular=False)
                seq = get_scale_component(obs_grid, seq, c.character_length, c.scale_id)

            obs_prior_seq[mem_id, obs_rec_id] = seq
    c.comm.Barrier()
    print_1p(' done.\n')

    ##additional output for debugging
    if c.debug:
        #np.save(os.path.join(c.analysis_dir, f'obs_prior_seq.{c.pid_mem}.{c.pid_rec}.npy'), obs_prior_seq)
        for key, seq in obs_prior_seq.items():
            mem_id, obs_rec_id = key
            file = os.path.join(c.analysis_dir, f'obs_prior_seq.rec{obs_rec_id}.mem{mem_id:03}.npy')
            np.save(file, seq)

    return obs_prior_seq

