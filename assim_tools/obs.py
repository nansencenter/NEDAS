import numpy as np
import os
import struct
import importlib

from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t, dt1h
from utils.progress import print_with_cache, progress_bar
from utils.parallel import by_rank, bcast_by_root, distribute_tasks

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
    obs_rec_id = 0  ##record id for an obs sequence
    pos = 0         ##seek position for rec

    ##loop through obs variables defined in obs_def
    for vrec in c.obs_def:
        vname = vrec['name']

        ##some properties of the variable is defined in its source module
        src = importlib.import_module('dataset.'+vrec['dataset_src'])
        assert vname in src.variables, 'variable '+vname+' not defined in dataset.'+vrec['dataset_src']+'.variables'

        ##loop through time steps in obs window
        for time in c.time + np.array(c.obs_time_steps)*dt1h:
            obs_rec = {'name': vname,
                       'dataset_src': vrec['dataset_src'],
                       'dataset_dir': vrec['dataset_dir'],
                       'model_src': vrec['model_src'],
                       'nobs': vrec.get('nobs', 0),
                       'obs_window_min': vrec['obs_window_min'],
                       'obs_window_max': vrec['obs_window_max'],
                       'dtype': src.variables[vname]['dtype'],
                       'is_vector': src.variables[vname]['is_vector'],
                       'units': src.variables[vname]['units'],
                       'z_units': src.variables[vname]['z_units'],
                       'time': time,
                       'dt': 0,
                       'pos': pos,
                       'err':{'type': vrec['err']['type'],
                              'std': vrec['err']['std'],
                              'hcorr': vrec['err'].get('hcorr',0.),
                              'vcorr': vrec['err'].get('vcorr',0.),
                              'tcorr': vrec['err'].get('tcorr',0.),
                              'cross_corr': vrec['err'].get('cross_corr',{}),
                              },
                       'hroi': vrec['hroi'],
                       'vroi': vrec['vroi'],
                       'troi': vrec['troi'],
                       'impact_on_state': vrec['impact_on_state'],
                       }
            obs_info['records'][obs_rec_id] = obs_rec

            ##update obs_rec_id
            obs_rec_id += 1

            ##we don't know the size of obs_seq yet
            ##will wait for process_all_obs to update the seek position

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
    - z: np.array[nz, ny, nx]
      z coordinate fields for all unique level k defined in state_info
    """
    ##first, get a list of indices k
    z_file = os.path.join(c.work_dir, 'cycle', t2s(time), 'analysis', c.s_dir, 'z_coords.bin')
    k_list = list(set([r['k'] for i,r in c.state_info['fields'].items() if r['time']==time]))

    ##get z coords for each level
    z = np.zeros((len(k_list), c.ny, c.nx))
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
    - obs_seq: from process_all_obs()

    Returns:
    - obs_inds: dict[obs_rec_id, dict[par_id, inds]]
      where inds is np.array with indices in the full obs_seq, for the subset of obs
      that belongs to partition par_id
    """

    ##each pid_rec has a subset of obs_rec_list
    obs_inds_pid = {}
    for obs_rec_id in c.obs_rec_list[c.pid_rec]:
        obs_rec = obs_seq[obs_rec_id]
        obs_inds_pid[obs_rec_id] = {}

        if c.assim_mode == 'batch':
            ##1. screen horizontally for obs inside hroi of partition par_id
            hroi = c.obs_info['records'][obs_rec_id]['hroi']
            xo = np.array(obs_rec['x'])  ##obs x,y
            yo = np.array(obs_rec['y'])
            x = c.grid.x[0,:]   ##grid x,y
            y = c.grid.y[:,0]

            ##loop over partitions with par_id
            for par_id in range(len(c.partitions)):
                ist,ied,di,jst,jed,dj = c.partitions[par_id]

                ##condition 1: within the four corner points of the tile
                hdist = np.hypot(np.minimum(np.abs(xo - x[ist]),
                                            np.abs(xo - x[ied-1])),
                                 np.minimum(np.abs(yo - y[jst]),
                                            np.abs(yo - y[jed-1])) )
                cond1 = (hdist < hroi)

                ##condition 2: within [x_ist:x_ied, y_jst-hroi:y_jed+hroi]
                cond2 = np.logical_and(np.logical_and(xo >= x[ist],
                                                      xo <= x[ied-1]),
                                       np.logical_and(yo > y[jst]-hroi,
                                                      yo < y[jed-1]+hroi) )

                ##condition 3: within [x_ist-hroi:x_ied+hroi, y_jst:y_jed]
                cond3 = np.logical_and(np.logical_and(xo > x[ist]-hroi,
                                                      xo < x[ied-1]+hroi),
                                       np.logical_and(yo >= y[jst],
                                                      yo <= y[jed-1]) )

                ##if any of the 3 condition satisfies, the obs is within
                ##hroi of any points in tile[par_id]
                inds = np.where(np.logical_or(cond1, np.logical_or(cond2, cond3)))[0]

                obs_inds_pid[obs_rec_id][par_id] = inds

        elif c.assim_mode == 'serial':
            ##locality doesn't matter, we just divide obs_rec into nproc_mem
            ##partitions with par_id from 0 to nproc_mem-1
            full_inds = np.arange(obs_rec['obs'].shape[-1])

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


def distribute_partitions(c):
    """
    Distribute par_id across processors according to the work load on each partition
    """
    par_list_full = np.arange(len(c.partitions))

    if c.assim_mode == 'batch':
        ##distribute the list of par_id according to workload to each pid
        ##number of unmasked grid points in each tile
        nlpts_loc = np.array([np.sum((~c.mask[jst:jed:dj, ist:ied:di]).astype(int))
                              for ist,ied,di,jst,jed,dj in c.partitions] )

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


def output_obs(obs_seq):
    pass


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

    2, If obs_name is one of the variables provided by obs_src.obs_operator, we call it to
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
    obs_src = importlib.import_module('dataset.'+kwargs['dataset_src'])
    ##model source module
    model = c.model_config[kwargs['model_src']]

    ##option 1  TODO: update README
    ## if dataset module provides an obs_operator, use it to compute obs
    if hasattr(obs_src, 'obs_operator') and kwargs['model_src'] in obs_src.obs_operator and kwargs['name'] in obs_src.obs_operator[kwargs['model_src']]:
        if synthetic:
            path = os.path.join(c.model_def[kwargs['model_src']]['truth_dir'])
        else:
            path = os.path.join(c.work_dir,'cycle',t2s(time),kwargs['model_src'])

        operator = obs_src.obs_operator[kwargs['model']]
        # assert kwargs['name'] in operator, 'obs variable '+kwargs['name']+' not provided by dataset '+kwargs['dataset_src']+'.obs_operator for '+kwargs['model_src']

        ##get the obs seq from operator
        seq = operator[kwargs['name']](path, c.grid, c.mask, **kwargs)

    ##option 2:
    ## if obs variable is one of the state variable, or can be computed by the model,
    ## then we just need to collect the 3D variable and interpolate in x,y,z
    elif obs_name in model.variables:

        levels = model.variables[obs_name]['levels']
        for k in range(len(levels)):
            if obs_name in [r['name'] for r in c.state_def] and not synthetic:
                ##the obs is one of the state variables
                ##find its corresponding rec_id
                rec_id = [i for i,r in c.state_info['fields'].items() if r['name']==obs_name and r['k']==levels[k]][0]

                ##option 1.1: if the current pid stores this field, just read it
                if rec_id in c.rec_list[c.pid_rec] and mem_id in c.mem_list[c.pid_mem] and 'model_fld' in kwargs and 'model_z' in kwargs:
                    z = kwargs['model_z'][mem_id, rec_id]
                    fld = kwargs['model_fld'][mem_id, rec_id]

                else:  ##option 1.2: read field from state binfile
                    path = os.path.join(c.work_dir,'cycle',t2s(time),'analysis',c.s_dir)
                    z = read_field(os.path.join(path,'/z_coords.bin'), c.state_info, c.mask, 0, rec_id)
                    fld = read_field(os.path.join(path,'/prior_state.bin'), c.state_info, c.mask, mem_id, rec_id)

            else:  ##option 1.3: get the field from model.read_var
                if synthetic:
                    path = os.path.join(c.model_def[kwargs['model_src']]['truth_dir'])
                else:
                    path = os.path.join(c.work_dir,'cycle',t2s(time),kwargs['model_src'])

                if k == 0:  ##initialize grid obj for conversion
                    model.read_grid(path=path, **kwargs)
                    model.grid.set_destination_grid(c.grid)

                z_ = model.grid.convert(model.z_coords(path=path, k=levels[k], **kwargs))
                z = np.array([z_, z_]) if is_vector else z_

                fld = model.grid.convert(model.read_var(path=path, k=levels[k], **kwargs), is_vector=is_vector)

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
    if c.debug:
        by_rank(c.comm,0)(print_with_cache)('read obs sequence from datasets\n')

    ##get obs_seq from dataset module, each pid_rec gets its own workload as a subset of obs_rec_list
    obs_seq = {}
    for obs_rec_id in c.obs_rec_list[c.pid_rec]:
        obs_rec = c.obs_info['records'][obs_rec_id]

        ##load the dataset module
        src = importlib.import_module('dataset.'+obs_rec['dataset_src'])
        assert obs_rec['name'] in src.variables, 'variable '+obs_rec['name']+' not defined in dataset.'+obs_rec['dataset_src']+'.variables'

        ##directory storing the dataset files for this variable
        path = obs_rec['dataset_dir']

        ##read ens-mean z coords from z_file for this obs network
        z = read_mean_z_coords(c, obs_rec['time'])

        if c.use_synthetic_obs:
            ##generate synthetic obs network
            truth_path = os.path.join(c.model_def[obs_rec['model_src']]['truth_dir'])

            seq = src.random_network(path, c.grid, c.mask, z, truth_path, **obs_rec)

            ##compute obs values
            seq['obs'] = state_to_obs(c, member=None, **obs_rec, **seq)

            ##perturb with obs err
            seq['obs'] += np.random.normal(0, 1, seq['obs'].shape) * obs_rec['err']['std']

        else:
            ##read dataset files and obtain obs sequence
            seq = src.read_obs(path, c.grid, c.mask, z, **obs_rec)
        del z

        if c.debug:
            by_rank(c.comm_rec, c.pid_rec)(print_with_cache)('number of '+obs_rec['name']+' obs from '+obs_rec['dataset_src']+': {}\n'.format(seq['obs'].shape[-1]))

        obs_seq[obs_rec_id] = seq
        obs_rec['nobs'] = seq['obs'].shape[-1]  ##update nobs

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

    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('compute obs priors\n')
    obs_prior_seq = {}

    ##process the obs, each proc gets its own workload as a subset of
    ##all proc goes through their own task list simultaneously
    nr = len(c.obs_rec_list[c.pid_rec])
    nm = len(c.mem_list[c.pid_mem])
    for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
        for r, obs_rec_id in enumerate(c.obs_rec_list[c.pid_rec]):
            if c.debug:
                print(progress_bar(m*nr+r, nr*nm))

            ##this is the obs record to process
            obs_rec = c.obs_info['records'][obs_rec_id]

            seq = state_to_obs(c, member=mem_id,
                               model_fld=fields, model_z=z_fields,
                               **obs_rec, **obs_seq[obs_rec_id])

            obs_prior_seq[mem_id, obs_rec_id] = seq
    if c.debug:
        print(' done.\n')

    return obs_prior_seq


