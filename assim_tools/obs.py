import numpy as np
import struct
import importlib
from datetime import datetime, timedelta
from memory_profiler import profile
import config as c
from conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from log import message, progress_bar
from parallel import bcast_by_root, distribute_tasks
from perturb import random_field_gaussian
from grid import Grid
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


@bcast_by_root(c.comm)
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
        src = importlib.import_module('dataset.'+vrec['source'])
        assert vname in src.variables, 'variable '+vname+' not defined in dataset.'+vrec['source']+'.variables'

        ##loop through time steps in obs window
        for time in s2t(c.time) + c.obs_ts*timedelta(hours=1):
            obs_rec = {'name': vname,
                       'source': vrec['source'],
                       'model': vrec['model'],
                       'dtype': src.variables[vname]['dtype'],
                       'is_vector': src.variables[vname]['is_vector'],
                       'units': src.variables[vname]['units'],
                       'z_units': src.variables[vname]['z_units'],
                       'time': time,
                       'obs_window_min': c.obs_window_min,
                       'obs_window_max': c.obs_window_max,
                       'pos': pos,
                       'err':{'type': vrec['err_type'],
                              'std': vrec['err_std'],
                              'hcorr': vrec['err_hcorr'],
                              'vcorr': vrec['err_vcorr'],
                              'tcorr': vrec['err_tcorr'],
                              'cross_corr': vrec['err_cross_corr'],
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
            ##will wait for prepare_obs to update the seek position

    return obs_info


##write obs_info to a .dat file accompanying the obs_seq bin file
def write_obs_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        f.write('{} {}\n'.format(info['nobs'], info['nens']))
        for rec in info['obs_seq'].values():
            f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(rec['name'], rec['source'], rec['model'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['z_units'], rec['x'], rec['y'], rec['z'], t2h(rec['time']), rec['pos']))


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
                   'source': ss[1],
                   'model': ss[2],
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


def build_obs_tasks(c, obs_info):
    ##obs_rec_id list
    obs_rec_list_full = [i for i in obs_info['records'].keys()]
    obs_rec_size = np.array([2 if r['is_vector'] else 1 for i,r in obs_info['records'].items()])
    obs_rec_list = distribute_tasks(c.comm_rec, obs_rec_list_full, obs_rec_size)

    return obs_rec_list


def read_mean_z_coords(c, state_info, time):
    """
    Read the ensemble-mean z coords from z_file at obs time

    Inputs: c, state_info, time

    Return: z fields [k, ...] for all unique level k defined in state_info
    """
    ##first, get a list of indices k
    z_file = c.work_dir+'/analysis/'+t2s(time)+c.s_dir+'/z_coords.bin'
    k_list = list(set([r['k'] for i,r in state_info['fields'].items() if r['time']==time]))

    ##get z coords for each level
    z = np.zeros((len(k_list), c.ny, c.nx))
    for k in range(len(k_list)):

        ##the rec_id in z_file corresponding to this level
        ##there can be multiple records (different state variables)
        ##we take the first one
        rec_id = [i for i,r in state_info['fields'].items() if r['time']==time and r['k']==k_list[k]][0]

        ##read the z field with (mem_id=0, rec_id) from z_file
        z_fld = read_field(z_file, state_info, c.mask, 0, rec_id)

        ##assign the coordinates to z(k)
        if state_info['fields'][rec_id]['is_vector']:
            z[k, ...] = z_fld[0, ...]
        else:
            z[k, ...] = z_fld

    return z


@bcast_by_root(c.comm_mem)
def prepare_obs(c, state_info, obs_info, obs_rec_list):
    """
    Prepare the obs in parallel, read dataset files and convert to obs_seq
    which contains obs value, coordinates and other info

    Since this is the actual obs (1 copy), only pid_mem==0 will do the work
    and broadcast to all pid_mem in comm_mem

    Inputs: c, state_info, obs_info, obs_rec_list

    Return: obs_seq
    """
    message(c.comm_rec, 'reading obs sequences from dataset\n', 0)
    obs_seq = {}

    ##get obs_seq from dataset module, each pid_rec gets its own workload
    ##as a subset of obs_rec_list
    for obs_rec_id in obs_rec_list[c.pid_rec]:
        obs_rec = obs_info['records'][obs_rec_id]

        ##load the dataset module
        src = importlib.import_module('dataset.'+obs_rec['source'])
        assert obs_rec['name'] in src.variables, 'variable '+obs_rec['name']+' not defined in dataset.'+obs_rec['source']+'.variables'

        ##directory storing the dataset files for this variable
        path = c.data_dir+'/'+obs_rec['source']

        ##read ens-mean z coords from z_file for this obs network
        z = read_mean_z_coords(c, state_info, obs_rec['time'])

        if c.use_synthetic_obs:
            ##generate synthetic obs network
            seq = src.random_network(c.grid, c.mask, z)

            ##get obs values from the truth file
            truth_file = c.data_dir+'/truth/'+t2s(obs_rec['time'])+'/'+rec['model']
            # obs_operator()

            ##perturb with obs err
            # obs_rec['obs'] += random_field(obs_rec['err'])

        else:
            ##read dataset files and obtain obs sequence
            seq = src.read_obs(path, c.grid, c.mask, z, **obs_rec)

        del z

        message(c.comm_rec, 'number of '+obs_rec['name']+' obs from '+obs_rec['source']+': {}\n'.format(len(seq['obs'])), c.pid_rec)

        obs_seq[obs_rec_id] = seq

    return obs_seq


@bcast_by_root(c.comm)
def partition_grid(c):
    """
    Generate spatial partitioning of the domain, given config c

    Return: partitions
    """

    if c.assim_mode == 'batch':
        ##divide into square tiles with nx_tile grid points in each direction
        ##the workload on each tile is uneven since there are masked points
        ##so we divide into 3*nproc tiles so that they can be distributed
        ##according to their load (number of unmasked points)
        nx_tile = np.maximum(int(np.round(np.sqrt(c.nx * c.ny / c.nproc_mem / 3))), 1)

        ##a list of (istart, iend, di, jstart, jend, dj) for tiles
        ##note: we have 3*nproc entries in the list
        partitions = [(i, np.minimum(i+nx_tile, c.nx), 1,   ##istart, iend, di
                       j, np.minimum(j+nx_tile, c.ny), 1)   ##jstart, jend, dj
                      for j in np.arange(0, c.ny, nx_tile)
                      for i in np.arange(0, c.nx, nx_tile) ]

    elif c.assim_mode == 'serial':
        ##the domain is divided into tiles, each is formed by nproc_mem elements
        ##each element is stored on a different pid_mem
        ##for each pid, its loc points cover the entire domain with some spacing

        ##list of possible factoring of nproc_mem = nx_intv * ny_intv
        ##pick the last factoring that is most 'square', so that the interval
        ##is relatively even in both directions for each pid
        nx_intv, ny_intv = [(i, int(c.nproc_mem / i))
                            for i in range(1, int(np.ceil(np.sqrt(c.nproc_mem))) + 1)
                            if c.nproc_mem % i == 0][-1]

        ##a list of (ist, ied, di, jst, jed, dj) for slicing
        ##note: we have nproc_mem entries in the list
        partitions = [(i, nx, nx_intv, j, ny, ny_intv)
                      for j in np.arange(ny_intv)
                      for i in np.arange(nx_intv) ]

    return partitions


@bcast_by_root(c.comm_mem)
def assign_obs(c, state_info, obs_info, partitions, obs_rec_list, obs_seq):
    """
    Assign the observation sequence to each partition par_id

    Inputs: c, state_info, obs_info, partitions, obs_rec_list, obs_seq

    Returns: obs_inds
    """

    ##each pid_rec has a subset of obs_rec_list
    obs_inds_pid = {}
    for obs_rec_id in obs_rec_list[c.pid_rec]:
        obs_rec = obs_seq[obs_rec_id]
        obs_inds_pid[obs_rec_id] = {}

        if c.assim_mode == 'batch':
            ##1. screen horizontally for obs inside hroi of partition par_id
            hroi = obs_info['records'][obs_rec_id]['hroi']
            xo = np.array(obs_rec['x'])  ##obs x,y
            yo = np.array(obs_rec['y'])
            x = c.grid.x[0,:]   ##grid x,y
            y = c.grid.y[:,0]

            ##loop over partitions with par_id
            for par_id in range(len(partitions)):
                ist,ied,di,jst,jed,dj = partitions[par_id]

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
            full_inds = np.arange(len(obs_rec['obs']))

            inds = distribute_tasks(c.comm, full_inds)
            for par_id in range(c.nproc_mem):
                obs_inds_pid[obs_rec_id][par_id] = inds[par_id]

    ##now each pid_rec has figured out obs_inds for its own list of obs_rec_ids, we
    ##gather all obs_rec_id from different pid_rec to form the complete obs_inds dict
    obs_inds = {}
    for entry in c.comm_rec.allgather(obs_inds_pid):
        for obs_rec_id, data in entry.items():
            obs_inds[obs_rec_id] = data

    return obs_inds


@bcast_by_root(c.comm_mem)
def build_par_tasks(c, partitions, obs_info, obs_inds):
    par_list_full = np.arange(len(partitions))

    if c.assim_mode == 'batch':
        ##distribute the list of par_id according to workload to each pid
        ##number of unmasked grid points in each tile
        nlpts_loc = np.array([np.sum((~c.mask[jst:jed:dj, ist:ied:di]).astype(int))
                              for ist,ied,di,jst,jed,dj in partitions] )

        ##number of observations within the hroi of each tile, at loc,
        ##sum over the len of obs_inds for obs_rec_id over all obs_rec_ids
        nlobs_loc = np.array([np.sum([len(obs_inds[r][p])
                                      for r in obs_info['records'].keys()])
                              for p in par_list_full] )

        workload = np.maximum(nlpts_loc, 1) * np.maximum(nlobs_loc, 1)
        par_list = distribute_tasks(c.comm_mem, par_list_full, workload)

    if c.assim_mode == 'serial':
        ##just assign each partition to each pid, pid==par_id
        par_list = {p:p for p in range(c.nproc_mem)}

    return par_list


def prepare_obs_from_state(c, state_info, mem_list, rec_list, obs_info, obs_rec_list, obs_seq, fields, z_fields):
    """
    Prepare the obs from state (obs_prior) in parallel, run state_to_obs to obtain obs_prior_seq

    Inputs: c, state_info, mem_list, rec_list, obs_info, obs_rec_list, obs_seq, fields, z_fiels

    Return: obs_prior_seq
    """
    pid_show = [p for p,lst in obs_rec_list.items() if len(lst)>0][0] * c.nproc_mem
    message(c.comm, 'compute obs priors\n', pid_show)
    obs_prior_seq = {}

    ##process the obs, each proc gets its own workload as a subset of
    ##all proc goes through their own task list simultaneously
    nr = len(obs_rec_list[c.pid_rec])
    nm = len(mem_list[c.pid_mem])
    for m, mem_id in enumerate(mem_list[c.pid_mem]):
        for r, obs_rec_id in enumerate(obs_rec_list[c.pid_rec]):

            ##this is the obs record to process
            obs_rec = obs_info['records'][obs_rec_id]

            seq = state_to_obs(c, state_info, mem_list, rec_list, member=mem_id,
                               model_fld=fields, model_z=z_fields,
                               **obs_rec, **obs_seq[obs_rec_id])

            obs_prior_seq[mem_id, obs_rec_id] = seq

            message(c.comm, progress_bar(m*nr+r, nr*nm), pid_show)

    message(c.comm, ' done.\n', pid_show)

    return obs_prior_seq


def state_to_obs(c, state_info, mem_list, rec_list, **kwargs):
    """
    ##the top-level obs forward operator routine
    ##provides several ways to compute obs_prior from state fields
    """

    mem_id = kwargs['member']  ##if None, try to get variable from truth_file
    time = kwargs['time']  ##obs time step
    obs_name = kwargs['name']
    is_vector = kwargs['is_vector']

    obs_x = np.array(kwargs['x'])   ##from obs_seq the x, y, z coordinates
    obs_y = np.array(kwargs['y'])
    obs_z = np.array(kwargs['z'])
    nobs = len(obs_x)

    obs_grid = Grid(c.grid.proj, obs_x, obs_y, regular=False)
    c.grid.dst_grid = obs_grid

    if is_vector:
        obs_seq = np.full((2, nobs), np.nan)
    else:
        obs_seq = np.full(nobs, np.nan)

    ##obs dataset source module
    obs_src = importlib.import_module('dataset.'+kwargs['source'])
    ##model source module
    model_src = importlib.import_module('models.'+kwargs['model'])

    ##if obs variable is one of the state variable, or can be computed by the model,
    ## then we just need to collect the 3D variable and interpolate in x,y,z
    if obs_name in model_src.variables:

        levels = model_src.variables[obs_name]['levels']
        for k in range(len(levels)):
            if obs_name in [r['name'] for r in c.state_def]:
                ##the obs is one of the state variables
                ##find its corresponding rec_id
                rec_id = [i for i,r in state_info['fields'].items() if r['name']==obs_name and r['k']==levels[k]][0]

                ##if the current pid stores this field, just read it
                if rec_id in rec_list and mem_id in mem_list and 'model_fld' in kwargs and 'model_z' in kwargs:
                    z = kwargs['model_z'][mem_id, rec_id]
                    fld = kwargs['model_fld'][mem_id, rec_id]

                else:  ##or read field from state binfile
                    path = c.work_dir+'/analysis/'+t2s(time)+c.s_dir
                    z = read_field(path+'/z_coords.bin', state_info, c.mask, 0, rec_id)
                    fld = read_field(path+'/prior_state.bin', state_info, c.mask, mem_id, rec_id)

            else:  ##or get the field from model_src.read_var
                if k == 0:  ##initialize grid obj for convert
                    path = c.work_dir+'/forecast/'+t2s(time)+'/'+kwargs['model']
                    grid = model_src.read_grid(path, **kwargs)
                    grid.dst_grid = c.grid
                z_ = grid.convert(model_src.z_coords(path, grid, **kwargs))
                z = np.array([z_, z_]) if is_vector else z_
                fld = grid.convert(model_src.read_var(path, grid, **kwargs), is_vector=is_vector)

            ##horizontal interp field to obs_x,y, for current layer k
            if is_vector:
                z = c.grid.convert(z[0, ...], coarse_grain=False)
                zc = np.array([z, z])
            else:
                zc = c.grid.convert(z, coarse_grain=False)

            vc = c.grid.convert(fld, is_vector=is_vector, coarse_grain=False)

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
                obs_seq[..., inds] = vc[..., inds]

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
                obs_seq[..., inds] = vi[..., inds]

            if k == len(levels)-1:
                ##the last level: constant vc from z=zc-dzc/2 to zc
                inds = np.where(np.logical_and(obs_z >= np.minimum(zc-0.5*dzc, zc),
                                               obs_z <= np.maximum(zc-0.5*dzc, zc)) )
                obs_seq[..., inds] = vc[..., inds]

            if k < len(levels)-1:
                ##make a copy of current layer as 'previous' for next k
                zp = zc.copy()
                vp = vc.copy()
                dzp = dzc.copy()

    ##or, if dataset module provides an obs_operator, use it to compute obs
    elif kwargs['model'] in obs_src.obs_operator:
        path = c.work_dir+'/forecast/'+c.time+'/'+obs_rec['model']
        operator = obs_src.obs_operator[kwargs['model']]
        obs_seq = operator(path, c.grid, c.mask, **kwargs)

    else:
        raise ValueError('unable to obtain obs prior for '+obs_name)

    return obs_seq

def transpose_obs_to_lobs(c, mem_list, rec_list, obs_rec_list, par_list, obs_inds, input_obs, ensemble=False):
    """
    Transpose obs from field-complete to ensemble-complete

    Within comm_mem, send the subset of input_obs with mem_id and par_id from the source proc (src_pid)
    to the destination proc (dst_pid)

    If ensemble = True: the input_obs is the obs_prior_seq[mem_id, obs_rec_id]
    Return: output_obs[mem_id, obs_rec_id][par_id] as the local observation priors sequence

    If ensemble = False: the input_obs is the original obs_seq[obs_rec_id]={'obs':array,'x':array,'y':...}
    Return: output_obs[obs_rec_id][par_id] as the local observation sequence {'obs':array,'x':array,'y':...}

    """
    pid_show = [p for p,lst in obs_rec_list.items() if len(lst)>0][0] * c.nproc_mem

    ##Step 1: transpose to ensemble-complete by exchanging mem_id, par_id in comm_mem
    ##        input_obs -> tmp_obs
    message(c.comm, 'transpose obs to local obs\n', pid_show)
    tmp_obs = {}  ##local obs at intermediate stage

    nr = len(rec_list[c.pid_rec])
    for r, obs_rec_id in enumerate(obs_rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in mem_list.items()])

        for m in range(nm_max):

            ##prepare the obs seq for sending if not at the end of mem_list
            if m < len(mem_list[c.pid_mem]):
                mem_id = mem_list[c.pid_mem][m]
                if ensemble:  ##this is the obs prior seq
                    seq = input_obs[mem_id, obs_rec_id].copy()
                else:
                    if mem_id == 0:  ##this is the obs seq, just let mem_id=0 send it
                        seq = input_obs[obs_rec_id].copy()

            ##the collective send/recv follows the same idea under state.transpose_field_to_state
            ##1) receive lobs_seq from src_pid, for src_pid<pid first
            for src_pid in np.arange(0, c.pid_mem):
                if m < len(mem_list[src_pid]):
                    src_mem_id = mem_list[src_pid][m]
                    if ensemble:
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                    else:
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

            ##2) send my obs chunk to a list of dst_pid, send to dst_pid>=pid first
            ##   then cycle back to send to dst_pid<pid. i.e. the dst_pid sequence is
            ##   [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
            if m < len(mem_list[c.pid_mem]):
                for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    if ensemble:
                        ##this is the obs prior seq for mem_id, obs_rec_id
                        ##for each par_id, assemble the subset lobs_seq using obs_inds
                        lobs_seq = {}
                        for par_id in par_list[dst_pid]:
                            inds = obs_inds[obs_rec_id][par_id]
                            lobs_seq[par_id] = seq[..., inds]

                        if dst_pid == c.pid_mem:
                            ##pid already stores the lobs_seq, just copy
                            tmp_obs[mem_id, obs_rec_id] = lobs_seq
                        else:
                            ##send lobs_seq to dst_pid
                            c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

                    else:
                        if mem_id == 0:
                            ##this is the obs seq with keys 'obs','x','y','z','t'
                            ##assemble the lobs_seq dict with same keys but subset obs_inds
                            ##do this for each par_id to get the full lobs_seq
                            lobs_seq = {}
                            for par_id in par_list[dst_pid]:
                                lobs_seq[par_id] = {}
                                inds = obs_inds[obs_rec_id][par_id]
                                for key, value in seq.items():
                                    lobs_seq[par_id][key] = value[..., inds]

                            if dst_pid == c.pid_mem:
                                ##pid already stores the lobs_seq, just copy
                                tmp_obs[obs_rec_id] = lobs_seq
                            else:
                                ##send lobs_seq to dst_pid's lobs
                                c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

            ##3) finish receiving lobs_seq from src_pid, for src_pid>pid now
            for src_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                if m < len(mem_list[src_pid]):
                    src_mem_id = mem_list[src_pid][m]
                    if ensemble:
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                    else:
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

            if m < len(mem_list[c.pid_mem]):
                ##free up memory
                if ensemble:
                    del input_obs[mem_id, obs_rec_id]
                else:
                    if mem_id == 0:
                        del input_obs[obs_rec_id]

        message(c.comm, progress_bar(r, nr), pid_show)

    message(c.comm, ' done.\n', pid_show)

    ##Step 2: collect all obs records (all obs_rec_ids) on pid_rec
    ##        tmp_obs -> output_obs
    output_obs = {}
    for entry in c.comm_rec.allgather(tmp_obs):
        for key, data in entry.items():
            output_obs[key] = data

    return output_obs


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


def output_obs(c, obs_seq):
    pass



