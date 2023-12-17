import numpy as np
import struct
import importlib
from datetime import datetime, timedelta
import config as c
from conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from log import message, progress_bar
from parallel import bcast_by_root, distribute_tasks
from perturb import random_field
from .state import read_field

##Note:
## The observation has dimensions: variable, time, z, y, x
## Since the observation network is typically irregular, we store the obs record
## for each variable in a 1d sequence, with coordinates (t,z,y,x), and size nobs
##
## To parallelize workload, we distribute each obs record over all the processors
## - for batch assimilation mode, each pid stores the list of local obs within the
##   hroi of its tiles, with size nlobs (number of local obs)
## - for serial mode, each pid stores a non-overlapping subset of the obs list,
##   here 'local' obs (in storage sense) is broadcast to all pid before computing
##   its update to the state/obs near that obs.
##
## The hroi is separately defined for each obs record.
## For very large hroi, the serial mode is more parallel efficient option, since
## in batch mode the same obs may need to be stored in multiple pids
##
## To compare to the observation, obs_prior simulated by the model needs to be
## computed, they have dimension [nens, nlobs], indexed by (mem_id, obs_id)


## parse c.obs_def, read the dataset files to get the obs records
## for each obs_id in record, obtain coordinates and properties for that obs_id
## input: c: config module with the environment variables
## return: info dict with some dimensions and list of uniq obs records
@bcast_by_root(c.comm)
def parse_obs_info(c):
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


def build_obs_tasks(c):
    ##this is called after build_state_tasks()
    ##so c.mem_list should be already set

    ##obs_rec_id list as tasks
    c.obs_rec_list = distribute_tasks(c.comm_rec, [i for i in c.obs_info['records'].keys()])

    ##collect (mem_id, obs_rec_id) to make the obs_list
    ##to make the task loop easier to read
    c.obs_field_list = {}
    ntask = []
    for p in range(c.nproc):
        lst = [(m, r)
               for m in c.mem_list[p%c.nproc_mem]
               for r in c.obs_rec_list[p//c.nproc_mem] ]
        ntask.append(len(lst))
        c.obs_field_list[p] = lst
    c.obs_ntask_max = np.max(ntask)


##read ens-mean z coords from z_file at obs time
def mean_z_coords(c, time):
    ##first, get a list of indices k
    k_list = list(set([r['k'] for i,r in c.state_info['fields'].items() if r['time']==time]))

    ##get z coords for each level
    z = np.zeros((len(k_list), c.ny, c.nx))
    for k in range(len(k_list)):

        ##the rec_id in z_file corresponding to this level
        ##there can be multiple records (different state variables)
        ##we take the first one
        rec_id = [i for i,r in c.state_info['fields'].items() if r['time']==time and r['k']==k_list[k]][0]

        ##read the z field with (mem_id=0, rec_id) from z_file
        z_file = c.work_dir+'/analysis/'+t2s(time)+c.s_dir+'/z_coords.bin'
        z_fld = read_field(z_file, c.state_info, c.mask, 0, rec_id)

        ##assign the coordinates to z(k)
        if c.state_info['fields'][rec_id]['is_vector']:
            z[k, ...] = z_fld[0, ...]
        else:
            z[k, ...] = z_fld

    return z


##prepare the obs in parallel, read dataset files and convert to obs_seq
##which contains obs value and coordinates time, z, y, x
##since this is the actual obs (1 copy), only pid_mem==0 will do the work
##and broadcast to all pid_mem in comm_mem
@bcast_by_root(c.comm_mem)
def prepare_obs(c):

    message(c.comm_rec, 'reading obs_seq from dataset\n', 0)
    obs_seq = {}

    ##get obs_seq from dataset module, each pid_rec gets its own workload
    ##as a subset of obs_rec_list
    for obs_rec_id in c.obs_rec_list[c.pid_rec]:
        obs_rec = c.obs_info['records'][obs_rec_id]

        ##load the dataset module
        src = importlib.import_module('dataset.'+obs_rec['source'])
        assert obs_rec['name'] in src.variables, 'variable '+obs_rec['name']+' not defined in dataset.'+obs_rec['source']+'.variables'

        ##directory storing the dataset files for this variable
        path = c.data_dir+'/'+obs_rec['source']

        ##read ens-mean z coords from z_file for this obs network
        z = mean_z_coords(c, obs_rec['time'])

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

        message(c.comm_rec, 'number of '+obs_rec['name']+' obs from '+obs_rec['source']+': {}\n'.format(len(seq['obs'])), c.pid_rec)

        obs_seq[obs_rec_id] = seq

    return obs_seq


##generate spatial partitioning of the domain
@bcast_by_root(c.comm)
def partition_grid(c):

    if c.assim_mode == 'batch':
        ##divide into square tiles with nx_tile grid points in each direction
        ##the workload on each tile is uneven since there are masked points
        ##so we divide into 3*nproc tiles so that they can be distributed
        ##according to their load (number of unmasked points)
        nx_tile = int(np.round(np.sqrt(c.nx * c.ny / c.nproc / 3)))

        ##a list of (istart, iend, di, jstart, jend, dj) for tiles
        ##note: we have 3*nproc entries in the list
        loc_list_full = [(i, np.minimum(i+nx_tile, c.nx), 1,   ##istart, iend, di
                          j, np.minimum(j+nx_tile, c.ny), 1)   ##jstart, jend, dj
                          for j in np.arange(0, c.ny, nx_tile)
                          for i in np.arange(0, c.nx, nx_tile) ]

    elif c.assim_mode == 'serial':
        ##the domain is divided into tiles, each tile is formed by nproc elements
        ##each element is stored on a different pid
        ##for each pid, its loc points cover the entire domain with some spacing

        ##list of possible factoring of nproc = nx_intv * ny_intv
        ##pick the last factoring that is most 'square', so that the interval
        ##is relatively even in both directions for each pid
        nx_intv, ny_intv = [(i, int(c.nproc / i))
                            for i in range(1, int(np.ceil(np.sqrt(c.nproc))) + 1)
                            if c.nproc % i == 0][-1]

        ##a list of (ist, ied, di, jst, jed, dj) for slicing
        ##note: we have nproc entries in the list
        loc_list_full = [(i, nx, nx_intv, j, ny, ny_intv)
                         for j in np.arange(ny_intv)
                         for i in np.arange(nx_intv) ]

    return loc_list_full


@bcast_by_root(c.comm_mem)
def assign_obs_to_loc(c, obs_seq):

    ##each pid_rec has a subset of obs_rec_list
    obs_inds_pid = {}
    for obs_rec_id in c.obs_rec_list[c.pid_rec]:
        obs_rec = obs_seq[obs_rec_id]
        obs_inds_pid[obs_rec_id] = {}

        if c.assim_mode == 'batch':
            ##loop over tiles in c.loc_list_full given by partition_grid
            for loc in range(len(c.loc_list_full)):
                ist,ied,di,jst,jed,dj = c.loc_list_full[loc]

                hroi = c.obs_info['records'][obs_rec_id]['hroi']
                xo = np.array(obs_rec['x'])  ##obs x,y
                yo = np.array(obs_rec['y'])
                x = c.grid.x[0,:]            ##grid x,y
                y = c.grid.y[:,0]

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
                ##hroi of any points in tile[loc]
                inds = np.where(np.logical_or(cond1, np.logical_or(cond2, cond3)))
                obs_inds_pid[obs_rec_id][loc] = inds

        elif c.assim_mode == 'serial':
            ##locality doesn't matter, we just divide obs_rec into nproc parts
            inds = distribute_tasks(c.comm, np.arange(len(obs_rec['obs'])))
            for p in range(c.nproc):
                obs_inds_pid[obs_rec_id][p] = inds[p]

    ##gather all obs_rec_id from pid_rec to form the complete obs_inds dict
    obs_inds = {}
    for entry in c.comm_rec.allgather(obs_inds_pid):
        for obs_rec_id,inds in entry.items():
            obs_inds[obs_rec_id] = inds

    return obs_inds


@bcast_by_root(c.comm)
def build_loc_tasks(c):
    if c.assim_mode == 'batch':
        ##distribute the loc_list_full according to workload to each pid
        ##number of unmasked grid points in each tile
        nlpts_loc = np.array([np.sum((~c.mask[jst:jed:dj, ist:ied:di]).astype(int))
                              for ist,ied,di,jst,jed,dj in c.loc_list_full] )

        ##number of observations within the hroi of each tile, at loc,
        ##sum over the len of obs_inds for obs_rec_id over all obs_rec_ids
        nlobs_loc = np.array([np.sum([len(c.obs_inds[r][loc])
                                      for r in c.obs_info['records'].keys()])
                              for loc in range(len(c.loc_list_full))] )

        workload = np.maximum(nlpts_loc, 1) * np.maximum(nlobs_loc, 1)
        print(workload)
        loc_list = distribute_tasks(c.comm, c.loc_list_full, workload)

    if c.assim_mode == 'serial':
        ##just assign each loc slice to each pid
        loc_list = {pid:c.loc_list_full[pid] for pid in range(c.nproc)}

    return loc_list


def state_to_obs(c, obs_info, fields, z_coords, mem_id, obs_rec_id):

    obs_rec = obs_info['records'][obs_rec_id]
    obs_src = importlib.import_module('dataset.'+obs_rec['source'])
    model_src = importlib.import_module('models.'+obs_rec['model'])

    ##if obs variable is one of the state variable, or can be computed by the model,
    ## then we just need to collect the 3D variable and interpolate in x,y,z
    if obs_rec['name'] in model_src.variables:

        ##collect all fields
        levels = model_src.variables[obs_rec['name']]['levels']
        obs_field = np.zeros((len(levels), c.ny, c.nx))
        for k in range(len(levels)):
            levels[k]
            ##if the current pid stores this field
            # if 

            ##or read from state_file to get the field
            # else:


        ##horizontal interp using grid.convert
        ##vertical interp
        obs_seq = []


    ##or, if dataset module provides an obs_operator, we try to compute it by obs_src
    elif obs_rec['model'] in obs_src.obs_operator:
        path = c.work_dir+'/forecast/'+c.time+'/'+obs_rec['model']
        operator = obs_src.obs_operator[obs_rec['model']]
        obs_seq = operator(path, c.grid, c.mask, **obs_rec)

    else:
        raise ValueError('unable to obtain obs prior for '+obs_rec['name'])

    return obs_seq


def prepare_obs_prior(c):

    message(c.comm, 'prepare obs priors:\n', 0)

    ##process the obs, each proc gets its own workload as a subset of
    ##obs_list[pid] pointing to the list of tasks for each pid
    ntask_max = np.max([len(lst) for p,lst in c.obs_list.items()])

    ##all proc goes through their own task list simultaneously
    for task in range(ntask_max):

        ##process an obs record if not at the end of task list
        if task < len(c.obs_list[c.pid]):
            ##this is the obs record to process
            mem_id, obs_rec_id = c.obs_list[c.pid][task]
            # obs_rec = c.obs_info[]

    # message(comm, 'process_obs: getting obs_prior for each obs variable and member', 0)
    # for member,obs_id in distribute_tasks(comm, [(m,i) for m in range(info['nens']) for i in range(info['nobs'])])[pid]:
    #     rec = info['obs_seq'][obs_id]

        ##if variable is in prior_state just interp to obs location

        ##else try auxilary variables in state_obs

        ##else try obs_operator from model src

        ##tmp solution::
        # obs_var_key = (rec['name'], rec['model'], member)
        # if obs_var_key in obs_var_bank:
        #     obs_var = obs_var_bank[obs_var_key]
        # else:
        #     message(comm, 'state_to_obs: getting variable '+rec['name']+' from '+rec['model'])
        #     src = importlib.import_module('dataset.'+rec['source'])
        #     obs_var = src.state_to_obs(c.work_dir+'/forecast/'+c.time+'/'+rec['model'], c.grid, **rec, k=-1, member=member)
        #     obs_var_bank[obs_var_key] = obs_var
        # obs_rec = info['obs_seq'][obs_id]
        # obs_rec['value'] = c.grid.interp(obs_var, obs_rec['x'], obs_rec['y'])
        # write_obs(obs_seq_file, info, [obs_rec], member=member)

         # message(comm, '    {:15s} t={} z={:10.4f} y={:10.4f} x={:10.4f} member={:3d}'.format(obs_rec['name'], obs_rec['time'], obs_rec['z'], obs_rec['y'], obs_rec['x'], member+1))


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



