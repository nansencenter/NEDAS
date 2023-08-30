import numpy as np
import struct
import importlib
from datetime import datetime, timedelta
from .common import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .parallel import distribute_tasks, message
from .state import xy_inds, read_local_state

##top-level routine to prepare the obs, read dataset files and convert to obs_seq
##obs_info contains obs value, err, and coordinates (t, z, y, x); when obs_info is
##ready, each obs is processed by state_to_obs
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallelization
def process_obs(c, comm, prior_state_file, obs_seq_file):
    proc_id = comm.Get_rank()
    message(comm, 'process_obs: parsing and generating obs_info', 0)
    ##parse c.obs_def, read_obs from each type and generate obs_seq
    ##this is done by the first processor and broadcast
    if proc_id  == 0:
        info = obs_info(c)
        with open(obs_seq_file, 'wb'):  ##initialize obs_seq.bin in case it doesn't exist
            pass
        write_obs_info(obs_seq_file, info)
        write_obs(obs_seq_file, info, info['obs_seq'].values(), member=None)
    else:
        info = None
    info = comm.bcast(info, root=0)

    obs_var_bank = {}

    ##now start processing the obs seq, each processor gets its own workload as a subset of nobs
    message(comm, 'process_obs: getting obs_prior for each obs variable and member', 0)
    for member,obs_id in distribute_tasks(comm, [(m,i) for m in range(info['nens']) for i in range(info['nobs'])])[proc_id]:
        rec = info['obs_seq'][obs_id]

        ##if variable is in prior_state fields, just read and interp in z

        ##else try obs_operator

        ##tmp solution::
        obs_var_key = (rec['name'], rec['model'], member)
        if obs_var_key in obs_var_bank:
            obs_var = obs_var_bank[obs_var_key]
        else:
            message(comm, 'state_to_obs: getting variable '+rec['name']+' from '+rec['model'])
            src = importlib.import_module('dataset.'+rec['source'])
            obs_var = src.state_to_obs(c.work_dir+'/forecast/'+c.time+'/'+rec['model'], c.grid, **rec, k=-1, member=member)
            obs_var_bank[obs_var_key] = obs_var
        obs_rec = info['obs_seq'][obs_id]
        obs_rec['value'] = c.grid.interp(obs_var, obs_rec['x'], obs_rec['y'])
        write_obs(obs_seq_file, info, [obs_rec], member=member)

         # message(comm, '    {:15s} t={} z={:10.4f} y={:10.4f} x={:10.4f} member={:3d}'.format(obs_rec['name'], obs_rec['time'], obs_rec['z'], obs_rec['y'], obs_rec['x'], member+1))


##parse obs_def in config, read the dataset files to get the needed obs_seq
##for each obs indexed by obs_id, obtain its coordinates and properties in info
##Note: this is run by the root processor
def obs_info(c):
    ##initialize
    info = {'nobs':0, 'nens':c.nens, 'obs_seq':{}}
    obs_id = 0  ##obs index in sequence
    pos = 0     ##f.seek position

    if c.use_synthetic_obs:
        network_bank = {}

    ##loop through time slots in obs window
    for time in s2t(c.time) + c.obs_ts*timedelta(hours=1):

        ##loop through obs variables defined in config
        for name, rec in c.obs_def.items():

            ##load the dataset module
            src = importlib.import_module('dataset.'+rec['source'])
            assert name in src.variables, 'variable '+name+' not defined in dataset.'+rec['source']+'.variables'

            ##directory storing the dataset files for this variable
            path = c.data_dir+'/'+rec['source']

            kwargs = {'name': name,
                      'source': rec['source'],
                      'model': rec['model'],
                      'err_type': rec['err_type'],
                      'err': rec['err'],
                      'dtype': src.variables[name]['dtype'],
                      'is_vector': src.variables[name]['is_vector'],
                      'z_units': src.variables[name]['z_units'],
                      'units': src.variables[name]['units'], }

            if c.use_synthetic_obs:
                ##generate synthetic obs network
                network_key = (rec['source'], time)
                if network_key not in network_bank:
                    obs_seq = src.random_network(c.grid, c.mask)
                else:
                    obs_seq = network_bank[network_key]

                ##read truth file and find obs value
                obs_var = src.state_to_obs(c.data_dir+'/truth/'+c.time+'/'+rec['model'], c.grid, **kwargs, k=-1, time=s2t(c.time))

                for i in obs_seq.keys():
                    obs_seq[i]['name'] = kwargs['name']
                    obs_seq[i]['time'] = s2t(c.time)
                    ##interp to obs location
                    obs_seq[i]['value'] = c.grid.interp(obs_var, obs_seq[i]['x'], obs_seq[i]['y'])

                    ##z_interp?

                    ##perturb with obs err
                    # obs_seq[i]['value'] += np.random.normal(0, 1) * kwargs['err']

            else:
                ##read dataset files and obtain a list of obs
                obs_seq = src.read_obs(path, c.grid, c.mask, **kwargs)

            ##loop through individual obs
            for obs_rec in obs_seq.values():
                ##add properties specific to this obs, then add the rec to the full obs_seq
                info['obs_seq'][obs_id] = {'pos':pos}
                info['obs_seq'][obs_id].update(kwargs)
                info['obs_seq'][obs_id].update(obs_rec)

                ##update f.seek position
                nv = 2 if src.variables[name]['is_vector'] else 1
                size = type_size[src.variables[name]['dtype']]
                pos += nv * size * (c.nens+1)
                obs_id += 1

    info['nobs'] = obs_id

    return info


##write obs_info to a .dat file accompanying the obs_seq bin file
def write_obs_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        f.write('{} {}\n'.format(info['nobs'], info['nens']))
        for rec in info['obs_seq'].values():
            f.write('{} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(rec['name'], rec['source'], rec['model'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['z_units'], rec['err_type'], rec['err'], rec['x'], rec['y'], rec['z'], t2h(rec['time']), rec['pos']))


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

##output an obs values to the binfile for a member (obs_prior), if member=None it is the actual obs
def write_obs(binfile, info, obs_seq, member=None):
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


def read_obs(binfile, info, obs_seq, member=None):
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


# def obs_operator(path, obs_seq, **kwargs):
#     pass


def assign_obs_inds(c, comm, obs_seq_file):
    proc_id = comm.Get_rank()

    if proc_id == 0:
        info = read_obs_info(obs_seq_file)
        nobs = info['nobs']

        inds = xy_inds(c.mask)   ##list of horizontal locale (grid points) indices
        x = c.grid.x.flatten()[inds]  ##x, y coords for each index
        y = c.grid.y.flatten()[inds]

        local_obs_inds = {}
        for i in inds:
            local_obs_inds[i] = []

        ##go through the list of obs
        for obs_id in range(nobs):
            rec = info['obs_seq'][obs_id]
            ##TODO: mfx,y needed here
            dist = np.hypot(x-rec['x'], y-rec['y']) ##horizontal distance from obs to each grid inds
            hroi = c.obs_def[rec['name']]['hroi']   ##horizontal localization distance for this obs
            local_inds = inds[dist<hroi]    ##the grid inds within the impact zone of this obs
            for i in local_inds:
                local_obs_inds[i].append(obs_id)  ##add this obs to the list of local grid inds
    else:
        local_obs_inds = None

    local_obs_inds = comm.bcast(local_obs_inds, root=0)

    return local_obs_inds


def uniq_obs(obs_seq):
    uoid = 0
    obs = {}
    for rec in obs_seq:
        for i in range(2 if rec['is_vector'] else 1):
            obs[uoid] = rec
            uoid += 1
    return obs

def read_local_obs(binfile, info, obs_inds):
    obs_seq = [info['obs_seq'][i] for i in obs_inds]

    nens = info['nens']
    uobs = uniq_obs(obs_seq)
    nuobs = len(obs_inds)

    local_obs = {'obs': np.full(nuobs, np.nan),
                 'obs_prior': np.full((nens, nuobs), np.nan),
                 'err': [r['err'] for r in uobs.values()],
                 'name': [r['name'] for r in uobs.values()],
                 'time': [t2h(r['time']) for r in uobs.values()],
                 'z': [r['z'] for r in uobs.values()],
                 'y': [r['y'] for r in uobs.values()],
                 'x': [r['x'] for r in uobs.values()],}

    for member in range(nens):
        obs_seq_member = read_obs(binfile, info, obs_seq, member=member)
        j = 0
        for rec in obs_seq_member:
            for i in range(2 if rec['is_vector'] else 1):
                local_obs['obs_prior'][member, j] = rec['value'][i]
                j += 1

    obs_seq_real = read_obs(binfile, info, obs_seq, member=None)
    j = 0
    for rec in obs_seq_member:
        for i in range(2 if rec['is_vector'] else 1):
            local_obs['obs'][j] = rec['value'][i]
            j += 1

    return local_obs


