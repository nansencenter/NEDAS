import numpy as np
import struct
import importlib
from datetime import datetime, timedelta
from .common import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from .parallel import distribute_tasks

##top-level routine to prepare the obs, read dataset files and convert to obs_seq
##obs_info contains obs value, err, and coordinates (t, z, y, x); when obs_info is
##ready, each obs is processed by state_to_obs
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallelization
def process_obs(c, comm):
    ##name of the binfile for the current c.time and c.scale
    ##make a separate dir for each scale if more than 1 scale
    s_dir = f'/s{c.scale+1}' if c.nscale>1 else ''
    binfile = c.work_dir+'/analysis/'+c.time+s_dir+'/obs_seq.bin'

    ##parse c.obs_def, read_obs from each type and generate obs_seq
    ##this is done by the first processor and broadcast
    if comm.Get_rank() == 0:
        info, seq = obs_seq(c)
        with open(binfile, 'wb'):  ##initialize the binfile in case it doesn't exist
            pass
        write_obs_info(binfile, info)
        write_obs_seq(binfile, info, seq)
    else:
        info = None
    info = comm.bcast(info, root=0)

    obs_bank = {}

    ##now start processing the obs seq, each processor gets its own workload as a subset of nobs
    for i in distribute_tasks(comm, np.arange(len(info['obs_seq']))):
        rec = info['obs_seq'][i]

        ##if variable is in prior_state fields, just read and interp in z

        ##else try obs_operator


##parse obs_def in config, read the dataset files to get the needed obs_seq
##for each obs indexed by obs_id, obtain its coordinates and properties in info
def obs_info(c):
    ##initialize
    info = {'nobs':0, 'nens':c.nens, 'obs_seq':{}}
    seq = []
    obs_id = 0  ##obs index in sequence
    pos = 0     ##f.seek position

    ##loop through time slots in obs window
    for t in s2t(c.time) + c.obs_ts*timedelta(hours=1):

        ##loop through obs variables defined in config
        for variable, rec in c.obs_def.items():

            ##load the dataset module
            src = importlib.import_module('dataset.'+rec['source'])
            assert variable in src.variables, 'variable '+variable+' not defined in dataset.'+rec['source']+'.variables'

            ##directory storing the dataset files for this variable
            path = c.data_dir+'/'+rec['source']

            kwargs = {'name': variable,
                      'source': rec['source'],
                      'model': rec['model'],
                      'err_type': rec['err_type'],
                      'err': rec['err'],
                      'dtype': src.variables[variable]['dtype'],
                      'is_vector': src.variables[variable]['is_vector'],
                      'z_type': src.variables[variable]['z_type'],
                      'units': src.variables[variable]['units'], }

            ##read dataset files and obtain a list of obs
            obs_info, obs_seq = src.read_obs(path, **kwargs)

            ##if using synthetic obs, we get simulated obs from model truth runs
            ##at obs coords and replace the obs values in obs_seq
            # if c.use_synthetic_obs:
            #     ##directory storing the truth state from a nature run
            #     state_path = c.data_dir+'/truth/'+c.time+'/'+rec['model']
            #     # ##compute obs from model states
            #     obs_seq = state_to_obs(state_path, obs_seq, **rec)

            ##loop through individual obs
            for obs_rec in obs_seq:
                ##add properties specific to this obs, then add the rec to the full obs_seq
                obs_rec.update(kwargs.copy())
                obs_rec.update({'pos':pos})
                info['obs_seq'][obs_id] = obs_rec
                ##update f.seek position
                nv = 2 if src.variables[variable]['is_vector'] else 1
                size = type_size[src.variables[variable]['dtype']]
                pos += nv * size * (c.nens+1)
                obs_id += 1

    info['nobs'] = obs_id

    return info, seq


def write_obs_info(binfile, info):
    with open(binfile.replace('.bin','.dat'), 'wt') as f:
        ##line 1: nobs, nens
        ##line 2:end: list of obs (one per line):
        ##      varname, source, model, dtype, is_vector, units, z_type, err_type, err, x, y, z, t, seek_position
        f.write('%i %i\n' %(info['nobs'], info['nens']))
        for i, rec in info['obs_seq'].items():
            f.write('%s %s %s %s %i %s %s %s %f %f %f %f %f %i\n' %(rec['name'], rec['source'], rec['model'], rec['dtype'], int(rec['is_vector']), rec['units'], rec['z_type'], rec['err_type'], rec['err'], rec['x'], rec['y'], rec['z'], t2h(rec['t']), rec['pos']))


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
                   'z_type':ss[6],
                   'err_type': ss[7],
                   'err': np.float32(ss[8]),
                   'x': np.float32(ss[9]),
                   'y': np.float32(ss[10]),
                   'z': np.float32(ss[11]),
                   't': h2t(np.float32(ss[12])),
                   'pos': int(ss[13]), }
            info['obs_seq'][obs_id] = rec
            obs_id += 1
        return info

##output an obs_seq to the binfile for a member (obs_prior), if member=None it is the actual obs
def write_obs_seq(binfile, info, obs_seq, member=None):
    ##
    with open(binfile, 'r+b') as f:
        for i, rec in info['obs_seq'].items():
            nv = 2 if rec['is_vector'] else 1
            size = type_size[rec['dtype']]
            if member is None:
                m = 0
            else:
                assert member<info['nens'], f'member = {member} is larger than ensemble size'
                m = member+1
            f.seek(rec['pos'] + nv*size*m)
            f.write(struct.pack(nv*type_dic[rec['dtype']], *obs_seq[i]))


def obs_operator(path, obs_seq, **kwargs):
    src = importlib.import_module('models.'+kwargs['model'])

    # if kwargs['name'] in src.
    ##if obs variables exists in state var_name, get it directly

    ##otherwise, try getting the obs_variable through model.src.obs_operator
    # obs_seq_bank

    ##else: error, don't know how to get obs_variable from model.src

    return 1


def assign_obs_inds(c, comm, field_info):

    return obs_inds


def read_local_obs(binfile):
    pass

