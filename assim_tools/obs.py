import numpy as np
from .common import type_convert, type_dic, type_size, t2h, h2t
from .state import state_variables

##observation variable definition
obs_variables = {'seaice_drift': {'is_vector':True, 'units':'m/s', 'z_units':'m'},
                 'seaice_conc':  {'is_vector':False, 'units':'%', 'z_units':'m'},
                 'seaice_thick': {'is_vector':False, 'units':'m', 'z_units':'m'},
                 'ocean_surf_temp': {'is_vector':False, 'units':'K', 'z_units':'m'},
                 'ocean_temp': {'is_vector':False, 'units':'K', 'z_units':'m'},
                 }

def obs_impact_factor(obs_def_file, state_def_file):
    ##observation impact factor dict[obs_variable][state_variable]
    ##first make a list of state variable and default impact as 1.
    state_list = {}
    with open(state_def_file, 'r') as f:
        for lin in f.readlines():
            vname = lin.split()[0]
            state_list.update({vname:1.})

    impact = {}
    ##go through obs_def
    with open(obs_def_file, 'r') as f:
        for lin in f.readlines():
            ss = lin.split()
            obs_vname = lin.split()[0]
            obs_impact = state_list.copy()
            ##adjust impact factor for state variables listed in obs_def
            if len(lin.split()) > 4:
                for ss in lin.split()[4].split(','):
                    vname, factor = ss.split('=')
                    obs_impact[vname] = np.float32(factor)
            impact.update({obs_vname:obs_impact})
    return impact

def read_obs_info(filename):
    with open(filename, 'r') as f:
        info = {}
        oid = 0
        for lin in f.readlines():
            ss = lin.split()
            rec = {'var_name': ss[0],
                   'source': ss[1],
                   'dtype': ss[2],
                   'is_vector': ss[3],
                   'z_units':ss[4],
                   'err_type': ss[5],
                   't': np.float32(ss[6]),
                   'z': np.float32(ss[7]),
                   'y': np.float32(ss[8]),
                   'x': np.float32(ss[9]),
                   'hroi': np.float32(ss[10]),
                   'vroi': np.float32(ss[11]),
                   'troi': np.float32(ss[12]),
                   'err': np.float32(ss[13])}
            info.update({oid:rec})
            oid += 1
        return info

def write_obs_info(filename, info):
    with open(filename, 'wt') as f:
        for i, rec in info.items():
            f.write('%s %s %s %i %s %s %f %f %f %f %f %f %f %f\n' %(rec['var_name'], rec['source'], rec['obs'], rec['t'], rec['z_type'], rec['z'], rec['y'], rec['x'], rec['hroi'], rec['vroi'], rec['err_type'], rec['err']))

def prepare_obs(c, time):
    ##parse the obs_def and collect all obs for the analysis time
    ##this can be done prior to the cycling (forecast and analysis) with parallel
    ##just as soon as observations are available
    ##saves the obs in obs.bin/dat file
    import importlib

    ##time window (t1, t2)
    t1 = h2t(t2h(time) - c.OBS_WINDOW_MIN)
    t2 = h2t(t2h(time) + c.OBS_WINDOW_MAX)

    binfile = c.WORK_DIR+'/obs.bin'  #TODO: timestr

    ##go through obs_def file and process obs variables one at a time
    with open(c.OBS_DEF_FILE, 'r') as f:
        info = {}
        nobs = np.uint64(0)
        for lin in f.readlines():
            ss = lin.split()
            assert len(ss) in (4,5), 'obs_def format error, should be "varname, src, z_units, err_type, err, hroi, vroi, troi, impact_state_list[optional]'
            obs_vname = ss[0]
            obs_src = ss[1]
            obs_hroi = np.float32(ss[2])
            obs_vroi = np.float32(ss[3])

            ##directory storing the obs dataset
            path = c.WORK_DIR + '/obs/' + obs_src

            ##load dataset module to process this obs
            src = importlib.import_module('dataset.'+obs_src)

            obs_seq = src.get_obs(path, name=obs_vname, time=(t1, t2))
            comp = ('_x','_y') if obs_variables[obs_vname]['is_vector'] else ('',)
            for p in range(len(obs_seq)):

                for i in range(len(comp)):
                    rec = {'var_name':obs_vname,
                           'source':obs_src,}
                    info['obs'].update({obs_id:rec})
                    obs_id += 1
        info['nobs'] = obs_id+1
        write_obs_info(binfile, info)

def assign_obs_inds(c, comm, field_info):

    return obs_inds

def read_obs(filename):
    pass

def write_obs(filename, obs, obs_prior, member=None):
    pass

def get_z_coords():
    pass

def prepare_obs_prior(c, comm, time):
    ##c: config module
    ##comm: mpi4py communicator
    ##time: analysis time (datetime obj)
    import importlib
    from .parallel import distribute_tasks

    ##if obs variables exists in state var_name, get it directly

    ##otherwise, try getting the obs_variable through model.src.get_obs_prior
    bank ##intermediate obs variables

    ##else: error, don't know how to get obs_variable from model.src

