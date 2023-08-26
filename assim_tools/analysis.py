import numpy as np
from numba import njit
from .parallel import distribute_tasks

##top-level routine for performing the DA analysis
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallization
def local_analysis(c, comm):

    ##
    s_dir = f'/s{c.scale+1}' if c.nscale>1 else ''
    state_file = c.work_dir+'/'+c.time+s_dir+'/state.bin'
    obs_seq_file = c.work_dir+'/'+c.time+s_dir+'/obs_seq.bin'

    field_info = read_field_info(state_file)
    x, y, mask = read_header(state_file, field_info)

    nens = field_info['nens']
    nfield = field_info['nfield']
    nx = field_info['nx']
    ny = field_info['ny']

    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    inds = jj*nx + ii

    obs_info = read_obs_info(obs_seq_file)
    for p, rec in obs_info['obs_seq'].items():
        vname = rec['name']
        xo = rec['x']
        yo = rec['y']

    # # obs_info = read_obs_info(obs_seq_file)

    # ##find locales with nlobs>0
    # local_inds = distribute_tasks(comm, inds):

    # local_state = read_local_state(state_file, field_info, mask, local_inds)
    # # local_obs = read_local_obs(obs_seq_file


###two-step solution, Anderson 2003, used in DART
###obs increment
# def obs_incr

###updating state variable using obs_incr
# def reg_factor
    # return state_ens_post


###(local) ETKF algorithm, similar to Hunt 2007, used in PDAF etc.
##
def ETKF(prior_state, obs, obs_prior, obs_err, loc_factor, ):
    post_state = prior_state.copy()
    return post_state

#@njit
#def T():


##location factor and distance calculation

##TODO: quantize distance so that some subrange local_factor are shared, so computation can speed up using look up table


##localization function
def local_factor(dist, ROI, local_type='GC'):
    ## dist: input distance, ndarray
    ## ROI: radius of influence, distance beyond which loc=0
    ## returns the localization factor loc
    loc = np.zeros(dist.shape)
    if ROI>0:
        if local_type == 'GC': ##Gaspari-Cohn localization function
            r = dist / (ROI / 2)
            loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
            ind1 = np.where(dist<ROI/2)
            loc[ind1] = loc1[ind1]
            r[np.where(r==0)] = 1e-10
            loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
            ind2 = np.where(np.logical_and(dist>=ROI/2, dist<ROI))
            loc[ind2] = loc2[ind2]
        else:
            raise ValueError('unknown localization function type: '+local_type)
    else:
        loc = np.ones(dist.shape)
    return loc


