import numpy as np
from numba import njit
from .parallel import distribute_tasks, message
from .state import xy_inds, read_field_info, uniq_fields, read_local_state, write_local_state
from .obs import read_obs_info, assign_obs_inds, read_local_obs

##top-level routine for performing the DA analysis
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallization
def local_analysis(c, comm, prior_state_file, post_state_file, obs_seq_file):
    rank = comm.Get_rank()

    message(comm, 'local_analysis: gathering field_info and obs_info', 0)
    if rank == 0:
        field_info = read_field_info(prior_state_file)
        obs_info = read_obs_info(obs_seq_file)
    else:
        field_info = None
        obs_info = None
    field_info = comm.bcast(field_info, root=0)
    obs_info = comm.bcast(obs_info, root=0)

    message(comm, 'local_analysis: assigning local observation indices to grid points', 0)
    inds = xy_inds(c.mask)
    local_obs_inds = assign_obs_inds(c, comm, obs_seq_file)
    nlobs = np.array([len(lst) for lst in local_obs_inds.values()])
    local_inds_tasks = distribute_tasks(comm, inds[nlobs>0])

    print(rank, ' working on: ',  len(local_inds_tasks[rank]), ' inds')

    message(comm, 'local_analysis: reading in local state ensemble', 0)
    local_state = read_local_state(prior_state_file, field_info, c.mask, local_inds_tasks[rank])

    # message(comm, 'local_analysis: reading in local observation ensemble', 0)
    # local_obs = read_local_obs(obs_seq_file, obs_info, obs_inds_tasks[rank])

    # message(comm, 'local_analysis: running ETKF analysis', 0)

    # message(comm, 'local_analysis: writing out the updated local state ensemble', 0)


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


