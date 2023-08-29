import numpy as np
from numba import njit
from .parallel import distribute_tasks, message
from .state import xy_inds, read_field_info, uniq_fields, read_local_state, write_local_state
from .obs import assign_obs_inds, read_local_obs

##top-level routine for performing the DA analysis
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallization
def local_analysis(c, comm, prior_state_file, post_state_file, obs_seq_file):

    # inds = xy_inds(c.mask)

    ##read obs_seq

    ##go through obs_seq and assign nlobs to each local_ind
    obs_local_inds = assign_obs_inds(c, comm, obs_seq_file)




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


