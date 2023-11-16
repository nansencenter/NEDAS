import numpy as np
from numba import njit
from .parallel import distribute_tasks, message
# from .state import loc_inds, read_field_info, uniq_fields, read_local_state, write_local_state
# from .obs import read_obs_info, assign_obs_inds, read_local_obs

##top-level routine for performing the DA analysis
##inputs: c: config module for parsing env variables
##        comm: mpi4py communicator for parallization
def local_analysis(c, comm, prior_state_file, post_state_file, obs_seq_file):
    proc_id = comm.Get_rank()

    message(comm, 'local_analysis: gathering field_info and obs_info', 0)
    if proc_id == 0:
        field_info = read_field_info(prior_state_file)
        obs_info = read_obs_info(obs_seq_file)
    else:
        field_info = None
        obs_info = None
    field_info = comm.bcast(field_info, root=0)
    obs_info = comm.bcast(obs_info, root=0)

    nfield = len(uniq_fields(field_info))

    message(comm, 'local_analysis: assigning local observation indices to grid points', 0)
    inds = loc_inds(c.mask)
    # local_obs_inds = assign_obs_inds(c, comm, obs_seq_file)

    local_inds_tasks = distribute_tasks(comm, inds)
    nlocal = len(local_inds_tasks[proc_id])

    message(comm, 'local_analysis: reading in local state ensemble', 0)
    state_dict = read_local_state(comm, prior_state_file, field_info, c.mask, local_inds_tasks[proc_id])

    comm.Barrier()

    # message(comm, 'local_analysis: performing local analysis for each local index', 0)
    # for i, local_ind in enumerate(local_inds_tasks[proc_id]):
    #     if i%(len(local_inds_tasks[proc_id])//20) == 0:   ##only output 20 progress messages
    #         message(comm, f'   progress ... {(100*(i+1)//nlocal)}%', 0)

    #     obs_dict = read_local_obs(obs_seq_file, obs_info, local_obs_inds[local_ind])
    #     nlobs = len(obs_dict['obs'])

    #     #state_ = state_dict['state'][:, :, i]  ##[nens, nfield] at local_ind position
    #     name_ = state_dict['name']
    #     time_ = state_dict['time']
    #     z_ = np.mean(state_dict['z'][:, :, i], axis=0)  ##ens mean z coords at local_ind
    #     k_ = state_dict['k']
    #     y_ = c.grid.y.flatten()[local_ind]
    #     x_ = c.grid.x.flatten()[local_ind]

    #     ##option1: ETKF
    #     if c.filter_type == 'ETKF':
    #         weights_bank = {}  ##stored transform weights for a uniq combo of obs local impact factors
    #         for n in range(nfield):
    #             ##compute local impact factors given hroi,vroi,troi,and cross-variable impact
    #             loc_fac = ()
    #             for p in range(nlobs):
    #                 hdist = np.hypot(obs_dict['x'][p] - x_, obs_dict['y'][p] - y_)
    #                 hroi = c.obs_def[obs_dict['name'][p]]['hroi']
    #                 ##TODO: add z,t localization and impact factor
    #                 ##add loc_fac for obs p to the tuple
    #                 # loc_fac += (local_factor(hdist, hroi)[0],)
    #                 loc_fac += (1.0,)

    #             ##compute transform weight given local obs and state
    #             if loc_fac in weights_bank:
    #                 ##lookup the existing weights if obs impact factor exists
    #                 ##  since sometimes no localization is applied in v, t, or across variables
    #                 weights = weights_bank[loc_fac]
    #             else:
    #                 ##compute ETKF transform weights
    #                 weights = transform_weights(obs_dict['obs'], obs_dict['obs_prior'], obs_dict['err'], loc_fac)
    #                 weights_bank[loc_fac] = weights

    #             ##transform the ensemble
    #             state_dict['state'][:, n, i] = ensemble_transform(state_dict['state'][:, n, i], weights)

    #     ##EAKF, square-root filter
    #     elif c.filter_type == 'EAKF':
    #         pass

    #     else:
    #         if proc_id == 0:
    #             raise ValueError('filter_type '+c.filter_type+' is unsupported')
    # message(comm, '   complete', 0)

    message(comm, 'local_analysis: writing out the updated local state ensemble', 0)
    write_local_state(comm, post_state_file, field_info, c.mask, local_inds_tasks[proc_id], state_dict)


###(local) ETKF algorithm, similar to Hunt 2007, used in PDAF etc.
##
# @njit
def transform_weights(obs, obs_prior, obs_err, loc_fac):
    nens, nlobs = obs_prior.shape

    obs_prior_mean = np.mean(obs_prior, axis=0)
    sqrtm = np.sqrt(nens-1)

    S = (obs_prior - np.tile(obs_prior_mean, (nens, 1))) / np.tile(obs_err, (nens, 1)) / sqrtm * np.tile(loc_fac, (nens, 1))
    dy = (obs - obs_prior_mean) / obs_err / sqrtm * loc_fac

    SS = S @ S.T
    if np.isinf(SS).any() or np.isnan(SS).any():
        weights = np.eye(nens)
    else:
        e, v = np.linalg.eig(np.eye(nens) + SS)
        e = e.real
        v = v.real
        invSS = v @ np.diag(e**-1) @ v.T
        invSSsqrt = v @ np.diag(e**-0.5) @ v.T
        weights = invSS @ S @ dy + invSSsqrt

    return weights


def ensemble_transform(ens_state, weights):
    ens_state = ens_state @ weights
    return ens_state



###two-step solution, Anderson 2003, used in DART
###obs increment
# def obs_incr

###updating state variable using obs_incr
# def reg_factor
    # return state_ens_post



##location factor and distance calculation

##TODO: quantize distance so that some subrange local_factor are shared, so computation can speed up using look up table


##localization function
def local_factor(dist, roi, local_type='GC'):
    ## dist: input distance, ndarray
    ## roi: radius of influence, distance beyond which loc=0
    ## returns the localization factor loc
    dist = np.atleast_1d(dist)
    loc = np.zeros(dist.shape)
    if roi>0:
        if local_type == 'GC': ##Gaspari-Cohn localization function
            r = dist / (roi / 2)
            loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
            ind1 = np.where(dist<roi/2)
            loc[ind1] = loc1[ind1]
            r[np.where(r==0)] = 1e-10
            loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
            ind2 = np.where(np.logical_and(dist>=roi/2, dist<roi))
            loc[ind2] = loc2[ind2]
        else:
            raise ValueError('unknown localization function type: '+local_type)
    else:
        loc = np.ones(dist.shape)
    return loc


