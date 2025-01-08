"""core assimilation algorithms in the analysis step"""

import numpy as np
from utils.parallel import by_rank, bcast_by_root
from utils.njit import njit
from utils.progress import print_with_cache, progress_bar
from .packing import pack_state_data, unpack_state_data, pack_obs_data, unpack_obs_data
from .covariance import ensemble_covariance
from .localization import local_factor_distance_based

def batch_assim(c, state_prior, z_state, lobs, lobs_prior):
    """Batch assimilation solves the matrix version EnKF analysis for each local state,
    the local states in each partition are processed in parallel
    Parameters:
    - c: config object
    - state_prior: np.array[nens, nfld, nloc]
    - z_state: np.array[nfld, nloc]
    - lobs: list of obs objects
    - lobs_prior: list of obs objects
    Returns:
    - state_post: np.array[nens, nfld, nloc], updated version of state_prior
    - lobs_post: list of obs objects, updated version of lobs_prior
    """
    ##pid with the most obs in its task list with show progress message
    obs_count = np.array([np.sum([len(c.obs_inds[r][p])
                                  for r in c.obs_info['records'].keys()
                                  for p in lst])
                          for lst in c.par_list.values()])
    c.pid_show = np.argsort(obs_count)[-1]
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)

    ##count number of tasks
    ntask = 0
    for par_id in c.par_list[c.pid_mem]:
        if len(c.grid.x.shape)==2:
            ist,ied,di,jst,jed,dj = c.partitions[par_id]
            msk = c.mask[jst:jed:dj, ist:ied:di]
        else:
            inds = c.partitions[par_id]
            msk = c.mask[inds]
        for loc_id in range(np.sum((~msk).astype(int))):
            ntask += 1

    ##now the actual work starts, loop through partitions stored on pid_mem
    print_1p('>>> assimilate in batch mode:\n')
    task = 0
    for par_id in c.par_list[c.pid_mem]:
        state_data = pack_state_data(c, par_id, state_prior, z_state)
        nloc = state_data['state_prior'].shape[-1]
        ##skip forward if the partition is empty
        if nloc == 0:
            continue

        obs_data = pack_obs_data(c, par_id, lobs, lobs_prior)
        nlobs = obs_data['x'].size
        ##if there is no obs to assimilate, update progress message and skip that partition
        if nlobs == 0:
            task += nloc
            if c.debug:
                print(f"PID {c.pid:4} processed partition {par_id:7} (which is empty)", flush=True)
            else:
                print_1p(progress_bar(task-1, ntask))
            continue

        ##loop through the unmasked grid points in the partition
        for loc_id in range(nloc):
            ##state variable metadata for this location
            state_var_id = state_data['var_id']  ##variable id for each field (nfld)
            state_x = state_data['x'][loc_id]
            state_y = state_data['y'][loc_id]
            state_z = state_data['z'][:, loc_id]
            state_t = state_data['t'][:]

            ##filter out obs outside the hroi in each direction first (using L1 norm to speed up)
            obs_rec_id = obs_data['obs_rec_id']
            hroi = obs_data['hroi'][obs_rec_id]
            hdist = c.grid.distance(state_x, obs_data['x'], state_y, obs_data['y'], p=1)
            ind = np.where(hdist<=hroi)[0]

            ##compute horizontal localization factor (using L2 norm for distance)
            obs_rec_id = obs_data['obs_rec_id'][ind]
            hroi = obs_data['hroi'][obs_rec_id]
            hdist = c.grid.distance(state_x, obs_data['x'][ind], state_y, obs_data['y'][ind], p=2)
            hlfactor = local_factor_distance_based(hdist, hroi, c.localization['htype'])
            ind1 = np.where(hlfactor>0)[0]
            ind = ind[ind1]
            hlfactor = hlfactor[ind1]

            if len(ind1) == 0:
                if c.debug:
                    print(f"PID {c.pid:4} processed partition {par_id:7} grid point {loc_id} (all local obs outside hroi)", flush=True)
                else:
                    print_1p(progress_bar(task, ntask))
                continue ##if all obs has no impact on state, just skip to next location
            
            ##vertical, time and cross-variable (impact_on_state) localization
            obs = obs_data['obs'][ind]
            obs_err = obs_data['err_std'][ind]
            obs_z = obs_data['z'][ind]
            obs_t = obs_data['t'][ind]
            obs_rec_id = obs_data['obs_rec_id'][ind]
            vroi = obs_data['vroi'][obs_rec_id]
            troi = obs_data['troi'][obs_rec_id]
            impact_on_state = obs_data['impact_on_state'][:, state_var_id][obs_rec_id]

            ##covariance between state and obs
            # stateV, obsV, corr = covariance(c.covariance, state_prior, obs_prior, state_var_id, obs_rec_id, h_dist, v_dist, t_dist)

            local_analysis(state_data['state_prior'][...,loc_id], obs_data['obs_prior'][:,ind],
                           obs, obs_err, hlfactor,
                           state_z, obs_z, vroi, c.localization['vtype'],
                           state_t, obs_t, troi, c.localization['ttype'],
                           impact_on_state, c.filter_type)

            ##add progress message
            if c.debug:
                print(f"PID {c.pid:4} processed partition {par_id:7} grid point {loc_id}", flush=True)
            else:
                print_1p(progress_bar(task, ntask))
            task += 1

        unpack_state_data(c, par_id, state_prior, state_data)
    print_1p(' done.\n')
    return state_prior, lobs_prior

@njit(cache=True)
def local_analysis(state_prior, obs_prior, obs, obs_err, hlfactor,
                   state_z, obs_z, vroi, localize_vtype,
                   state_t, obs_t, troi, localize_ttype,
                   impact_on_state, filter_type) ->None:
    """perform local analysis for one location in the analysis grid partition"""
    nens, nfld = state_prior.shape
    nens_obs, nlobs = obs_prior.shape
    if nens_obs != nens:
        raise ValueError('Error: number of ensemble members in state and obs do not match!')

    lfactor_old = np.zeros(nlobs)
    weights_old = np.eye(nens)

    ##loop through the field records
    for n in range(nfld):

        ##vertical localization
        vdist = np.abs(obs_z - state_z[n])
        vlfactor = local_factor_distance_based(vdist, vroi, localize_vtype)
        if (vlfactor==0).all():
            continue  ##the state is outside of vroi of all obs, skip

        ##temporal localization
        tdist = np.abs(obs_t - state_t[n])
        tlfactor = local_factor_distance_based(tdist, troi, localize_ttype)
        if (tlfactor==0).all():
            continue  ##the state is outside of troi of all obs, skip

        ##total lfactor
        lfactor =  hlfactor * vlfactor * tlfactor * impact_on_state[:, n]
        if (lfactor==0).all():
            continue

        ##if prior spread is zero, don't update
        if np.std(state_prior[:, n]) == 0:
            continue

        ##only need to assimilate obs with lfactor>0
        ind = np.where(lfactor>0)[0]

        ##TODO:get rid of obs if obs_prior is nan
        # valid = np.array([np.isnan(obs_prior[:,i]).any() for i in ind])
        # ind = ind[valid]

        ##sort the obs from high to low lfactor
        sort_ind = np.argsort(lfactor[ind])[::-1]
        ind = ind[sort_ind]

        ##limit number of local obs if needed
        ###e.g. topaz only keep the first 3000 obs with highest lfactor
        nlobs_max = 3000
        ind = ind[:nlobs_max]

        ##use cached weight if no localization is applied, to avoid repeated computation
        if n>0 and len(ind)==len(lfactor_old) and (lfactor[ind]==lfactor_old).all():
            weights = weights_old
        else:
            weights = ensemble_transform_weights(obs[ind], obs_err[ind], obs_prior[:, ind], filter_type, lfactor[ind])

        ##perform local analysis and update the ensemble state
        state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)

        lfactor_old = lfactor[ind]
        weights_old = weights

@njit(cache=True)
def ensemble_transform_weights(obs, obs_err, obs_prior, filter_type, local_factor):
    """
    Compute the transform weights for the local ensemble

    Inputs:
    - obs: np.array[nlobs]
      The local observation sequence

    - obs_err: np.array[nlobs]
      The observation error, sqrt(R) vector

    - obs_prior: np.array[nens, nlobs]
      The observation priors

    - filter_type: str
      Type of filter to use: "ETKF", or "DEnKF"

    - local_factor: np.array[nlobs]
      Localization/impact factor for each observation

    Return:
    - weights: np.array[nens, nens]
      The ensemble transform weights
    """
    nens, nlobs = obs_prior.shape

    ##ensemble weight matrix, weights[:, m] is for the m-th member
    ##also known as T in Bishop 2001, and X5 in Evensen textbook (and in Sakov 2012)
    weights = np.zeros((nens, nens))

    ##find mean of obs_prior
    obs_prior_mean = np.zeros(nlobs)
    for m in range(nens):
        obs_prior_mean += obs_prior[m, :]
    obs_prior_mean /= nens

    ##obs_prior_pert S and innovation dy, normalized by sqrt(nens-1) R^-0.2
    S = np.zeros((nlobs, nens))
    dy = np.zeros((nlobs))
    for p in range(nlobs):
        S[p, :] = (obs_prior[:, p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
        dy[p] = (obs[p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]

    ##TODO:factor in the correlated R in obs_err, need another SVD of R
    # obs_err_corr
    S /= np.sqrt(nens-1)
    dy /= np.sqrt(nens-1)

    ##find singular values of the inverse variance ratio matrix (I + S^T S)
    ##note: the added I actually helps prevent issues if S^T S is not full rank
    ##      when nlobs<nens, there will be singular values of 0, but the full matrix
    ##      can still be inverted with singular values of 1.
    var_ratio_inv = np.eye(nens) + S.T @ S

    ##TODO: var_ratio_inv = np.eye(nlobs) + S @ S.T  if nlobs<nens

    try:
        L, sv, Rh = np.linalg.svd(var_ratio_inv)
    except:
        ##if svd failed just return equal weights (no update)
        print('Error: failed to invert var_ratio_inv=', var_ratio_inv)
        return np.eye(nens)

    ##the update of ens mean is given by (I + S^T S)^-1 S^T dy
    ##namely, var_ratio * obs_prior_var / obs_var * dy = G dy
    var_ratio = L @ np.diag(sv**-1) @ Rh

    ##the gain matrix
    gain = var_ratio @ S.T

    for m in range(nens):
        weights[m, :] = np.sum(gain[m, :] * dy)

    if filter_type == 'ETKF':
        ##the update of ens pert is (I + S^T S)^-0.5, namely sqrt(var_ratio)
        var_ratio_sqrt = L @ np.diag(sv**-0.5) @ Rh

    elif filter_type == 'DEnKF':
        ##take Taylor approx. of var_ratio_sqrt (Sakov 2008)
        var_ratio_sqrt = np.eye(nens) - 0.5 * gain @ S

    else:
        print('Error: unknown filter type: '+filter_type)
        raise ValueError

    weights += var_ratio_sqrt

    return weights

@njit(cache=True)
def apply_ensemble_transform(ens_prior, weights):
    """Apply the weights to transform local ensemble"""

    nens = ens_prior.size
    ens_post = ens_prior.copy()

    ##check if weights sum to 1
    for m in range(nens):
        sum_wgts = np.sum(weights[:, m])
        if np.abs(sum_wgts - 1) > 1e-5:
            print('Warning: sum of weights != 1 detected!')

    ##apply the weights
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weights[:, m])

    return ens_post
