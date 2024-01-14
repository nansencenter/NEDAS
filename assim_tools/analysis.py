import numpy as np
from numba import njit
import config as c
from log import message, show_progress
from conversion import t2h, h2t
from parallel import bcast_by_root

def batch_assim(c, state_info, obs_info, obs_inds, partitions, par_list, rec_list, state_prior, z_state, lobs, lobs_prior):
    """
    batch assimilation solves the matrix version EnKF analysis for a given local state
    the local_analysis updates for different variables are computed in parallel
    """
    message(c.comm, 'assimilate in batch mode\n', c.pid_show)

    state_post = state_prior.copy() ##save a copy for posterior states

    ##pid with the most obs in its task list with show progress message
    obs_count = np.array([np.sum([len(obs_inds[r][p])
                                  for r in obs_info['records'].keys()
                                  for p in lst])
                          for lst in par_list.values()])
    c.pid_show = np.argsort(obs_count)[-1]

    ##count number of tasks
    ntask = 0
    for par_id in par_list[c.pid_mem]:
        ist,ied,di,jst,jed,dj = partitions[par_id]
        ii, jj = np.meshgrid(np.arange(ist,ied,di), np.arange(jst,jed,dj))
        mask_chk = c.mask[jst:jed:dj, ist:ied:di]
        for l in range(len(ii[~mask_chk])):
            ntask += 1

    ##now the actual work starts, loop through tiles stored on pid
    task = 0
    for par_id in par_list[c.pid_mem]:
        ist,ied,di,jst,jed,dj = partitions[par_id]
        ii, jj = np.meshgrid(np.arange(ist,ied,di), np.arange(jst,jed,dj))
        mask_chk = c.mask[jst:jed:dj, ist:ied:di]

        ##fetch local obs seq and obs_prior seq on par_id
        nlobs = np.sum([len(lobs[r][par_id]['obs'].flatten()) for r in obs_info['records'].keys()])
        if nlobs == 0:
            task += len(ii[~mask_chk])
            show_progress(c.comm, task, ntask, c.pid_show)
            continue

        obs = np.full(nlobs, np.nan)
        obs_name = np.full(nlobs, np.nan)
        obs_x = np.full(nlobs, np.nan)
        obs_y = np.full(nlobs, np.nan)
        obs_z = np.full(nlobs, np.nan)
        obs_t = np.full(nlobs, np.nan)
        obs_err = np.full(nlobs, np.nan)
        hroi = np.ones(nlobs)
        vroi = np.ones(nlobs)
        troi = np.ones(nlobs)
        obs_prior = np.full((c.nens, nlobs), np.nan)
        i = 0
        for r, obs_rec in obs_info['records'].items():
            ##TODO: if obs is vector???
            d = len(lobs[r][par_id]['obs'].flatten())
            obs[i:i+d] = lobs[r][par_id]['obs'].flatten()
            obs_x[i:i+d] = lobs[r][par_id]['x'].flatten()
            obs_y[i:i+d] = lobs[r][par_id]['y'].flatten()
            obs_z[i:i+d] = lobs[r][par_id]['z'].flatten()
            obs_t[i:i+d] = np.array([t2h(t) for t in lobs[r][par_id]['t'].flatten()])
            # obs_name[i:i+d] = np.array([obs_info['records'][r]['name'] for 
            obs_err[i:i+d] = np.ones(d) * obs_info['records'][r]['err']['std']
            hroi[i:i+d] = np.ones(d) * obs_info['records'][r]['hroi']
            vroi[i:i+d] = np.ones(d) * obs_info['records'][r]['vroi']
            troi[i:i+d] = np.ones(d) * obs_info['records'][r]['troi']
            for m in range(c.nens):
                obs_prior[m, i:i+d] = lobs_prior[m,r][par_id].flatten()
            i += d

        ##loop through unmasked grid points in the tile
        for l in range(len(ii[~mask_chk])):

            state_x = c.grid.x[0, ii[~mask_chk][l]]
            state_y = c.grid.y[jj[~mask_chk][l], 0]
            hdist = np.hypot(obs_x-state_x, obs_y-state_y)
            lfactor = local_factor(hdist, hroi, c.localize_type)
            if (lfactor==0).all():
                task += 1
                show_progress(c.comm, task, ntask, c.pid_show)
                continue

            # inds = np.where(lfactor>0)[0] #[0:3000]
            # print(lfactor(inds))
            ##TODO: keep only the first 3000 obs with most impact
            ##TODO: lfactor is maybe wrong
            # obs_prior_sub = np.zeros((c.nens, len(inds)))
            # for m in range(c.nens):
            #     obs_prior_sub[m, :] = obs_prior[m, :][inds]
            weight = local_analysis(obs, obs_err, None, obs_prior, c.filter_type, lfactor)

            ##loop through each field rec_id on pid_rec
            for rec_id in rec_list[c.pid_rec]:
                rec = state_info['fields'][rec_id]

                keys = [(0, l), (1, l)] if rec['is_vector'] else [l]
                for key in keys:
                    ens_prior = np.array([state_prior[m, rec_id][par_id][key] for m in range(c.nens)])

                    ##localization factor
                    # state_z = z_state[m, rec_id][par_id][key]
                    # state_t = t2h(state_info['fields'][rec_id]['time'])
                    # vdist = np.abs(obs_z-state_z)
                    # tdist = np.abs(obs_t-state_t)
                    #* local_factor(vdist, vroi, c.localize_type) * local_factor(tdist, troi, c.localize_type)

                    ##save the posterior ensemble to the state
                    for m in range(c.nens):
                        state_post[m, rec_id][par_id][key] = np.sum(ens_prior * weight[:, m])

            task += 1
            show_progress(c.comm, task, ntask, c.pid_show)

    message(c.comm, ' done.\n', c.pid_show)

    return state_post


def form_local_obs():
    return obs, obs_prior


@njit
def local_analysis(obs, obs_err, obs_err_corr, obs_prior, filter_type, local_factor):
    """
    Local analysis for batch assimilation mode

    Inputs:
    - ens_prior: np.array[nens]
      The prior ensemble state variables

    - obs: np.array[nlobs]
      The local observation sequence

    - obs_err: np.array[nlobs]
      The observation error standard deviations

    - obs_err_corr: np.array
      If None, the observation errors are uncorrelated (np.eye(nlobs) will be used here)

    - obs_prior: np.array[nens, nlobs]
      The observation priors

    - filter_type: str
      Type of filter to use: "ETKF", or "DEnKF"

    - local_factor: np.array[nlobs]
      Localization/impact factor for each observation

    Return:
    - weights: np.array[nens, nens]
      The transform weights

    - ens_post: np.array[nens]
      The posterior ensmble state variables
    """
    ##update the local state variable ens_prior with the obs
    nens, nlobs = obs_prior.shape

    ##ensemble weight matrix, weight[:, m] is for the m-th member
    ##also known as T in Bishop 2001, and X5 in Evensen textbook (and in Sakov 2012)
    weight = np.zeros((nens, nens))

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
    L, sv, Rh = np.linalg.svd(var_ratio_inv)

    ##the update of ens mean is given by (I + S^T S)^-1 S^T dy
    ##namely, var_ratio * obs_prior_var / obs_var * dy = G dy
    var_ratio = L @ np.diag(sv**-1) @ Rh

    ##the gain matrix
    gain = var_ratio @ S.T

    for m in range(nens):
        weight[m, :] = np.sum(gain[m, :] * dy)

    if filter_type == 'ETKF':
        ##the update of ens pert is (I + S^T S)^-0.5, namely sqrt(var_ratio)
        var_ratio_sqrt = L @ np.diag(sv**-0.5) @ Rh

    elif filter_type == 'DEnKF':
        ##take Taylor approx. of var_ratio_sqrt (Sakov 2008)
        var_ratio_sqrt = np.eye(nens) - 0.5 * gain @ S

    else:
        print('Error: unknown filter type: '+filter_type)
        raise ValueError

    weight += var_ratio_sqrt

    return weight


@njit
def ensemble_transform(ens_prior, weight):
    """transform the prior ensemble with the weight matrix"""
    nens = ens_prior.size

    ##check if weights sum to 1
    for m in range(nens):
        sum_wgts = np.sum(weight[:, m])
        if np.abs(sum_wgts - 1) > 1e-5:
            print('Warning: sum of weights != 1 detected!')

    ##apply the weight
    ens_post = ens_prior.copy()
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weight[:, m])

    return ens_post


def serial_assim(c, state_info, obs_info, obs_inds, partitions, par_list, rec_list, state_prior, z_state, lobs, lobs_prior):
    """
    serial assimilation goes through the list of observations one by one
    for each obs the near by state variables are updated one by one.
    so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
    """
    message(c.comm, 'assimilate in serial mode\n', c.pid_show)

    ##x,y coordinates for local state variables on pid
    ist,ied,di,jst,jed,dj = partitions[c.pid_mem]
    ii, jj = np.meshgrid(np.arange(ist,ied,di), np.arange(jst,jed,dj))
    mask_chk = c.mask[jst:jed:dj, ist:ied:di]
    x = c.grid.x[jst:jed:dj, ist:ied:di][~mask_chk]
    y = c.grid.y[jst:jed:dj, ist:ied:di][~mask_chk]

    obs_list = form_obs_list(obs_info, obs_inds)

    ##go through the entire obs list, indexed by p, one scalar obs at a time
    for p in range(len(obs_list)):

        obs_rec_id, obs_id, v = obs_list[p]
        obs_rec = obs_info['records'][obs_rec_id]

        ##figure out with pid owns the obs_id
        pid_owner_obs = [p for p,lst in obs_inds[obs_rec_id].items() if obs_id in lst][0]

        ##1. if the pid owns this obs, compute obs_incr and broadcast
        if c.pid_mem == pid_owner_obs:
            ##local obs index in lobs seq
            i = np.where(obs_inds[obs_rec_id][pid_owner_obs]==obs_id)[0]

            ##collect obs info
            obs = {}
            obs['obs'] = lobs[obs_rec_id][pid_owner_obs]['obs'][v, i].item()
            obs['x'] = lobs[obs_rec_id][pid_owner_obs]['x'][i].item()
            obs['y'] = lobs[obs_rec_id][pid_owner_obs]['y'][i].item()
            obs['err_std'] = lobs[obs_rec_id][pid_owner_obs]['err_std'][i].item()
            obs_prior = np.array([lobs_prior[m, obs_rec_id][c.pid_mem][v, i].item() for m in range(c.nens)])

            ##compute obs-space increment
            obs_incr = obs_increment(obs_prior, obs['obs'], obs['err_std'], c.filter_type)

        # else:
        #     obs = None
        #     obs_prior = None
        #     obs_incr = None
        # obs = c.comm_mem.bcast(obs, root=pid_owner_obs)
        # obs_prior = c.comm_mem.bcast(obs_prior, root=pid_owner_obs)
        # obs_incr = c.comm_mem.bcast(obs_incr, root=pid_owner_obs)

        ##2. all pid update their own locally stored state/obs:
        ##
        ##distance between local state variable and the obs
        # hdist = np.hypot(x-obs['x'], y-obs['y'])
        # lfactor = local_factor(hdist, obs_rec['hroi'])

        ##go through location within roi (lfactor>0)
        # for l in range(len(x)):
        #     if lfactor[l] == 0:
        #         continue

        #     ##loop through each field rec_id on pid_rec
        #     for rec_id in rec_list[c.pid_rec]:
        #         rec = state_info['fields'][rec_id]

        #         keys = [(0, l), (1, l)] if rec['is_vector'] else [l]
        #         for key in keys:
        #             ##collect members from state_prior
        #             ens_prior = np.array([state_prior[m, rec_id][c.pid_mem][key] for m in range(c.nens)])

        #             ##compute the update
        #             ens_post = update_ensemble(ens_prior, obs_prior, obs_incr, lfactor[l], c.regress_type)

        #             ##save to the state_prior (iteratively updated by each obs, by the end becomes the posterior)
        #             for m in range(c.nens):
        #                 state_prior[m, rec_id][c.pid_mem][key] = ens_post[m]

        ##update locally stored obs, indexed by q
        ##only need to update the unused obs
        # for q in range(p+1, len(obs_list)):
        #     l_obs_rec_id, l_obs_id, l_v = obs_list[q]
        #     ##distance between local obs and the obs

        #     if :
        #         continue

        show_progress(c.comm, p, len(obs_list), c.pid_show)

    message(c.comm, ' done.\n', c.pid_show)

    return state_prior


@bcast_by_root(c.comm)
def form_obs_list(obs_info, obs_inds):
    ##count number of obs for each obs_rec_id
    nobs = np.array([np.sum([len(ind) for ind in obs_inds[r].values()])
                     for r in obs_info['records'].keys()])

    ##form the full list of obs_ids
    obs_list = []
    for obs_rec_id in obs_info['records'].keys():
        for obs_id in range(nobs[obs_rec_id]):
            v_list = [0, 1] if obs_info['records'][obs_rec_id]['is_vector'] else [None]
            for v in v_list:
                obs_list.append((obs_rec_id, obs_id, v))

    ##randomize the order of obs (this is optional)
    np.random.shuffle(obs_list)

    return obs_list


@njit
def obs_increment(obs_prior, obs, obs_err, filter_type):
    """
    Compute analysis increment in observation space

    Inputs:
    - obs_prior: np.array[nens]
      The observation prior ensemble

    - obs: float
      The real observation

    - obs_err: float
      The observation error standard deviation

    - filter_type: str
      Type of filtering to apply: "EAKF" (default), "RHF", etc.

    Return:
    - obs_incr: np.array[nens]
      Analysis increments in observation space
    """
    nens = obs_prior.size

    ##obs error variance
    obs_var = obs_err**2

    ##obs_prior separate into mean+perturbation
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_pert = obs_prior - obs_prior_mean

    ##compute prior error variance
    obs_prior_var = np.sum(obs_prior_pert**2) / (nens-1)

    ##ensemble adjustment Kalman filter (Anderson 2003)
    if filter_type == 'EAKF':
        var_ratio = obs_var / (obs_prior_var + obs_var)

        ##new mean is weighted average between obs_prior_mean and obs
        obs_post_mean = var_ratio * obs_prior_mean + (1 - var_ratio) * obs

        ##new pert is adjusted by sqrt(var_ratio), a deterministic square-root filter
        obs_post_pert = np.sqrt(var_ratio) * obs_prior_pert

        ##assemble the increments
        obs_incr = obs_post_mean + obs_post_pert - obs_prior

    elif filter_type == 'RHF':
        pass

    else:
        print('Error: unknown filter type: '+filter_type)

    return obs_incr


@njit
def update_ensemble(ens_prior, obs_prior, obs_incr, local_factor, regress_type):
    """
    Update the ensemble variable using the obs increments

    Inputs:
    - ens_prior: np.array[nens]
      The prior ensemble variables to be updated

    - obs_prior: np.array[nens]
      Observation prior ensemble

    - obs_incr: np.array[nens]
      Observation space analysis increment

    - local_factor: float
      The localization factor to reduce spurious correlation in regression

    - regress_kind: str
      Type of regression to perform, 'linear', 'probit', etc.

    Return:
    - ens_post: np.array[nens]
      The posterior ensemble values after update
    """
    nens = ens_prior.size

    ens_post = ens_prior.copy()

    ##don't allow update of ensemble with any missing values

    ##obs_prior separate into mean+perturbation
    ###put some of these in input, to avoid repeated calculation
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_pert = obs_prior - obs_prior_mean

    if regress_type == 'linear':
        ##linear regression relates the obs_prior with ens_prior
        obs_prior_var = np.sum(obs_prior_pert**2) / (nens-1)
        cov = np.sum(ens_prior * obs_prior_pert) / (nens-1)
        reg_factor = cov / obs_prior_var

        ##the updated posterior ensemble
        ens_post = ens_prior + local_factor * reg_factor * obs_incr

    # elif regress_type == 'probit':
    #     pass


    return ens_post


@njit
def local_factor(dist, roi, localize_type='GC'):
    """
    Localization factor based on distance and radius of influence (roi)

    Inputs:
    - dist: np.array
      Distance between observation and state (being updated)

    - roi: float or np.array same shape as dist
      The radius of influence, distance beyond which local_factor is tapered to 0

    Return:
    - lfactor: np.array
      The localization factor, same shape as dist
    """
    dist = np.atleast_1d(dist)
    lfactor = np.zeros(dist.shape)

    if localize_type == 'GC': ##Gaspari-Cohn function (default)
        r = dist / (roi / 2)

        ind1 = np.where(r<1)
        loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
        lfactor[ind1] = loc1[ind1]

        ind2 = np.where(np.logical_and(r>=1, r<2))
        r[np.where(r==0)] = 1e-10  ##avoid divide by 0
        loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
        lfactor[ind2] = loc2[ind2]

    elif localize_type == 'step':  #step function from 1 to 0 at roi
        lfactor[np.where(dist<=roi)] = 1.0

    elif localize_type == 'exp':  ##exponential decay
        lfactor = np.exp(-dist/roi)

    else:
        print('Error: unknown localization function type: '+localize_type)
        raise ValueError

    return lfactor


