import numpy as np
from numba import njit
from log import message, progress_bar
from conversion import t2h, h2t

##batch assimilation solves the matrix version EnKF analysis for a given local state
##the local_analysis updates for different variables are computed in parallel
def batch_assim(c, state_info, obs_info, obs_inds, partitions, par_list, state_prior, z_state, lobs, lobs_prior):

    state_post = state_prior.copy() ##save a copy for posterior states

    ##a small dummy call to local_analysis to compile it with njit
    _ = local_analysis(np.ones(5), np.ones(10), np.ones(10), None, np.ones((5,10)), 'ETKF', np.ones(10))

    ##pid with most obs to work with will print out progress
    # obs_count = [np.sum([len(obs_inds[r][p])
    #                      for r in obs_info['records'].keys()
    #                      for p in par_list[pid] ])
    #              for pid in range(c.nproc)]
    # pid_show = obs_count.index(max(obs_count))

    ##loop through tiles stored on pid
    for p in range(len(par_list[c.pid])):
        par_id = par_list[c.pid][p]

        ist,ied,di,jst,jed,dj = partitions[par_id]
        ii, jj = np.meshgrid(np.arange(ist,ied,di), np.arange(jst,jed,dj))
        mask_chk = c.mask[jst:jed:dj, ist:ied:di]

        ##fetch local obs seq and obs_prior seq on tile
        nlobs = np.sum([len(lobs[r][par_id]['obs'].flatten()) for r in obs_info['records'].keys()])
        if nlobs == 0:
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

            ##loop through each field record
            for rec_id, rec in state_info['fields'].items():
                keys = [(0, l), (1, l)] if rec['is_vector'] else [l]

                for key in keys:
                    ens_prior = np.array([state_prior[m, rec_id][par_id][key] for m in range(c.nens)])

                    ##localization factor
                    state_x = c.grid.x[0, ii[~mask_chk][l]]
                    state_y = c.grid.y[jj[~mask_chk][l], 0]
                    state_z = z_state[m, rec_id][par_id][key]
                    state_t = t2h(state_info['fields'][rec_id]['time'])
                    hdist = np.hypot(obs_x-state_x, obs_y-state_y)
                    # vdist = np.abs(obs_z-state_z)
                    # tdist = np.abs(obs_t-state_t)
                    lfactor = local_factor(hdist, hroi, c.localize_type)
                    #* local_factor(vdist, vroi, c.localize_type) * local_factor(tdist, troi, c.localize_type)
                    if (lfactor==0).all():
                        continue
                    inds = np.where(lfactor>0)

                    obs_prior_sub = np.zeros((c.nens, len(inds[0])))
                    for m in range(c.nens):
                        obs_prior_sub[m, :] = obs_prior[m, :][inds]

                    ens_post = local_analysis(ens_prior, obs[inds], obs_err[inds], None, obs_prior_sub, c.filter_type, lfactor[inds])

                    ##save the posterior ensemble to the state
                    for m in range(c.nens):
                        state_post[m, rec_id][par_id][key] = ens_post[m]

        message(c.comm, progress_bar(p, len(par_list[c.pid])), 0)

    message(c.comm, ' done.\n', 0)

    return state_post


##serial assimilation goes through the list of observations one by one
##for each obs the near by state variables are updated one by one.
##so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
def serial_assim(c, state_info, obs_info, obs_inds, partitions, par_list, state_prior, lobs, lobs_prior):

    state_post = state_prior.copy()  ##make a copy for posterior states

    ##go through the entire obs list one at a time
    # for 

    return state_post


##core algorithms for ensemble data assimilation:
@njit
def local_analysis(ens_prior,          ##ensemble state [nens]
                   obs,                ##obs [nlobs]
                   obs_err,            ##obs err std [nlobs]
                   obs_err_corr,       ##obs err corr matrix, if None, uncorrelated
                   obs_prior,          ##ensemble obs values [nens, nlobs]
                   filter_type,        ##type of filter algorithm to apply
                   local_factor,       ##localization factor [nlobs]
                   ):
    ##update the local state variable ens_prior with the obs
    nens, nlobs = obs_prior.shape
    ens_post = ens_prior.copy()

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

    ##check if weights sum to 1
    for m in range(nens):
        sum_wgts = np.sum(weight[:, m])
        if np.abs(sum_wgts - 1) > 1e-5:
            print('Warning: sum of weights != 1 detected!')

    ##finally, transform the prior ensemble with the weight matrix
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weight[:, m])

    return ens_post


@njit
def obs_increment(obs_prior,          ##obs prior [nens]
                  obs,                ##obs, scalar
                  obs_err,            ##obs err std, scalar
                  filter_type,        ##kind of filter algorithm
                  ):
    ##compute analysis increment for 1 obs
    ##obs_prior[nens] is the prior ensemble values corresponding to this obs
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
def update_ensemble(ens_prior,             ##prior ens [nens] being updated
                    obs_prior,             ##obs prior ens [nens]
                    obs_incr,              ##obs increments [nens]
                    local_factor,          ##localization factor, scalar
                    regress_kind='linear', ##kind of regression method
                    ):
    ##update the ensemble for 1 variable using the obs increments
    ##local_factor is the localization factor to reduce spurious correlation
    ##ens_prior[nens] is the ensemble values prior to update
    nens = ens_prior.size

    ens_post = ens_prior.copy()

    ##don't allow update of ensemble with any missing values

    ##obs_prior separate into mean+perturbation
    ###put some of these in input, to avoid repeated calculation
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_pert = obs_prior - obs_prior_mean

    if reg_kind == 'linear':
        ##linear regression relates the obs_prior with ens_prior
        obs_prior_var = np.sum(obs_prior_pert**2) / (nens-1)
        cov = np.sum(ens_prior * obs_prior_pert) / (nens-1)
        reg_factor = cov / obs_prior_var

        ##the updated posterior ensemble
        ens_post = ens_prior + local_factor * reg_factor * obs_incr

    # elif reg_kind == 'probit':
    #     pass


    return ens_post


##localization factor based on distance and roi
@njit
def local_factor(dist, roi, localize_type='GC'):
    ## dist: distance
    ## roi: radius of influence, distance beyond which loc=0
    ## returns the localization factor loc
    dist = np.atleast_1d(dist)
    lfactor = np.zeros(dist.shape)

    if localize_type == 'GC': ##Gaspari-Cohn function (default)
        r = dist / (roi / 2)
        loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
        ind1 = np.where(r<0.5)
        lfactor[ind1] = loc1[ind1]
        r[np.where(r==0)] = 1e-10
        loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
        ind2 = np.where(np.logical_and(r>=0.5, r<1))
        lfactor[ind2] = loc2[ind2]

    elif localize_type == 'step':  #step function from 1 to 0 at roi
        r = dist / roi
        ind1 = np.where(r<=1)
        lfactor[ind1] = 1.0
        ind2 = np.where(r>1)
        lfactor[ind2] = 0.0

    elif localize_type == 'exp':  ##exponential decay
        r = dist / roi
        lfactor = np.exp(-r)

    else:
        print('Error: unknown localization function type: '+localize_type)
        raise ValueError

    return lfactor


