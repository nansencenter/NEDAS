import numpy as np
from numba import njit

##batch assimilation solves the matrix version EnKF analysis for a given local state
##the local_analysis updates for different variables are computed in parallel
@njit
def local_analysis(ens_prior,          ##ensemble state [nens]
                   obs,                ##obs [nlobs]
                   obs_err,            ##obs err std [nlobs]
                   obs_prior,          ##ensemble obs values [nens, nlobs]
                   local_factor,       ##localization factor [nlobs]
                   filter_type='ETKF', ##type of filter algorithm to apply
                   obs_err_corr=None,  ##obs err corr matrix, if None, uncorrelated
                   ):
    ##update the local state variable ens_prior with the obs

    nens, nlobs = obs_prior.shape
    assert nens == ens_prior.size, 'ens_prior[{}] size mismatch with obs_prior[{},:]'.format(ens_prior.size, nens)

    ##don't allow update of ensemble with any missing values
    if any(np.isnan(ens_prior)):
        return ens_prior

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
        S[p, :] = (obs_prior[:, p] - obs_prior_mean[p]) * local_factor[p] / obs_err
        dy[p] = (obs[p] - obs_prior_mean[p]) * local_factor[p] / obs_err

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
        raise ValueError('unknown filter type '+filter_type+' for local analysis')

    weight += var_ratio_sqrt

    ##check if weights sum to 1
    for m in range(nens):
        sum_wgts = np.sum(weight[:, m])
        if np.abs(sum_wgts - 1) > 1e-5:
            raise ValueError('sum of weights is not 1')

    ##finally, transform the prior ensemble with the weight matrix
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weight[:, m])

    return ens_post


##serial assimilation goes through the list of observations one by one
##for each obs the near by state variables are updated one by one.
##so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
@njit
def obs_increment(obs_prior,          ##obs prior [nens]
                  obs,                ##obs, scalar
                  obs_err,            ##obs err std, scalar
                  filter_type='EAKF', ##kind of filter algorithm
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
        raise ValueError('unknown filter_type: '+filter_type)

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

    elif reg_kind == 'probit':
        pass

    else:
        raise ValueError('unknown regression kind: '+reg_kind)

    return ens_post


##localization factor based on distance and roi
@njit
def local_factor(dist, roi, local_type='GC'):
    ## dist: input distance, ndarray
    ## roi: radius of influence, distance beyond which loc=0
    ## returns the localization factor loc
    dist = np.atleast_1d(dist)
    lfactor = np.zeros(dist.shape)

    if roi>0:
        if local_type == 'GC': ##Gaspari-Cohn function (default)
            r = dist / (roi / 2)
            loc1 = (((-0.25*r + 0.5)*r + 0.625)*r - 5.0/3.0) * r**2 + 1
            ind1 = np.where(dist<roi/2)
            lfactor[ind1] = loc1[ind1]
            r[np.where(r==0)] = 1e-10
            loc2 = ((((r/12.0 - 0.5)*r + 0.625)*r + 5.0/3.0)*r - 5.0)*r + 4 - 2.0/(3.0*r)
            ind2 = np.where(np.logical_and(dist>=roi/2, dist<roi))
            lfactor[ind2] = loc2[ind2]

        elif local_type == 'step':  #step function from 1 to 0 at roi
            ind1 = np.where(dist<=roi)
            lfactor[ind1] = 1.0
            ind2 = np.where(dist>roi)
            lfactor[ind2] = 0.0

        else:
            raise ValueError('unknown localization function type: '+local_type)
    else:
        ##no localization, all ones
        lfactor = np.ones(dist.shape)

    return lfactor


