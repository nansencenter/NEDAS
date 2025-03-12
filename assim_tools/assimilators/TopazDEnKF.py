import numpy as np
from utils.njit import njit
from ..localization import local_factor_distance_based
from .batch import BatchAssimilator

class TopazDEnKFAssimilator(BatchAssimilator):

    @classmethod
    def local_analysis(cls, *args, **kwargs):
        return cls._local_analysis(*args, **kwargs)

    @staticmethod
    @njit(cache=True)
    def _local_analysis(state_prior, obs_prior, obs, obs_err, hlfactor,
                    state_z, obs_z, vroi, localize_vtype,
                    state_t, obs_t, troi, localize_ttype,
                    impact_on_state, filter_type,
                    rfactor=1., kfactor=1., nlobs_max=None) ->None:
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
            # nlobs_max = 3000
            ind = ind[:nlobs_max]

            ##use cached weight if no localization is applied, to avoid repeated computation
            if n>0 and len(ind)==len(lfactor_old) and (lfactor[ind]==lfactor_old).all():
                weights = weights_old
            else:
                weights = ensemble_transform_weights(obs[ind], obs_err[ind], obs_prior[:, ind], filter_type, lfactor[ind], rfactor, kfactor)

            ##perform local analysis and update the ensemble state
            state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)

            lfactor_old = lfactor[ind]
            weights_old = weights

@njit(cache=True)
def ensemble_transform_weights(obs, obs_err, obs_prior, filter_type, local_factor, rfactor, kfactor):
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
    
    - rfactor: float

    - kfactor: float

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
    ##find variance of obs_prior
    obs_prior_var = np.zeros(nlobs)
    for m in range(nens):
        obs_prior_var += (obs_prior[m, :] - obs_prior_mean)**2
    obs_prior_var /= nens-1
    
    innov = obs - obs_prior_mean
    obs_var = obs_err**2

    ##inflate obs error by rfactor, and kfactor where innovation is large
    obs_var = np.sqrt((obs_prior_var + obs_var)**2 + obs_prior_var*(innov/kfactor)**2) - obs_prior_var
    obs_err = np.sqrt(obs_var)

    ##obs_prior_pert S and innovation dy, normalized by sqrt(nens-1) R^-0.2
    S = np.zeros((nlobs, nens))
    dy = np.zeros((nlobs))
    for p in range(nlobs):
        S[p, :] = (obs_prior[:, p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
        dy[p] = (obs[p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
    S /= np.sqrt(nens-1)
    dy /= np.sqrt(nens-1)

    ##----first part of weights: update of mean
    ##find singular values of the inverse variance ratio matrix (I + S^T S)
    ##note: the added I actually helps prevent issues if S^T S is not full rank
    ##      when nlobs<nens, there will be singular values of 0, but the full matrix
    ##      can still be inverted with singular values of 1.
    if nlobs >= nens:   
        var_ratio_inv = np.eye(nens) + S.T @ S

        try:
            L = np.linalg.cholesky(var_ratio_inv)
            L_inv = np.linalg.inv(L)
        except:
            ##if svd failed just return equal weights (no update)
            print('Error: failed to invert var_ratio_inv=', var_ratio_inv)
            return np.eye(nens)
        
    else:
        var_ratio_inv = S @ S.T + np.eye(nlobs)

        try:
            L = np.linalg.cholesky(var_ratio_inv)
            L_inv = np.linalg.inv(L)
        except:
            ##if svd failed just return equal weights (no update)
            print('Error: failed to invert var_ratio_inv=', var_ratio_inv)
            return np.eye(nlobs)

    ##the update of ens mean is given by (I + S^T S)^-1 S^T dy
    ##namely, var_ratio * obs_prior_var / obs_var * dy = G dy
    var_ratio = L_inv.T @ L_inv
    
    ##the gain matrix
    gain = var_ratio @ S.T

    for m in range(nens):
        weights[m, :] = np.sum(gain[m, :] * dy)

    ##---second part of weights: update of ensemble spread
    if rfactor > 1:  ##scale S with rfactor and recompute the var_ratio
        S /= np.sqrt(rfactor)
        try:
            var_ratio_inv = np.eye(nens) + S.T @ S
            L = np.linalg.cholesky(var_ratio_inv)
            L_inv = np.linalg.inv(L)
            var_ratio = L_inv.T @ L_inv
            gain = var_ratio @ S.T
        except:
            ##if failed just return equal weights (no update)
            return np.eye(nens)

    ##DEnKF: take Taylor approx. of var_ratio_sqrt (Sakov 2008)
    var_ratio_sqrt = np.eye(nens) - 0.5 * gain @ S

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
