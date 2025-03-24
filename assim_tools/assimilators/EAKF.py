import numpy as np
from utils.njit import njit
from .serial import SerialAssimilator

class EAKFAssimilator(SerialAssimilator):
    @classmethod
    def obs_increment(cls, *args, **kwargs):
        return cls._obs_increment(*args, **kwargs)

    @staticmethod
    @njit
    def _obs_increment(obs_prior, obs, obs_err):
        """
        Compute analysis increment in observation space

        Inputs:
        - obs_prior: np.array[nens]
        The observation prior ensemble

        - obs: float
        The real observation

        - obs_err: float
        The observation error standard deviation

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

        """ensemble adjustment Kalman filter (Anderson 2003)"""
        var_ratio = obs_var / (obs_prior_var + obs_var)

        ##new mean is weighted average between obs_prior_mean and obs
        obs_post_mean = var_ratio * obs_prior_mean + (1 - var_ratio) * obs

        ##new pert is adjusted by sqrt(var_ratio), a deterministic square-root filter
        obs_post_pert = np.sqrt(var_ratio) * obs_prior_pert

        ##assemble the increments
        obs_incr = obs_post_mean + obs_post_pert - obs_prior

        return obs_incr

    @classmethod
    def update_local_state(cls, *args, **kwargs):
        return cls._update_local_state(*args, **kwargs)

    @staticmethod
    @njit
    def _update_local_state(state_data, obs_prior, obs_incr,
                            h_dist, v_dist, t_dist,
                            hroi, vroi, troi,
                            h_local_func, v_local_func, t_local_func):

        nens, nfld, nloc = state_data.shape

        h_lfactor = h_local_func(h_dist, hroi)
        v_lfactor = v_local_func(v_dist, vroi)
        t_lfactor = t_local_func(t_dist, troi)

        nloc_sub = np.where(h_lfactor>0)[0]  ##subset of range(nloc) to update

        lfactor = np.zeros((nfld, nloc))
        for l in nloc_sub:
            for n in range(nfld):
                lfactor[n, l] = h_lfactor[l] * v_lfactor[n, l] * t_lfactor[n]

        state_data[:, :, nloc_sub] = update_ensemble(state_data[:, :, nloc_sub], obs_prior, obs_incr, lfactor[:, nloc_sub])

    @classmethod
    def update_local_obs(cls, *args, **kwargs):
        return cls._update_local_obs(*args, **kwargs)

    @staticmethod
    @njit
    def _update_local_obs(obs_data, used, obs_prior, obs_incr,
                          h_dist, v_dist, t_dist,
                          hroi, vroi, troi,
                          h_local_func, v_local_func, t_local_func):

        ##distance between local obs_data and the obs being assimilated
        h_lfactor = h_local_func(h_dist, hroi)
        v_lfactor = v_local_func(v_dist, vroi)
        t_lfactor = t_local_func(t_dist, troi)

        lfactor = h_lfactor * v_lfactor * t_lfactor

        ##update the unused obs within roi
        ind = np.where(np.logical_and(~used, lfactor>0))[0]

        obs_data[:, ind] = update_ensemble(obs_data[:, ind], obs_prior, obs_incr, lfactor[ind])

@njit
def update_ensemble(ens_prior, obs_prior, obs_incr, local_factor):
    """
    Update the ensemble variable using the obs increments

    Inputs:
    - ens_prior: np.array[nens, ...]
    The prior ensemble variables to be updated to posterior, dimension 0 is ensemble members

    - obs_prior: np.array[nens]
    Observation prior ensemble

    - obs_incr: np.array[nens]
    Observation space analysis increment

    - local_factor: float
    The localization factor to reduce spurious correlation in regression

    Output:
    - ens_post: np.array[nens, ...]
    Updated ensemble
    """
    nens = ens_prior.shape[0]
    ens_post = ens_prior.copy()

    ##obs-space statistics
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_var = np.sum((obs_prior - obs_prior_mean)**2) / (nens-1)

    ##if there is no prior spread, don't update at all
    if obs_prior_var == 0:
        return ens_post

    cov = np.zeros(ens_prior.shape[1:])
    for m in range(nens):
        cov += ens_prior[m, ...] * (obs_prior[m] - obs_prior_mean) / (nens-1)

    reg_factor = cov / obs_prior_var

    ##the updated posterior ensemble
    for m in range(nens):
        ens_post[m, ...] = ens_prior[m, ...] + local_factor * reg_factor * obs_incr[m]

    return ens_post
