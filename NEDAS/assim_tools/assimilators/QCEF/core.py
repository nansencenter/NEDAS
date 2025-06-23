import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator

class QCEFAssimilator(SerialAssimilator):
    def obs_increment(self, obs_prior, obs, obs_err):
        return obs_increment_qcef(obs_prior, obs, obs_err)

    def update_local_state(self, state_prior, obs_prior, obs_incr,
                        state_h_dist, state_v_dist, state_t_dist,
                        hroi, vroi, troi,
                        h_local_func, v_local_func, t_local_func) -> None:
        return update_local_state_linear(state_prior, obs_prior, obs_incr,
                                         state_h_dist, state_v_dist, state_t_dist,
                                         hroi, vroi, troi,
                                         h_local_func, v_local_func, t_local_func)

    def update_local_obs(self, obs_data, used, obs_prior, obs_incr,
                         h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func) -> None:
        return update_local_obs_linear(obs_data, used, obs_prior, obs_incr,
                                       h_dist, v_dist, t_dist,
                                       hroi, vroi, troi,
                                       h_local_func, v_local_func, t_local_func)

    def transform_ens_state_forward(self, state_data):
        #here the implementation of probit transform_to_probit for all state variables
        pass

    def transform_ens_state_backward(self, state_data):
        pass

    def transform_ens_obs_forward(self, obs_data):
        #here the implementation of probit transform_to_probit for all obs variables
        pass

    def transform_ens_obs_backward(self, obs_data):
        pass

def transform_to_probit():
    pass

def transform_from_probit():
    pass

@njit
def obs_increment_qcef(obs_prior, obs, obs_err) -> np.ndarray:
    pass

@njit
def update_local_state_linear(state_data, obs_prior, obs_incr,
                              h_dist, v_dist, t_dist,
                              hroi, vroi, troi,
                              h_local_func, v_local_func, t_local_func) -> None:

    nens, nfld, nloc = state_data.shape

    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    nloc_sub = np.where(h_lfactor>0)[0]  ##subset of range(nloc) to update

    ##TODO: impact_on_state missing
    lfactor = np.zeros((nfld, nloc))
    for l in nloc_sub:
        for n in range(nfld):
            lfactor[n, l] = h_lfactor[l] * v_lfactor[n, l] * t_lfactor[n]

    state_data[:, :, nloc_sub] = update_ensemble(state_data[:, :, nloc_sub], obs_prior, obs_incr, lfactor[:, nloc_sub])

@njit
def update_local_obs_linear(obs_data, used, obs_prior, obs_incr,
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
def update_ensemble(ens_prior, obs_prior, obs_incr, local_factor) -> np.ndarray:
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
