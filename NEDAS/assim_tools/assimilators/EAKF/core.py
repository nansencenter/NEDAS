import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator

class EAKFAssimilator(SerialAssimilator):
    supports_static_members = True
    supports_hybrid_perturbation = True

    def assimilation_algorithm(self, c):
        # weights of the dynamic and static covariance in P = weight_dynamic*P_d + weight_static*P_s
        # (weight_dynamic = 1-beta, weight_static = beta*static_var_scaling, see assim_tools/covariance)
        self.weight_dynamic = 1.0 - c.covariance.beta
        self.weight_static = c.covariance.beta * c.covariance.static_var_scaling
        self.hybrid_perturbation = c.covariance.hybrid_perturbation
        super().assimilation_algorithm(c)

    def obs_increment(self, obs_prior, obs_prior_static, obs, obs_err):
        if obs_prior_static.size == 0:
            return obs_increment_eakf(obs_prior, obs, obs_err)
        return obs_increment_eakf_hybrid(obs_prior, obs_prior_static, obs, obs_err,
                                         self.weight_dynamic, self.weight_static, self.hybrid_perturbation)

    def update_local_state(self, state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                           state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func,
                           impact_on_variable) -> None:
        if state_static.shape[0] == 0:
            return update_local_state_linear(state_prior, obs_prior, obs_incr,
                                             state_h_dist, state_v_dist, state_t_dist,
                                             hroi, vroi, troi,
                                             h_local_func, v_local_func, t_local_func,
                                             impact_on_variable)
        return update_local_state_hybrid(state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                                         state_h_dist, state_v_dist, state_t_dist,
                                         hroi, vroi, troi,
                                         h_local_func, v_local_func, t_local_func,
                                         impact_on_variable,
                                         self.weight_dynamic, self.weight_static, self.hybrid_perturbation)

    def update_local_obs(self, obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                         h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func,
                         impact_on_variable) -> None:
        if obs_data_static.shape[0] == 0:
            return update_local_obs_linear(obs_data, used, obs_prior, obs_incr,
                                           h_dist, v_dist, t_dist,
                                           hroi, vroi, troi,
                                           h_local_func, v_local_func, t_local_func,
                                           impact_on_variable)
        return update_local_obs_hybrid(obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                                       h_dist, v_dist, t_dist,
                                       hroi, vroi, troi,
                                       h_local_func, v_local_func, t_local_func,
                                       impact_on_variable,
                                       self.weight_dynamic, self.weight_static, self.hybrid_perturbation)

@njit
def obs_increment_eakf(obs_prior, obs, obs_err) -> np.ndarray:
    nens = obs_prior.size

    # obs error variance
    obs_var = obs_err**2

    # obs_prior separate into mean+perturbation
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_pert = obs_prior - obs_prior_mean

    # compute prior error variance
    obs_prior_var = np.sum(obs_prior_pert**2) / (nens-1)

    """ensemble adjustment Kalman filter (Anderson 2003)"""
    var_ratio = obs_var / (obs_prior_var + obs_var)

    # new mean is weighted average between obs_prior_mean and obs
    obs_post_mean = var_ratio * obs_prior_mean + (1 - var_ratio) * obs

    # new pert is adjusted by sqrt(var_ratio), a deterministic square-root filter
    obs_post_pert = np.sqrt(var_ratio) * obs_prior_pert

    # assemble the increments
    obs_incr = obs_post_mean + obs_post_pert - obs_prior

    return obs_incr

@njit
def obs_increment_eakf_hybrid(obs_prior, obs_prior_static, obs, obs_err,
                              weight_dynamic, weight_static, hybrid_perturbation) -> np.ndarray:
    """
    EAKF obs-space increments of the dynamic members (obs_prior) with the hybrid covariance
    P = weight_dynamic*P_d + weight_static*P_s, P_s from the static members (obs_prior_static).
    The mean moves with the hybrid prior variance; the perturbations contract with the hybrid variance
    (hybrid_perturbation=True, the Whitaker and Hamill 2002 square root with the hybrid gain, Counillon
    et al. 2009) or with the dynamic ensemble variance alone (False, as in Wang et al. 2007).
    The mean of the returned increments is the mean increment, the rest the perturbation increments.
    """
    nens = obs_prior.size
    nens_static = obs_prior_static.size
    obs_var = obs_err**2

    obs_prior_mean = np.mean(obs_prior)
    obs_prior_pert = obs_prior - obs_prior_mean
    obs_prior_var_dynamic = np.sum(obs_prior_pert**2) / max(nens - 1, 1)

    # hybrid prior variance
    obs_prior_var = weight_dynamic * obs_prior_var_dynamic
    if nens_static > 1:
        obs_prior_static_pert = obs_prior_static - np.mean(obs_prior_static)
        obs_prior_var += weight_static * np.sum(obs_prior_static_pert**2) / (nens_static - 1)

    var_ratio = obs_var / (obs_prior_var + obs_var)
    obs_post_mean = var_ratio * obs_prior_mean + (1 - var_ratio) * obs

    if not hybrid_perturbation:
        var_ratio = obs_var / (obs_prior_var_dynamic + obs_var)
    obs_post_pert = np.sqrt(var_ratio) * obs_prior_pert

    return obs_post_mean + obs_post_pert - obs_prior

@njit
def update_local_state_linear(state_data, obs_prior, obs_incr,
                              h_dist, v_dist, t_dist,
                              hroi, vroi, troi,
                              h_local_func, v_local_func, t_local_func,
                              impact_on_variable) -> None:

    nens, nfld, nloc = state_data.shape

    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    nloc_sub = np.where(h_lfactor>0)[0]  # subset of range(nloc) to update

    lfactor = np.zeros((nfld, nloc))
    for l in nloc_sub:
        for n in range(nfld):
            lfactor[n, l] = h_lfactor[l] * v_lfactor[n, l] * t_lfactor[n] * impact_on_variable[n]

    state_data[:, :, nloc_sub] = update_ensemble(state_data[:, :, nloc_sub], obs_prior, obs_incr, lfactor[:, nloc_sub])

@njit
def update_local_state_hybrid(state_data, state_static, obs_prior, obs_prior_static, obs_incr,
                              h_dist, v_dist, t_dist,
                              hroi, vroi, troi,
                              h_local_func, v_local_func, t_local_func,
                              impact_on_variable,
                              weight_dynamic, weight_static, hybrid_perturbation) -> None:
    """update_local_state_linear with the hybrid covariance; the static members are not updated"""
    nens, nfld, nloc = state_data.shape

    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    nloc_sub = np.where(h_lfactor>0)[0]  # subset of range(nloc) to update

    lfactor = np.zeros((nfld, nloc))
    for l in nloc_sub:
        for n in range(nfld):
            lfactor[n, l] = h_lfactor[l] * v_lfactor[n, l] * t_lfactor[n] * impact_on_variable[n]

    state_data[:, :, nloc_sub] = update_ensemble_hybrid(state_data[:, :, nloc_sub], state_static[:, :, nloc_sub],
                                                        obs_prior, obs_prior_static, obs_incr, lfactor[:, nloc_sub],
                                                        weight_dynamic, weight_static, hybrid_perturbation)

@njit
def update_local_obs_linear(obs_data, used, obs_prior, obs_incr,
                            h_dist, v_dist, t_dist,
                            hroi, vroi, troi,
                            h_local_func, v_local_func, t_local_func,
                            impact_on_variable):

    # distance between local obs_data and the obs being assimilated
    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    lfactor = h_lfactor * v_lfactor * t_lfactor * impact_on_variable

    # update the unused obs within roi
    ind = np.where(np.logical_and(~used, lfactor>0))[0]

    obs_data[:, ind] = update_ensemble(obs_data[:, ind], obs_prior, obs_incr, lfactor[ind])

@njit
def update_local_obs_hybrid(obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                            h_dist, v_dist, t_dist,
                            hroi, vroi, troi,
                            h_local_func, v_local_func, t_local_func,
                            impact_on_variable,
                            weight_dynamic, weight_static, hybrid_perturbation):
    """update_local_obs_linear with the hybrid covariance; the static members' obs priors are not updated"""
    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    lfactor = h_lfactor * v_lfactor * t_lfactor * impact_on_variable

    # update the unused obs within roi
    ind = np.where(np.logical_and(~used, lfactor>0))[0]

    obs_data[:, ind] = update_ensemble_hybrid(obs_data[:, ind], obs_data_static[:, ind],
                                              obs_prior, obs_prior_static, obs_incr, lfactor[ind],
                                              weight_dynamic, weight_static, hybrid_perturbation)

@njit
def update_ensemble(ens_prior, obs_prior, obs_incr, local_factor) -> np.ndarray:
    nens = ens_prior.shape[0]
    ens_post = ens_prior.copy()

    # obs-space statistics
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_var = np.sum((obs_prior - obs_prior_mean)**2) / (nens-1)

    # if there is no prior spread, don't update at all
    if obs_prior_var == 0:
        return ens_post

    cov = np.zeros(ens_prior.shape[1:])
    for m in range(nens):
        cov += ens_prior[m, ...] * (obs_prior[m] - obs_prior_mean) / (nens-1)

    reg_factor = cov / obs_prior_var

    # the updated posterior ensemble
    for m in range(nens):
        ens_post[m, ...] = ens_prior[m, ...] + local_factor * reg_factor * obs_incr[m]

    return ens_post

@njit
def update_ensemble_hybrid(ens_prior, ens_static, obs_prior, obs_prior_static, obs_incr, local_factor,
                           weight_dynamic, weight_static, hybrid_perturbation) -> np.ndarray:
    """
    Regress the obs-space increments onto the dynamic members (ens_prior) with the hybrid covariance:
    the mean increment with the hybrid regression cov_h(x,y)/var_h(y); the perturbation increments with
    the same hybrid regression (hybrid_perturbation=True) or with the dynamic ensemble regression (False).
    The static members (ens_static, obs_prior_static) enter the hybrid covariance and are not updated.
    """
    nens = ens_prior.shape[0]
    nens_static = ens_static.shape[0]
    ens_post = ens_prior.copy()

    # dynamic ensemble statistics
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_var = np.sum((obs_prior - obs_prior_mean)**2) / max(nens - 1, 1)
    cov = np.zeros(ens_prior.shape[1:])
    for m in range(nens):
        cov += ens_prior[m, ...] * (obs_prior[m] - obs_prior_mean) / max(nens - 1, 1)

    # hybrid variance and covariance
    obs_prior_var_hybrid = weight_dynamic * obs_prior_var
    cov_hybrid = weight_dynamic * cov
    if nens_static > 1:
        obs_prior_mean_static = np.mean(obs_prior_static)
        obs_prior_var_hybrid += weight_static * np.sum((obs_prior_static - obs_prior_mean_static)**2) / (nens_static - 1)
        for m in range(nens_static):
            cov_hybrid += weight_static * ens_static[m, ...] * (obs_prior_static[m] - obs_prior_mean_static) / (nens_static - 1)

    # if there is no prior spread, don't update at all
    if obs_prior_var_hybrid == 0:
        return ens_post

    reg_factor_mean = cov_hybrid / obs_prior_var_hybrid
    if hybrid_perturbation or obs_prior_var == 0:
        reg_factor_pert = reg_factor_mean
    else:
        reg_factor_pert = cov / obs_prior_var

    # the updated posterior ensemble, mean and perturbation increments regressed separately
    obs_incr_mean = np.mean(obs_incr)
    for m in range(nens):
        ens_post[m, ...] = ens_prior[m, ...] + local_factor * (reg_factor_mean * obs_incr_mean
                                                               + reg_factor_pert * (obs_incr[m] - obs_incr_mean))

    return ens_post
