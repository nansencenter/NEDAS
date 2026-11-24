import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.serial import SerialAssimilator

class EAKFAssimilator(SerialAssimilator):
    supports_static_members = True
    supports_hybrid_perturbation = True

    def assimilation_algorithm(self, c):
        # weights of the dynamic and static covariance in P = weight_dynamic*P_d + weight_static*P_s
        # (weight_dynamic = 1-beta, weight_static = beta*static_var_scaling, see assim_tools/covariance);
        # without static members this is the plain EAKF, weight_dynamic = 1
        self.weight_dynamic = 1.0 - c.covariance.beta
        self.weight_static = c.covariance.beta * c.covariance.static_var_scaling
        self.hybrid_perturbation = c.covariance.hybrid_perturbation
        super().assimilation_algorithm(c)

    def obs_increment(self, obs_prior, obs_prior_static, obs, obs_err):
        return obs_increment_eakf(obs_prior, obs_prior_static, obs, obs_err,
                                  self.weight_dynamic, self.weight_static, self.hybrid_perturbation)

    def update_local_state(self, state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                           ind, state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func, correlation_local_func,
                           impact_on_variable) -> None:
        return update_local_state_linear(state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                                         ind, state_h_dist, state_v_dist, state_t_dist,
                                         hroi, vroi, troi,
                                         h_local_func, v_local_func, t_local_func, correlation_local_func,
                                         impact_on_variable,
                                         self.weight_dynamic, self.weight_static, self.hybrid_perturbation)

    def update_local_obs(self, obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                         ind, h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func, correlation_local_func,
                         impact_on_variable) -> None:
        return update_local_obs_linear(obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                                       ind, h_dist, v_dist, t_dist,
                                       hroi, vroi, troi,
                                       h_local_func, v_local_func, t_local_func, correlation_local_func,
                                       impact_on_variable,
                                       self.weight_dynamic, self.weight_static, self.hybrid_perturbation)

@njit
def obs_increment_eakf(obs_prior, obs_prior_static, obs, obs_err,
                       weight_dynamic, weight_static, hybrid_perturbation) -> np.ndarray:
    """
    Ensemble adjustment Kalman filter (Anderson 2003) obs-space increments of the dynamic members.

    The prior variance is the hybrid one, weight_dynamic*var_dynamic + weight_static*var_static, with the
    static members (obs_prior_static, covariance_def.nens_static) held fixed; the mean moves with it. The
    perturbations contract with the hybrid variance (hybrid_perturbation=True: the Whitaker and Hamill 2002
    serial square root with the hybrid gain, Counillon et al. 2009) or with the dynamic ensemble variance
    alone (False, as in Wang et al. 2007). Plain EAKF = no static members and weight_dynamic = 1.
    The mean of the returned increments is the mean increment, the rest the perturbation increments.
    """
    nens = obs_prior.size
    nens_static = obs_prior_static.size

    # obs error variance
    obs_var = obs_err**2

    # obs_prior separate into mean+perturbation
    obs_prior_mean = np.mean(obs_prior)
    obs_prior_pert = obs_prior - obs_prior_mean

    # compute prior error variance, of the dynamic members and of the hybrid covariance
    obs_prior_var_dynamic = np.sum(obs_prior_pert**2) / max(nens - 1, 1)
    obs_prior_var = weight_dynamic * obs_prior_var_dynamic
    if nens_static > 1:
        obs_prior_static_pert = obs_prior_static - np.mean(obs_prior_static)
        obs_prior_var += weight_static * np.sum(obs_prior_static_pert**2) / (nens_static - 1)

    var_ratio = obs_var / (obs_prior_var + obs_var)

    # new mean is weighted average between obs_prior_mean and obs
    obs_post_mean = var_ratio * obs_prior_mean + (1 - var_ratio) * obs

    # new pert is adjusted by sqrt(var_ratio), a deterministic square-root filter
    if not hybrid_perturbation:
        var_ratio = obs_var / (obs_prior_var_dynamic + obs_var)
    obs_post_pert = np.sqrt(var_ratio) * obs_prior_pert

    # assemble the increments
    obs_incr = obs_post_mean + obs_post_pert - obs_prior

    return obs_incr

@njit
def update_local_state_linear(state_data, state_static, obs_prior, obs_prior_static, obs_incr,
                              ind, h_dist, v_dist, t_dist,
                              hroi, vroi, troi,
                              h_local_func, v_local_func, t_local_func, correlation_local_func,
                              impact_on_variable,
                              weight_dynamic, weight_static, hybrid_perturbation) -> None:

    nloc, nfld, nens = state_data.shape   # point-major (see update_ensemble)

    # distances are given for the candidate points ind
    h_lfactor = h_local_func(h_dist, hroi)
    near = np.where(h_lfactor>0)[0]
    if near.size == 0:
        return
    nloc_sub = ind[near]  # subset of range(nloc) to update

    v_lfactor = v_local_func(v_dist[:, near], vroi)
    t_lfactor = t_local_func(t_dist, troi)

    lfactor = np.empty((nfld, nloc_sub.size))
    updated = np.zeros(nfld, dtype=np.bool_)
    for n in range(nfld):
        for j in range(nloc_sub.size):
            lfactor[n, j] = h_lfactor[near[j]] * v_lfactor[n, j] * t_lfactor[n] * impact_on_variable[n]
            if lfactor[n, j] > 0:
                updated[n] = True
    # only the fields within reach (levels beyond vroi, other times, no impact are left out)
    flds = np.where(updated)[0]
    if flds.size == 0:
        return

    update_ensemble(state_data, state_static, flds, nloc_sub, lfactor[flds],
                            obs_prior, obs_prior_static, obs_incr,
                            correlation_local_func, weight_dynamic, weight_static, hybrid_perturbation)

@njit
def update_local_obs_linear(obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                            ind, h_dist, v_dist, t_dist,
                            hroi, vroi, troi,
                            h_local_func, v_local_func, t_local_func, correlation_local_func,
                            impact_on_variable,
                            weight_dynamic, weight_static, hybrid_perturbation):

    # distance between the candidate local obs ind and the obs being assimilated
    h_lfactor = h_local_func(h_dist, hroi)
    v_lfactor = v_local_func(v_dist, vroi)
    t_lfactor = t_local_func(t_dist, troi)

    lfactor = h_lfactor * v_lfactor * t_lfactor * impact_on_variable

    # update the unused obs within roi
    near = np.where(np.logical_and(~used[ind], lfactor>0))[0]
    if near.size == 0:
        return

    # the obs are one field of nlobs points, point-major: (nlobs, 1, nens) views, so the writes land in obs_data
    nlobs, nens = obs_data.shape
    update_ensemble(obs_data.reshape((nlobs, 1, nens)),
                            obs_data_static.reshape((nlobs, 1, obs_data_static.shape[1])),
                            np.zeros(1, dtype=np.int64), ind[near], lfactor[near].reshape((1, near.size)),
                            obs_prior, obs_prior_static, obs_incr,
                            correlation_local_func, weight_dynamic, weight_static, hybrid_perturbation)

@njit
def update_ensemble(ens, ens_static, flds, sub, lfactor, obs_prior, obs_prior_static, obs_incr,
                            correlation_local_func, weight_dynamic, weight_static, hybrid_perturbation) -> None:
    """
    Regress the obs-space increments onto the members, in place: ens[sub[j], flds[n], :] += gain * obs_incr,
    with lfactor[n, j] the local factor of field flds[n] at point sub[j]. ens is point-major,
    (nloc, nfld, nens), so the members of one (point, field) are one contiguous row, as DART stores its
    copies: each row is read twice and written once and nothing its size is allocated.

    Hybrid covariance: the static members (ens_static, (nloc, nfld, nens_static), and obs_prior_static)
    enter the regression with weight_static, the dynamic ones with weight_dynamic; the static members
    are never updated. Unless hybrid_perturbation, the perturbations are updated with the dynamic
    covariance alone. Correlation-based localization, if given, multiplies lfactor by a factor of the
    sample correlation of the dynamic members.
    """
    nens, nfld = ens.shape[2], flds.size
    nsub = sub.size
    nens_static = ens_static.shape[2]

    # obs-space statistics of the dynamic members. ss = 'sum of squares'
    obs_prior_mean = np.mean(obs_prior)
    ypert = obs_prior - obs_prior_mean
    obs_prior_ss = np.sum(ypert**2)
    ypert_sum = np.sum(ypert)    # 0 up to roundoff, kept so cov is exactly sum((x - mean(x)) * ypert)

    # sum(x) and sum(x * ypert) over each row, hence cov = sample covariance * (nens - 1)
    xsum = np.zeros((nfld, nsub))
    cov = np.zeros((nfld, nsub))
    for n in range(nfld):
        for j in range(nsub):
            row = ens[sub[j], flds[n]]
            xs, c = 0.0, 0.0
            for m in range(nens):
                xs += row[m]
                c += row[m] * ypert[m]
            xsum[n, j] = xs
            cov[n, j] = c - xs / nens * ypert_sum

    if correlation_local_func is not None:
        # a second pass, about the mean, for the state sum of squares (no cancellation)
        r = np.zeros((nfld, nsub))
        for n in range(nfld):
            for j in range(nsub):
                row = ens[sub[j], flds[n]]
                ss = 0.0
                for m in range(nens):
                    d = row[m] - xsum[n, j] / nens
                    ss += d * d
                if obs_prior_ss > 0.0 and ss > 0.0:
                    r[n, j] = cov[n, j] / np.sqrt(ss * obs_prior_ss)
        lfactor = lfactor * correlation_local_func(r, nens)

    # variance and covariance of the hybrid covariance
    obs_prior_var_hybrid = weight_dynamic * obs_prior_ss / (nens - 1)
    reg_factor = weight_dynamic * cov / (nens - 1)
    if nens_static > 1:
        ypert_static = obs_prior_static - np.mean(obs_prior_static)
        obs_prior_var_hybrid += weight_static * np.sum(ypert_static**2) / (nens_static - 1)
        w = weight_static * ypert_static / (nens_static - 1)
        for n in range(nfld):
            for j in range(nsub):
                row = ens_static[sub[j], flds[n]]
                for m in range(nens_static):
                    reg_factor[n, j] += row[m] * w[m]

    # if there is no prior spread, don't update at all
    if obs_prior_var_hybrid == 0:
        return
    reg_factor /= obs_prior_var_hybrid

    # the mean and perturbation increments are regressed with the same coefficient, unless the
    # perturbations are updated with the dynamic ensemble covariance alone
    split_increment = (not hybrid_perturbation) and nens_static > 1 and obs_prior_ss > 0
    if not split_increment:
        gain = lfactor * reg_factor
        for n in range(nfld):
            for j in range(nsub):
                row = ens[sub[j], flds[n]]
                for m in range(nens):
                    row[m] += gain[n, j] * obs_incr[m]
        return

    obs_incr_mean = np.mean(obs_incr)
    gain_mean = lfactor * reg_factor * obs_incr_mean
    gain_pert = lfactor * cov / obs_prior_ss
    for n in range(nfld):
        for j in range(nsub):
            row = ens[sub[j], flds[n]]
            for m in range(nens):
                row[m] += gain_mean[n, j] + gain_pert[n, j] * (obs_incr[m] - obs_incr_mean)
