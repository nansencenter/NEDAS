import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.batch import BatchAssimilator


class TopazDEnKFAssimilator(BatchAssimilator):
    rfactor1: float
    rfactor: float
    kfactor: float
    nlobs_max: int | None
    supports_static_members = True
    supports_hybrid_perturbation = True

    def assimilation_algorithm(self, c):
        # scaling of the dynamic/static anomalies that gives the hybrid covariance
        self.anomaly_factors = c.covariance.anomaly_factors()
        super().assimilation_algorithm(c)

    def local_analysis(self, c, loc_id, ind, hlfactor, state_data, obs_data):
        state_var_id = state_data['var_id']  # variable id for each field (nfld)
        state_z = state_data['z'][:, loc_id]
        state_t = state_data['t'][:]

        # vertical, time and cross-variable (impact_on_variable) localization
        obs_value = obs_data['obs'][ind]
        obs_err_raw = obs_data['err_std'][ind]
        obs_z = obs_data['z'][ind]
        obs_t = obs_data['t'][ind]
        obs_rec_id = obs_data['obs_rec_id'][ind]
        vroi = obs_data['vroi'][obs_rec_id]
        troi = obs_data['troi'][obs_rec_id]
        impact_on_variable = obs_data['impact_on_variable'][:, state_var_id][obs_rec_id]

        # ---------------------------------------------------------------
        # Pre-analysis obs error adjustment (matches Fortran's obs_QC +
        # RFACTOR1 in m_prep_4_EnKF/m_obs):
        #
        # Fortran rfactor1:  obs(o)%var = obs(o)%var * RFACTOR1
        # Fortran kfactor:   obs(o)%var = sqrt((svar+ovar)^2 + svar*(inn/kf)^2) - svar
        #
        # Fortran's obs_QC computes svar/inn once per observation, before the
        # per-gridpoint local-analysis loop starts, using that observation's
        # own ensemble of model-predicted values (S(o,:)). obs_data['obs_prior']
        # is likewise fixed per observation regardless of which location's
        # `ind` selects it, so recomputing this here per location (rather than
        # once globally beforehand) gives the exact same svar/inn per
        # observation, not merely an approximation -- it's just redundant
        # computation, repeated once per location instead of once overall.
        # ---------------------------------------------------------------
        obs_err = obs_err_raw * np.sqrt(self.rfactor1)

        obs_prior = obs_data['obs_prior'][:, ind]
        obs_prior_static = obs_data['obs_prior_static'][:, ind]
        obs_prior_mean = np.mean(obs_prior, axis=0)
        if obs_prior_static.shape[0] == 0:
            obs_prior_var = np.var(obs_prior, axis=0, ddof=1)
        else:
            # prior variance of the hybrid covariance, P = (1-beta)*P_d + beta*static_var_scaling*P_s
            fac_dynamic, fac_static = self.anomaly_factors
            obs_prior_var = (fac_dynamic**2 * np.sum((obs_prior - obs_prior_mean)**2, axis=0)
                             + fac_static**2 * np.sum((obs_prior_static - np.mean(obs_prior_static, axis=0))**2, axis=0))
        innov = obs_value - obs_prior_mean
        obs_var = obs_err**2
        obs_var = np.sqrt((obs_prior_var + obs_var)**2
                          + obs_prior_var * (innov / self.kfactor)**2) - obs_prior_var
        obs_err = np.sqrt(obs_var)

        local_analysis_main(state_data['state_prior'][..., loc_id], obs_prior,
                            state_data['state_static'][..., loc_id], obs_prior_static,
                            obs_value, obs_err, hlfactor,
                            state_z, obs_z, vroi, c.localization_funcs['vertical'],
                            state_t, obs_t, troi, c.localization_funcs['temporal'],
                            impact_on_variable, self.rfactor, self.nlobs_max,
                            *self.anomaly_factors, c.covariance.hybrid_perturbation)


@njit
def local_analysis_main(state_prior, obs_prior, state_static, obs_prior_static,
                        obs, obs_err, hlfactor,
                        state_z, obs_z, vroi, vlocal_func,
                        state_t, obs_t, troi, tlocal_func,
                        impact_on_variable, rfactor, nlobs_max,
                        fac_dynamic, fac_static, hybrid_perturbation) -> None:
    """perform local analysis for one location in the analysis grid partition, updating the
    dynamic members in state_prior; the static members (state_static, obs_prior_static) enter
    through the hybrid covariance, see ensemble_transform_weights

    obs_err: already adjusted by rfactor1 and kfactor (applied once per
             location in the parent method) — matches Fortran's convention
             of a single modified obs variance used throughout.
    """
    nens, nfld = state_prior.shape
    nens_obs, nlobs = obs_prior.shape
    if nens_obs != nens:
        raise ValueError('Error: number of ensemble members in state and obs do not match!')
    nens_static = state_static.shape[0]
    if obs_prior_static.shape[0] != nens_static:
        raise ValueError('Error: number of static members in state and obs do not match!')

    lfactor_old = np.zeros(nlobs)
    weights_old = np.eye(nens)
    weights_static_old = np.zeros((nens_static, nens))

    # loop through the field records
    for n in range(nfld):

        # vertical localization
        vdist = np.abs(obs_z - state_z[n])
        vlfactor = vlocal_func(vdist, vroi)
        if (vlfactor == 0).all():
            continue  # the state is outside of vroi of all obs, skip

        # temporal localization
        tdist = np.abs(obs_t - state_t[n])
        tlfactor = tlocal_func(tdist, troi)
        if (tlfactor == 0).all():
            continue  # the state is outside of troi of all obs, skip

        # total lfactor
        lfactor = hlfactor * vlfactor * tlfactor * impact_on_variable[:, n]
        if (lfactor == 0).all():
            continue

        # if prior spread is zero (in both the dynamic and the static members), don't update
        if np.std(state_prior[:, n]) == 0 and (nens_static == 0 or np.std(state_static[:, n]) == 0):
            continue

        # only need to assimilate obs with lfactor>0
        ind = np.where(lfactor > 0)[0]

        # TODO:get rid of obs if obs_prior is nan
        # valid = np.array([np.isnan(obs_prior[:,i]).any() for i in ind])
        # ind = ind[valid]

        # limit number of local obs if needed
        # Fortran (get_local_obs): sorts by distance, keeps nlobs closest
        # Python: sorts by lfactor (descending), keeps nlobs_max highest-impact
        if nlobs_max > 0 and len(ind) > nlobs_max:
            sort_ind = np.argsort(lfactor[ind])[::-1]
            ind = ind[sort_ind]
            ind = ind[:nlobs_max]

        # use cached weight if no localization is applied, to avoid repeated computation
        if n > 0 and len(ind) == len(lfactor_old) and (lfactor[ind] == lfactor_old).all():
            weights = weights_old
            weights_static = weights_static_old
        else:
            weights, weights_static = ensemble_transform_weights(obs[ind], obs_err[ind],
                                                                 obs_prior[:, ind], obs_prior_static[:, ind],
                                                                 lfactor[ind], rfactor,
                                                                 fac_dynamic, fac_static, hybrid_perturbation)

        # perform local analysis and update the ensemble state,
        # the static members contribute to the dynamic members' update through weights_static
        state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)
        for m in range(nens_static):
            state_prior[:, n] += state_static[m, n] * weights_static[m, :]

        lfactor_old = lfactor[ind]
        weights_old = weights
        weights_static_old = weights_static


@njit
def ensemble_transform_weights(obs, obs_err, obs_prior, obs_prior_static, local_factor, rfactor,
                               fac_dynamic, fac_static, hybrid_perturbation):
    """Compute ensemble transform weight matrix (X5 in Evensen/Sakov notation) of the DEnKF
    (Sakov and Oke 2008).

    obs_err: observation error standard deviation, already adjusted by
             rfactor1 and kfactor (one-time adjustment, not inside this
             function — matches Fortran's obs_QC convention).

    rfactor : additional error inflation for the *anomaly* (spread) update
              only — matches Fortran's RFACTOR2 / ``rfactor`` argument to
              calc_X5().

    Hybrid covariance (see assim_tools/covariance): obs_prior (nens, nlobs) is from the dynamic
    members and obs_prior_static (nens_static, nlobs) from the static members; their anomalies scaled
    by ``fac_dynamic`` and ``fac_static`` form Z with Z Z^T = P = (1-beta)*P_d + beta*static_var_scaling*P_s
    (the combined anomalies of Counillon et al. 2009, Eq. 5). The mean is updated with the Kalman gain
    of P; the dynamic perturbations with A_d - 0.5 K H A_d, where K is the gain of P
    (hybrid_perturbation=True, Counillon et al. 2009) or of the dynamic ensemble alone (False, as in
    Wang et al. 2007). The plain DEnKF is no static members with fac_dynamic = 1/sqrt(nens-1).

    Returns  weights (nens x nens) and weights_static (nens_static x nens), such that
             E_post = E_prior @ weights + E_static @ weights_static  (columns for the dynamic members).
    """
    nens, nlobs = obs_prior.shape
    nens_static = obs_prior_static.shape[0]
    nens_combined = nens + nens_static

    # find mean of obs_prior, for the dynamic and the static members
    obs_prior_mean = np.zeros(nlobs)
    for m in range(nens):
        obs_prior_mean += obs_prior[m, :]
    obs_prior_mean /= nens
    obs_prior_mean_static = np.zeros(nlobs)
    for m in range(nens_static):
        obs_prior_mean_static += obs_prior_static[m, :]
    if nens_static > 0:
        obs_prior_mean_static /= nens_static

    # whitened, localized obs_prior perts of the dynamic (nlobs, nens) and static (nlobs, nens_static)
    # members and innovation dy w.r.t. the dynamic mean; S (nlobs, nens_combined) has the perts
    # scaled by their anomaly factors
    obs_anomaly = np.zeros((nlobs, nens))
    obs_anomaly_static = np.zeros((nlobs, nens_static))
    dy = np.zeros(nlobs)
    for p in range(nlobs):
        whitening_factor = local_factor[p] / obs_err[p]
        obs_anomaly[p, :] = (obs_prior[:, p] - obs_prior_mean[p]) * whitening_factor
        obs_anomaly_static[p, :] = (obs_prior_static[:, p] - obs_prior_mean_static[p]) * whitening_factor
        dy[p] = (obs[p] - obs_prior_mean[p]) * whitening_factor
    S = np.zeros((nlobs, nens_combined))
    S[:, :nens] = obs_anomaly * fac_dynamic
    S[:, nens:] = obs_anomaly_static * fac_static

    # ----first part of weights: update of mean, gain = (I + S^T S)^-1 S^T
    success, gain = kalman_gain_weights(S)
    if not success:
        return np.eye(nens), np.zeros((nens_static, nens))  # no update
    mean_gain_weights = gain @ dy
    w = fac_dynamic * mean_gain_weights[:nens]
    w_static = fac_static * mean_gain_weights[nens:]

    # ---second part of weights: update of ensemble spread, DEnKF: Taylor approx. of
    # var_ratio_sqrt (Sakov 2008); rfactor inflates the obs error for this part only
    pert_weights_static = np.zeros((nens_static, nens))
    if hybrid_perturbation:
        # A_d - 0.5 K H A_d in ensemble space, with K the gain of the hybrid covariance
        S_pert = S / np.sqrt(rfactor)
        success, gain = kalman_gain_weights(S_pert)
        if not success:
            return np.eye(nens), np.zeros((nens_static, nens))
        gain_obs_anomaly = gain @ (obs_anomaly / np.sqrt(rfactor))
        pert_weights = np.eye(nens) - 0.5 * fac_dynamic * gain_obs_anomaly[:nens, :]
        pert_weights_static = -0.5 * fac_static * gain_obs_anomaly[nens:, :]
    else:
        # plain DEnKF of the dynamic ensemble alone
        S_dynamic = obs_anomaly / np.sqrt(max(nens - 1, 1)) / np.sqrt(rfactor)
        success, gain = kalman_gain_weights(S_dynamic)
        if not success:
            return np.eye(nens), np.zeros((nens_static, nens))
        pert_weights = np.eye(nens) - 0.5 * gain @ S_dynamic

    # ensemble weight matrix, weights[:, m] is for the m-th member
    # also known as T in Bishop 2001, and X5 in Evensen textbook (and in Sakov 2012)
    weights = np.zeros((nens, nens))
    weights_static = np.zeros((nens_static, nens))
    for m in range(nens):
        weights[:, m] = w + pert_weights[:, m]
        weights_static[:, m] = w_static + pert_weights_static[:, m]

    return weights, weights_static


@njit
def kalman_gain_weights(S):
    """Gain weights (I + S^T S)^-1 S^T (n, nlobs) for the whitened obs perturbations S (nlobs, n),
    solved in ensemble space if nlobs >= n, else in obs space; returns (success, gain).

    note: the added I actually helps prevent issues if S^T S is not full rank
          when nlobs<n, there will be singular values of 0, but the full matrix
          can still be inverted with singular values of 1.
    """
    nlobs, n = S.shape
    if nlobs >= n:  # use ensemble space
        var_ratio_inv = np.eye(n) + S.T @ S
        try:
            L = np.linalg.cholesky(var_ratio_inv)
            L_inv = np.linalg.inv(L)
            var_ratio = L_inv.T @ L_inv
            gain = var_ratio @ S.T
        except Exception:
            # if inversion failed just return no update
            print('Error: failed to invert var_ratio_inv=', var_ratio_inv)
            return False, np.zeros((n, nlobs))
    else:  # use obs space
        var_ratio_inv = S @ S.T + np.eye(nlobs)
        try:
            L = np.linalg.cholesky(var_ratio_inv)
            L_inv = np.linalg.inv(L)
            var_ratio = L_inv.T @ L_inv
            gain = S.T @ var_ratio
        except Exception:
            # if inversion failed just return no update
            print('Error: failed to invert var_ratio_inv=', var_ratio_inv)
            return False, np.zeros((n, nlobs))
    return True, np.ascontiguousarray(gain)


@njit
def apply_ensemble_transform(ens_prior, weights):
    """Apply the weights to transform local ensemble"""

    nens = ens_prior.size
    ens_post = ens_prior.copy()

    # check if weights sum to 1
    for m in range(nens):
        sum_wgts = np.sum(weights[:, m])
        if np.abs(sum_wgts - 1) > 1e-5:
            print('Warning: sum of weights != 1 detected!')

    # apply the weights
    for m in range(nens):
        ens_post[m] = np.sum(ens_prior * weights[:, m])

    return ens_post
