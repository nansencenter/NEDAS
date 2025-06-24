import numpy as np
from NEDAS.utils.njit import njit
from NEDAS.assim_tools.assimilators.batch import BatchAssimilator

class ETKFAssimilator(BatchAssimilator):

    def local_analysis(self, c, loc_id, ind, hlfactor, state_data, obs_data):
        state_var_id = state_data['var_id']  ##variable id for each field (nfld)
        state_z = state_data['z'][:, loc_id]
        state_t = state_data['t'][:]

        ##vertical, time and cross-variable (impact_on_state) localization
        obs_value = obs_data['obs'][ind]
        obs_err = obs_data['err_std'][ind]
        obs_z = obs_data['z'][ind]
        obs_t = obs_data['t'][ind]
        obs_rec_id = obs_data['obs_rec_id'][ind]
        vroi = obs_data['vroi'][obs_rec_id]
        troi = obs_data['troi'][obs_rec_id]
        impact_on_state = obs_data['impact_on_state'][:, state_var_id][obs_rec_id]

        local_analysis_main(state_data['state_prior'][...,loc_id], obs_data['obs_prior'][:,ind],
                            obs_value, obs_err, hlfactor,
                            state_z, obs_z, vroi, c.localization_funcs['vertical'],
                            state_t, obs_t, troi, c.localization_funcs['temporal'],
                            impact_on_state, self.rfactor, self.kfactor, self.nlobs_max)

@njit
def local_analysis_main(state_prior, obs_prior,
                        obs, obs_err, hlfactor,
                        state_z, obs_z, vroi, vlocal_func,
                        state_t, obs_t, troi, tlocal_func,
                        impact_on_state, rfactor, kfactor, nlobs_max) -> None:
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
        vlfactor = vlocal_func(vdist, vroi)
        if (vlfactor==0).all():
            continue  ##the state is outside of vroi of all obs, skip

        ##temporal localization
        tdist = np.abs(obs_t - state_t[n])
        tlfactor = tlocal_func(tdist, troi)
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
            weights = ensemble_transform_weights(obs[ind], obs_err[ind], obs_prior[:, ind], lfactor[ind], rfactor, kfactor)

        ##perform local analysis and update the ensemble state
        state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)

        lfactor_old = lfactor[ind]
        weights_old = weights

@njit
def ensemble_transform_weights(obs, obs_err, obs_prior, local_factor, rfactor, kfactor):
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

    ##obs_prior_pert S and innovation dy, normalized by sqrt(nens-1) R^-0.5
    S = np.zeros((nlobs, nens))
    dy = np.zeros((nlobs))
    for p in range(nlobs):
        S[p, :] = (obs_prior[:, p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
        dy[p] = (obs[p] - obs_prior_mean[p]) * local_factor[p] / obs_err[p]
    S /= np.sqrt(nens-1)
    dy /= np.sqrt(nens-1)

    ##TODO:factor in the correlated R in obs_err, need another SVD of R
    # obs_err_corr

    ##----first part of weights: update of mean
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

    ##---second part of weights: update of ensemble spread
    if rfactor > 1:  ##scale S with rfactor and recompute the var_ratio
        S /= np.sqrt(rfactor)
        try:
            var_ratio_inv = np.eye(nens) + S.T @ S
            L, sv, Rh = np.linalg.svd(var_ratio_inv)
            var_ratio = L @ np.diag(sv**-1) @ Rh
            gain = var_ratio @ S.T
        except:
            ##if failed just return equal weights (no update)
            return np.eye(nens)

    var_ratio_sqrt = L @ np.diag(sv**-0.5) @ Rh

    weights += var_ratio_sqrt

    return weights

@njit
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

