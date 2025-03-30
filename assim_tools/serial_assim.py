
import numpy as np
from utils.parallel import by_rank, bcast_by_root
from utils.njit import njit
from utils.progress import print_with_cache, progress_bar
#TODO from utils.distribution import normal_cdf, inv_weighted_normal_cdf
from .obs import global_obs_list
# from .covariance import covariance_model
from .packing import pack_state_data, unpack_state_data, pack_obs_data, unpack_obs_data
from .localization import local_factor_distance_based

def serial_assim(c, state_prior, z_state, lobs, lobs_prior):
    """
    serial assimilation goes through the list of observations one by one
    for each obs the near by state variables are updated one by one.
    so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
    """
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    par_id = c.pid_mem

    state_data = pack_state_data(c, par_id, state_prior, z_state)
    nens, nfld, nloc = state_data['state_prior'].shape

    obs_data = pack_obs_data(c, par_id, lobs, lobs_prior)
    obs_list = bcast_by_root(c.comm)(global_obs_list)(c)

    print_1p('>>> assimilate in serial mode:\n')
    ##go through the entire obs list, indexed by p, one scalar obs at a time
    for p in range(len(obs_list)):
        print_1p(progress_bar(p, len(obs_list)))

        obs_rec_id, v, owner_pid, i = obs_list[p]
        obs_rec = c.obs_info['records'][obs_rec_id]

        ##1. if the pid owns this obs, broadcast it to all pid
        if c.pid_mem == owner_pid:
            ##collect obs info
            obs = {}
            obs['prior'] = obs_data['obs_prior'][:, i]
            for key in ('obs', 'x', 'y', 'z', 't', 'err_std'):
                obs[key] = obs_data[key][i]
            for key in ('hroi', 'vroi', 'troi', 'impact_on_state'):
                obs[key] = obs_data[key][obs_rec_id]
            ##mark this obs as used
            obs_data['used'][i] = True

        else:
            obs = None
        obs = c.comm_mem.bcast(obs, root=owner_pid)

        ##compute obs-space increment
        obs_incr = obs_increment(obs['prior'], obs['obs'], obs['err_std'], c.filter_type)

        ##2. all pid update their own locally stored state:
        state_h_dist = c.grid.distance(obs['x'], state_data['x'], obs['y'], state_data['y'], p=2)
        state_v_dist = np.abs(obs['z'] - state_data['z'])
        state_t_dist = np.abs(obs['t'] - state_data['t'])
        update_local_state(state_data['state_prior'], obs['prior'], obs_incr,
                           state_h_dist, state_v_dist, state_t_dist,
                           obs['hroi'], obs['vroi'], obs['troi'],
                           c.localization['htype'], c.localization['vtype'], c.localization['ttype'])

        ##3. all pid update their own locally stored obs:
        obs_h_dist = c.grid.distance(obs['x'], obs_data['x'], obs['y'], obs_data['y'], p=2)
        obs_v_dist = np.abs(obs['z'] - obs_data['z'])
        obs_t_dist = np.abs(obs['t'] - obs_data['t'])
        update_local_obs(obs_data['obs_prior'], obs_data['used'], obs['prior'], obs_incr,
                         obs_h_dist, obs_v_dist, obs_t_dist,
                         obs['hroi'], obs['vroi'], obs['troi'],
                         c.localization['htype'], c.localization['vtype'], c.localization['ttype'])
    unpack_state_data(c, par_id, state_prior, state_data)
    unpack_obs_data(c, par_id, lobs, lobs_prior, obs_data)
    print_1p(' done.\n')
    return state_prior, lobs_prior

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

    if filter_type == 'EAKF':
        obs_incr = obs_increment_EAKF(nens, obs_prior, obs_prior_mean, obs_prior_pert, obs_prior_var, obs, obs_var)

    elif filter_type == 'RHF':
        obs_incr = obs_increment_RHF(nens, obs_prior, obs_prior_var, obs, obs_var)

    else:
        raise NotImplementedError('Error: unknown filter type: '+filter_type)

    return obs_incr

@njit(cache=True)
def obs_increment_EAKF(nens, obs_prior, obs_prior_mean, obs_prior_pert, obs_prior_var, obs, obs_var):
    """ensemble adjustment Kalman filter (Anderson 2003)"""
    var_ratio = obs_var / (obs_prior_var + obs_var)

    ##new mean is weighted average between obs_prior_mean and obs
    obs_post_mean = var_ratio * obs_prior_mean + (1 - var_ratio) * obs

    ##new pert is adjusted by sqrt(var_ratio), a deterministic square-root filter
    obs_post_pert = np.sqrt(var_ratio) * obs_prior_pert

    ##assemble the increments
    obs_incr = obs_post_mean + obs_post_pert - obs_prior
    return obs_incr

@njit(cache=True)
def obs_increment_RHF(nens, obs_prior, obs_prior_var, obs, obs_var):
    """rank histogram filter (Anderson 2010)"""
    ##index sort of the ensemble members
    ens_ind = np.argsort(obs_prior)
    x = obs_prior[ens_ind]

    ##initialize arrays
    like = np.zeros(nens)
    like_dense = np.zeros(nens)
    height = np.zeros(nens)
    mass = np.zeros(nens+1)
    obs_incr = np.zeros(nens)

    ##compute likelihood
    norm_const = 1.0 / np.sqrt(2 * np.pi * obs_var)
    for m in range(nens):
        like[m] = norm_const * np.exp( -1. * (x[m] - obs)**2 / (2. * obs_var))
    for m in range(1, nens):
        like_dense[m] = (like[m-1] + like[m]) / 2.

    ##product of likelihood gaussian with prior gaussian tails
    # var_ratio = obs_var / (obs_prior_var + obs_var)
    # new_var = var_ratio * obs_prior_var
    # dist_for_unit_sd = inv_weighted_normal_cdf(1., 0., 1., 1./(nens+1))

    # left_mean = x[0] - dist_for_unit_sd * np.sqrt(obs_prior_var)
    # new_mean_left = var_ratio * (left_mean + obs_prior_var * obs / obs_var)
    # prod_weight_left = np.exp(-0.5 * (left_mean**2 / obs_prior_var + obs**2 / obs_var - new_mean_left**2 / new_var)) / np.sqrt(obs_prior_var + obs_var) / np.sqrt(2.*np.pi)
    # mass[0] = normal_cdf(x[0], new_mean_left, np.sqrt(new_var)) * prod_weight_left

    # right_mean = x[-1] + dist_for_unit_sd * np.sqrt(obs_prior_var)
    # new_mean_right = var_ratio * (right_mean + obs_prior_var * obs / obs_var)
    # prod_weight_right = np.exp(-0.5 * (right_mean**2 / obs_prior_var + obs**2 / obs_var - new_mean_right**2 / new_var)) / np.sqrt(obs_prior_var + obs_var) / np.sqrt(2.*np.pi)
    # mass[-1] = (1. - normal_cdf(x[-1], new_mean_right, np.sqrt(new_var))) * prod_weight_right

    ##The mass in each interior box m is the height times bin width
    ##The height of likelihood function is like_dense[m]
    ##For the prior, mass is 1/(nens+1), multiply by mean like_dense to get posterior
    # for m in range(1, nens):
    #     mass[m] = like_dense[m] / (nens + 1)
    #     if x[m] == x[m-1]:
    #         height[m] = -1.
    #     else:
    #         height[m] = 1. / ((nens + 1) * (x[m] - x[m-1]))

    ##normalize the mass to get pdf
    # cum_mass = 

    return obs_incr

@njit(cache=True)
def update_local_state(state_data, obs_prior, obs_incr,
                       h_dist, v_dist, t_dist,
                       hroi, vroi, troi,
                       localize_htype, localize_vtype, localize_ttype,
                       ):

    nens, nfld, nloc = state_data.shape

    ##localization factor
    h_lfactor = local_factor_distance_based(h_dist, hroi, localize_htype)
    v_lfactor = local_factor_distance_based(v_dist, vroi, localize_vtype)
    t_lfactor = local_factor_distance_based(t_dist, troi, localize_ttype)

    nloc_sub = np.where(h_lfactor>0)[0]  ##subset of range(nloc) to update

    lfactor = np.zeros((nfld, nloc))
    for l in nloc_sub:
        for n in range(nfld):
            lfactor[n, l] = h_lfactor[l] * v_lfactor[n, l] * t_lfactor[n]

    state_data[:, :, nloc_sub] = update_ensemble(state_data[:, :, nloc_sub], obs_prior, obs_incr, lfactor[:, nloc_sub])

@njit(cache=True)
def update_local_obs(obs_data, used, obs_prior, obs_incr,
                     h_dist, v_dist, t_dist,
                     hroi, vroi, troi,
                     localize_htype, localize_vtype, localize_ttype,
                     ):

    nens, nlobs = obs_data.shape

    ##distance between local obs_data and the obs being assimilated
    h_lfactor = local_factor_distance_based(h_dist, hroi, localize_htype)
    v_lfactor = local_factor_distance_based(v_dist, vroi, localize_vtype)
    t_lfactor = local_factor_distance_based(t_dist, troi, localize_ttype)

    lfactor = h_lfactor * v_lfactor * t_lfactor

    ##update the unused obs within roi
    ind = np.where(np.logical_and(~used, lfactor>0))[0]

    obs_data[:, ind] = update_ensemble(obs_data[:, ind], obs_prior, obs_incr, lfactor[ind])

@njit(cache=True)
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

@njit(cache=True)
def transform_to_probit():
    pass

@njit(cache=True)
def transform_from_probit():
    pass
