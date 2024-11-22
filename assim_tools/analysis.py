import numpy as np
from utils.parallel import by_rank, bcast_by_root
from utils.njit import njit
from utils.progress import print_with_cache, progress_bar
from utils.conversion import t2h, h2t
# from utils.distribution import normal_cdf, inv_weighted_normal_cdf
from .localization import local_factor

def analysis(c, state_prior, z_state, lobs, lobs_prior):
    if c.assim_mode == 'batch':
        return batch_assim(c, state_prior, z_state, lobs, lobs_prior)
    elif c.assim_mode == 'serial':
        return serial_assim(c, state_prior, z_state, lobs, lobs_prior)

###pack/unpack local state and obs data for jitted functions:
def pack_local_state_data(c, par_id, state_prior, z_state):
    """pack state dict into arrays to be more easily handled by jitted funcs"""
    data = {}

    ##x,y coordinates for local state variables on pid
    if len(c.grid.x.shape)==2:
        ist,ied,di,jst,jed,dj = c.partitions[par_id]
        msk = c.mask[jst:jed:dj, ist:ied:di]
        data['x'] = c.grid.x[jst:jed:dj, ist:ied:di][~msk]
        data['y'] = c.grid.y[jst:jed:dj, ist:ied:di][~msk]
    else:
        inds = c.partitions[par_id]
        msk = c.mask[inds]
        data['x'] = c.grid.x[inds][~msk]
        data['y'] = c.grid.y[inds][~msk]

    data['fields'] = []
    for rec_id in c.rec_list[c.pid_rec]:
        rec = c.state_info['fields'][rec_id]
        v_list = [0, 1] if rec['is_vector'] else [None]
        for v in v_list:
            data['fields'].append((rec_id, v))

    ##var_id, index in state_info['variables'] list

    nfld = len(data['fields'])
    nloc = len(data['x'])
    data['t'] = np.full(nfld, np.nan)
    data['z'] = np.zeros((nfld, nloc))
    data['var_id'] = np.full(nfld, 0)
    data['err_type'] = np.full(nfld, 0)
    data['state_prior'] = np.full((c.nens, nfld, nloc), np.nan)
    for n in range(nfld):
        rec_id, v = data['fields'][n]
        rec = c.state_info['fields'][rec_id]
        data['t'][n] = t2h(rec['time'])
        data['err_type'][n] = c.state_info['err_types'].index(rec['err_type'])
        data['var_id'][n] = c.state_info['variables'].index(rec['name'])
        for m in range(c.nens):
            data['z'][n, :] += np.squeeze(z_state[m, rec_id][par_id][v, :]).astype(np.float32) / c.nens  ##ens mean z
            data['state_prior'][m, n, :] = np.squeeze(state_prior[m, rec_id][par_id][v, :])
    return data

def unpack_local_state_data(c, par_id, state_prior, data):
    """unpack data and write back to the original state_prior dict"""
    nfld = len(data['fields'])
    nloc = len(data['x'])

    for m in range(c.nens):
        for n in range(nfld):
            rec_id, v = data['fields'][n]
            state_prior[m, rec_id][par_id][v, :] = data['state_prior'][m, n, :]

def pack_local_obs_data(c, par_id, lobs, lobs_prior):
    """pack lobs and lobs_prior into arrays for the jitted functions"""
    data = {}

    ##number of local obs on partition
    nlobs = np.sum([lobs[r][par_id]['obs'].size for r in c.obs_info['records'].keys()])
    n_obs_rec = len(c.obs_info['records'])        ##number of obs records
    n_state_var = len(c.state_info['variables'])  ##number of state variable names

    data['obs_rec_id'] = np.full(nlobs, np.nan)
    data['obs'] = np.full(nlobs, np.nan)
    data['x'] = np.full(nlobs, np.nan)
    data['y'] = np.full(nlobs, np.nan)
    data['z'] = np.full(nlobs, np.nan)
    data['t'] = np.full(nlobs, np.nan)
    data['err_std'] = np.full(nlobs, np.nan)
    data['obs_prior'] = np.full((c.nens, nlobs), np.nan)
    data['used'] = np.full(nlobs, False)
    data['hroi'] = np.ones(n_obs_rec)
    data['vroi'] = np.ones(n_obs_rec)
    data['troi'] = np.ones(n_obs_rec)
    data['impact_on_state'] = np.ones((n_obs_rec, n_state_var))

    i = 0
    for obs_rec_id in range(n_obs_rec):
        obs_rec = c.obs_info['records'][obs_rec_id]

        data['hroi'][obs_rec_id] = obs_rec['hroi']
        data['vroi'][obs_rec_id] = obs_rec['vroi']
        data['troi'][obs_rec_id] = obs_rec['troi']
        for state_var_id in range(len(c.state_info['variables'])):
            state_vname = c.state_info['variables'][state_var_id]
            data['impact_on_state'][obs_rec_id, state_var_id] = obs_rec['impact_on_state'][state_vname]

        local_inds = c.obs_inds[obs_rec_id][par_id]
        d = len(local_inds)
        v_list = [0, 1] if obs_rec['is_vector'] else [None]
        for v in v_list:
            data['obs_rec_id'][i:i+d] = np.full(d, obs_rec_id)
            data['obs'][i:i+d] = np.squeeze(lobs[obs_rec_id][par_id]['obs'][v, :])
            data['x'][i:i+d] = lobs[obs_rec_id][par_id]['x']
            data['y'][i:i+d] = lobs[obs_rec_id][par_id]['y']
            data['z'][i:i+d] = lobs[obs_rec_id][par_id]['z'].astype(np.float32)
            data['t'][i:i+d] = np.array([t2h(t) for t in lobs[obs_rec_id][par_id]['t']])
            data['err_std'][i:i+d] = lobs[obs_rec_id][par_id]['err_std']
            for m in range(c.nens):
                data['obs_prior'][m, i:i+d] = np.squeeze(lobs_prior[m, obs_rec_id][par_id][v, :])
            i += d
    return data

def unpack_local_obs_data(c, par_id, lobs, lobs_prior, data):
    """unpack data and write back to the original lobs_prior dict"""
    n_obs_rec = len(c.obs_info['records'])
    i = 0
    for obs_rec_id in range(n_obs_rec):
        obs_rec = c.obs_info['records'][obs_rec_id]

        local_inds = c.obs_inds[obs_rec_id][par_id]
        d = len(local_inds)
        v_list = [0, 1] if obs_rec['is_vector'] else [None]
        for v in v_list:
            for m in range(c.nens):
                lobs_prior[m, obs_rec_id][par_id][v, :] = data['obs_prior'][m, i:i+d]
            i += d

def global_obs_list(c):
    ##form the global list of obs (in serial mode the main loop is over this list)
    n_obs_rec = len(c.obs_info['records'])

    i = {}  ##location in full obs vector on owner pid
    for owner_pid in range(c.nproc_mem):
        i[owner_pid] = 0

    obs_list = []
    for obs_rec_id in range(n_obs_rec):
        obs_rec = c.obs_info['records'][obs_rec_id]
        v_list = [0, 1] if obs_rec['is_vector'] else [None]
        for owner_pid in c.obs_inds[obs_rec_id].keys():
            for _ in c.obs_inds[obs_rec_id][owner_pid]:
                for v in v_list:
                    obs_list.append((obs_rec_id, v, owner_pid, i[owner_pid]))
                    i[owner_pid] += 1

    np.random.shuffle(obs_list)  ##randomize the order of obs (this is optional)
    return obs_list

###functions for the batch assimilation mode:
def batch_assim(c, state_prior, z_state, lobs, lobs_prior):
    """batch assimilation solves the matrix version EnKF analysis for each local state, the local states in each partition are processed in parallel"""
    ##pid with the most obs in its task list with show progress message
    obs_count = np.array([np.sum([len(c.obs_inds[r][p])
                                  for r in c.obs_info['records'].keys()
                                  for p in lst])
                          for lst in c.par_list.values()])
    c.pid_show = np.argsort(obs_count)[-1]
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)

    ##count number of tasks
    ntask = 0
    for par_id in c.par_list[c.pid_mem]:
        if len(c.grid.x.shape)==2:
            ist,ied,di,jst,jed,dj = c.partitions[par_id]
            msk = c.mask[jst:jed:dj, ist:ied:di]
        else:
            inds = c.partitions[par_id]
            msk = c.mask[inds]
        for l in range(np.sum((~msk).astype(int))):
            ntask += 1

    ##now the actual work starts, loop through partitions stored on pid_mem
    print_1p('>>> assimilate in batch mode:\n')
    task = 0
    for par_id in c.par_list[c.pid_mem]:
        state_data = pack_local_state_data(c, par_id, state_prior, z_state)
        nens, nfld, nloc = state_data['state_prior'].shape

        ##skip forward if the partition is empty
        if nloc == 0:
            continue

        obs_data = pack_local_obs_data(c, par_id, lobs, lobs_prior)

        nlobs = obs_data['x'].size
        impact_on_state =1

        ##if there is no obs to assimilate, update progress message and skip
        if nlobs == 0:
            task += nloc
            print_1p(progress_bar(task-1, ntask))
            continue

        ##loop through the unmasked grid points in the partition
        for l in range(nloc):
            print_1p(progress_bar(task, ntask))
            task += 1
            hdist = c.grid.distance(state_data['x'][l], obs_data['x'], state_data['y'][l], obs_data['y'], p=2)

            for n in range(nfld):
                corr = corrcoef(state_data['state_prior'][:,n,l], obs_data['obs_prior'][:,:])

            local_analysis(state_data['state_prior'][:, :, l],
                           state_data['x'][l], state_data['y'][l],
                           state_data['z'][:, l], state_data['t'][:],
                           obs_data['obs'], obs_data['err_std'],
                           obs_data['x'], obs_data['y'],
                           obs_data['z'], obs_data['t'],
                           obs_data['obs_prior'],
                           obs_data['hroi'], obs_data['vroi'],
                           obs_data['troi'], impact_on_state,
                           c.localization['htype'], c.localization['vtype'], c.localization['ttype'],
                           c.filter_type)
        unpack_local_state_data(c, par_id, state_prior, state_data)
    print_1p(' done.\n')
    return state_prior, lobs_prior

@njit
def corrcoef(x, y):
    nens, nlobs = y.shape
    x_mean = np.mean(x)
    y_mean = np.zeros(nlobs)
    for m in range(nens):
        y_mean += y[m,:]
    y_mean /= nens
    cov = np.zeros(nlobs)
    for m in range(nens):
        cov += (x[m] - x_mean) * (y[m,:] - y_mean)
    cov /= (nens -1)
    return cov

@njit(cache=True)
def local_analysis(state_prior, state_x, state_y, state_z, state_t,
                   obs, obs_err, obs_x, obs_y, obs_z, obs_t,
                   obs_prior,
                   hroi, vroi, troi, impact_on_state,
                   localize_htype, localize_vtype, localize_ttype,
                   filter_type):
    """perform local analysis for one location"""
    nens, nfld = state_prior.shape

    ##horizontal localization
    ##TODO: should use c.grid.distance
    h_dist = np.abs(obs_x - state_x) + np.abs(obs_y - state_y)
    h_lfactor = local_factor(h_dist, hroi, localize_htype)
    if (h_lfactor==0).all():
        return  ##if the state is outside of hroi of all local obs, skip

    lfactor_old = np.zeros(obs_x.size)
    weights_old = np.eye(nens)

    ##loop through the field records
    for n in range(nfld):

        ##vertical localization
        v_dist = np.abs(obs_z - state_z[n])
        v_lfactor = local_factor(v_dist, vroi, localize_vtype)
        if (v_lfactor==0).all():
            continue  ##the state is outside of vroi of all obs, skip

        ##temporal localization
        t_dist = np.abs(obs_t - state_t[n])
        t_lfactor = local_factor(t_dist, troi, localize_ttype)
        if (t_lfactor==0).all():
            continue  ##the state is outside of troi of all obs, skip

        ##total lfactor
        lfactor = h_lfactor * v_lfactor * t_lfactor #* impact_on_state[n, :]
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
        nlobs_max = None  ##3000
        ind = ind[:nlobs_max]

        ##use cached weight if no localization is applied, to avoid repeated computation
        if n>0 and len(ind)==len(lfactor_old) and (lfactor[ind]==lfactor_old).all():
            weights = weights_old
        else:
            weights = ensemble_transform_weights(obs[ind], obs_err[ind], obs_prior[:, ind], filter_type, lfactor[ind])

        ##perform local analysis and update the ensemble state
        state_prior[:, n] = apply_ensemble_transform(state_prior[:, n], weights)

        lfactor_old = lfactor[ind]
        weights_old = weights

@njit(cache=True)
def ensemble_transform_weights(obs, obs_err, obs_prior, filter_type, local_factor):
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

    if filter_type == 'ETKF':
        ##the update of ens pert is (I + S^T S)^-0.5, namely sqrt(var_ratio)
        var_ratio_sqrt = L @ np.diag(sv**-0.5) @ Rh

    elif filter_type == 'DEnKF':
        ##take Taylor approx. of var_ratio_sqrt (Sakov 2008)
        var_ratio_sqrt = np.eye(nens) - 0.5 * gain @ S

    else:
        print('Error: unknown filter type: '+filter_type)
        raise ValueError

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

###functions for the serial assimilation mode:
def serial_assim(c, state_prior, z_state, lobs, lobs_prior):
    """
    serial assimilation goes through the list of observations one by one
    for each obs the near by state variables are updated one by one.
    so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
    """
    print_1p = by_rank(c.comm, c.pid_show)(print_with_cache)
    par_id = c.pid_mem

    state_data = pack_local_state_data(c, par_id, state_prior, z_state)
    nens, nfld, nloc = state_data['state_prior'].shape

    obs_data = pack_local_obs_data(c, par_id, lobs, lobs_prior)
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
    unpack_local_state_data(c, par_id, state_prior, state_data)
    unpack_local_obs_data(c, par_id, lobs, lobs_prior, obs_data)
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
    h_lfactor = local_factor(h_dist, hroi, localize_htype)
    v_lfactor = local_factor(v_dist, vroi, localize_vtype)
    t_lfactor = local_factor(t_dist, troi, localize_ttype)

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
    h_lfactor = local_factor(h_dist, hroi, localize_htype)
    v_lfactor = local_factor(v_dist, vroi, localize_vtype)
    t_lfactor = local_factor(t_dist, troi, localize_ttype)

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

    - regress_kind: str
      Type of regression to perform, 'linear', 'probit', etc.

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

