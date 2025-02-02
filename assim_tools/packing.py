"""pack/unpack local state and obs data for jitted functions"""

import numpy as np
from utils.conversion import t2h

def pack_state_data(c, par_id, state_prior, z_state):
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

def unpack_state_data(c, par_id, state_prior, data):
    """unpack data and write back to the original state_prior dict"""
    nfld = len(data['fields'])
    nloc = len(data['x'])

    for m in range(c.nens):
        for n in range(nfld):
            rec_id, v = data['fields'][n]
            state_prior[m, rec_id][par_id][v, :] = data['state_prior'][m, n, :]

def pack_obs_data(c, par_id, lobs, lobs_prior):
    """pack lobs and lobs_prior into arrays for the jitted functions"""
    data = {}

    ##number of local obs on partition
    nlobs = np.sum([lobs[r][par_id]['obs'].size for r in c.obs_info['records'].keys()])
    n_obs_rec = len(c.obs_info['records'])        ##number of obs records
    n_state_var = len(c.state_info['variables'])  ##number of state variable names

    data['obs_rec_id'] = np.zeros(nlobs, dtype=int)
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
            data['obs_rec_id'][i:i+d] = obs_rec_id
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

def unpack_obs_data(c, par_id, lobs, lobs_prior, data):
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
