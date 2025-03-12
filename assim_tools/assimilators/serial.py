import numpy as np
from utils.parallel import by_rank, bcast_by_root
from utils.njit import njit
from utils.progress import print_with_cache, progress_bar
#TODO from utils.distribution import normal_cdf, inv_weighted_normal_cdf
from ..obs import global_obs_list
from ..packing import pack_state_data, unpack_state_data, pack_obs_data, unpack_obs_data

class SerialAssimilator:

    def assimilate(self, c, state_prior, z_state, lobs, lobs_prior):
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
            obs_incr = self.obs_increment(obs['prior'], obs['obs'], obs['err_std'], c.filter_type)

            ##2. all pid update their own locally stored state:
            state_h_dist = c.grid.distance(obs['x'], state_data['x'], obs['y'], state_data['y'], p=2)
            state_v_dist = np.abs(obs['z'] - state_data['z'])
            state_t_dist = np.abs(obs['t'] - state_data['t'])
            self.update_local_state(state_data['state_prior'], obs['prior'], obs_incr,
                            state_h_dist, state_v_dist, state_t_dist,
                            obs['hroi'], obs['vroi'], obs['troi'],
                            c.localization['htype'], c.localization['vtype'], c.localization['ttype'])

            ##3. all pid update their own locally stored obs:
            obs_h_dist = c.grid.distance(obs['x'], obs_data['x'], obs['y'], obs_data['y'], p=2)
            obs_v_dist = np.abs(obs['z'] - obs_data['z'])
            obs_t_dist = np.abs(obs['t'] - obs_data['t'])
            self.update_local_obs(obs_data['obs_prior'], obs_data['used'], obs['prior'], obs_incr,
                            obs_h_dist, obs_v_dist, obs_t_dist,
                            obs['hroi'], obs['vroi'], obs['troi'],
                            c.localization['htype'], c.localization['vtype'], c.localization['ttype'])
        unpack_state_data(c, par_id, state_prior, state_data)
        unpack_obs_data(c, par_id, lobs, lobs_prior, obs_data)
        print_1p(' done.\n')
        return state_prior, lobs_prior

    def obs_increment(self):
        raise NotImplementedError

    def update_local_state(self):
        raise NotImplementedError

    def update_local_obs(self):
        raise NotImplementedError
