import numpy as np
from utils.parallel import by_rank, bcast_by_root
from utils.njit import njit
from utils.progress import print_with_cache, progress_bar
from ..packing import pack_state_data, unpack_state_data, pack_obs_data, unpack_obs_data
from ..localization import local_factor_distance_based

class BatchAssimilator:

    def assimilate(self, c, state_prior, z_state, lobs, lobs_prior):
        """Batch assimilation solves the matrix version EnKF analysis for each local state,
        the local states in each partition are processed in parallel
        Parameters:
        - c: config object
        - state_prior: np.array[nens, nfld, nloc]
        - z_state: np.array[nfld, nloc]
        - lobs: list of obs objects
        - lobs_prior: list of obs objects
        Returns:
        - state_post: np.array[nens, nfld, nloc], updated version of state_prior
        - lobs_post: list of obs objects, updated version of lobs_prior
        """
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
            for loc_id in range(np.sum((~msk).astype(int))):
                ntask += 1

        ##now the actual work starts, loop through partitions stored on pid_mem
        print_1p('>>> assimilate in batch mode:\n')
        task = 0
        for par_id in c.par_list[c.pid_mem]:
            state_data = pack_state_data(c, par_id, state_prior, z_state)
            nloc = state_data['state_prior'].shape[-1]
            ##skip forward if the partition is empty
            if nloc == 0:
                continue

            obs_data = pack_obs_data(c, par_id, lobs, lobs_prior)
            nlobs = obs_data['x'].size
            ##if there is no obs to assimilate, update progress message and skip that partition
            if nlobs == 0:
                task += nloc
                if c.debug:
                    print(f"PID {c.pid:4} processed partition {par_id:7} (which is empty)", flush=True)
                else:
                    print_1p(progress_bar(task-1, ntask))
                continue

            ##loop through the unmasked grid points in the partition
            for loc_id in range(nloc):
                ##state variable metadata for this location
                state_var_id = state_data['var_id']  ##variable id for each field (nfld)
                state_x = state_data['x'][loc_id]
                state_y = state_data['y'][loc_id]
                state_z = state_data['z'][:, loc_id]
                state_t = state_data['t'][:]

                ##filter out obs outside the hroi in each direction first (using L1 norm to speed up)
                obs_rec_id = obs_data['obs_rec_id']
                hroi = obs_data['hroi'][obs_rec_id]
                hdist = c.grid.distance(state_x, obs_data['x'], state_y, obs_data['y'], p=1)
                ind = np.where(hdist<=hroi)[0]

                ##compute horizontal localization factor (using L2 norm for distance)
                obs_rec_id = obs_data['obs_rec_id'][ind]
                hroi = obs_data['hroi'][obs_rec_id]
                hdist = c.grid.distance(state_x, obs_data['x'][ind], state_y, obs_data['y'][ind], p=2)
                hlfactor = local_factor_distance_based(hdist, hroi, c.localization['htype'])
                ind1 = np.where(hlfactor>0)[0]
                ind = ind[ind1]
                hlfactor = hlfactor[ind1]

                if len(ind1) == 0:
                    if c.debug:
                        print(f"PID {c.pid:4} processed partition {par_id:7} grid point {loc_id} (all local obs outside hroi)", flush=True)
                    else:
                        print_1p(progress_bar(task, ntask))
                    continue ##if all obs has no impact on state, just skip to next location
                
                ##vertical, time and cross-variable (impact_on_state) localization
                obs = obs_data['obs'][ind]
                obs_err = obs_data['err_std'][ind]
                obs_z = obs_data['z'][ind]
                obs_t = obs_data['t'][ind]
                obs_rec_id = obs_data['obs_rec_id'][ind]
                vroi = obs_data['vroi'][obs_rec_id]
                troi = obs_data['troi'][obs_rec_id]
                impact_on_state = obs_data['impact_on_state'][:, state_var_id][obs_rec_id]

                ##covariance between state and obs
                # stateV, obsV, corr = covariance(c.covariance, state_prior, obs_prior, state_var_id, obs_rec_id, h_dist, v_dist, t_dist)

                self.local_analysis(state_data['state_prior'][...,loc_id], obs_data['obs_prior'][:,ind],
                            obs, obs_err, hlfactor,
                            state_z, obs_z, vroi, c.localization['vtype'],
                            state_t, obs_t, troi, c.localization['ttype'],
                            impact_on_state, c.filter_type,
                            c.rfactor, c.kfactor, c.nlobs_max)

                ##add progress message
                if c.debug:
                    print(f"PID {c.pid:4} processed partition {par_id:7} grid point {loc_id}", flush=True)
                else:
                    print_1p(progress_bar(task, ntask))
                task += 1

            unpack_state_data(c, par_id, state_prior, state_data)
        print_1p(' done.\n')
        return state_prior, lobs_prior

    def local_analysis(self):
        """Local analysis scheme for each model state variable (grid point)
        to be implemented by derived classes"""
        raise NotImplementedError
