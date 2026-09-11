import copy
from abc import abstractmethod
import numpy as np
from NEDAS.utils.parallel import bcast_by_root, distribute_tasks
from NEDAS.core import Context, Assimilator

class SerialAssimilator(Assimilator):
    """
    Subclass for serial assimilation algorithms
    """
    assim_mode = 'serial'

    def init_partitions(self, c: Context) -> list:
        """
        Generate spatial partitioning of the domain
        """
        if len(c.grid.x.shape) == 2:
            ny, nx = c.grid.x.shape
            # the domain is divided into tiles, each is formed by nproc_mem elements
            # each element is stored on a different pid_mem
            # for each pid, its loc points cover the entire domain with some spacing

            # list of possible factoring of nproc_mem = nx_intv * ny_intv
            # pick the last factoring that is most 'square', so that the interval
            # is relatively even in both directions for each pid
            nx_intv, ny_intv = [(i, int(c.config.nproc_mem / i))
                                for i in range(1, int(np.ceil(np.sqrt(c.config.nproc_mem))) + 1)
                                if c.config.nproc_mem % i == 0][-1]

            # a list of (ist, ied, di, jst, jed, dj) for slicing
            # note: we have nproc_mem entries in the list
            partitions = [(i, nx, nx_intv, j, ny, ny_intv)
                        for j in range(ny_intv) for i in range(nx_intv) ]

        else:
            npoints = c.grid.x.size
            # just divide the list of points into nproc_mem parts, each part spanning the entire domain
            nparts = c.config.nproc_mem
            partitions = [np.arange(i, npoints, nparts) for i in np.arange(nparts)]

        return partitions

    def assign_obs(self, c: Context):
        obs_inds_pid = {}
        for obs_rec_id in c.obs.obs_rec_list[c.pid_rec]:
            full_inds = np.arange(c.obs.obs_seq[obs_rec_id]['obs'].shape[-1])
            obs_inds_pid[obs_rec_id] = {}

            # locality doesn't matter, we just divide obs_rec into nproc_mem parts
            inds = distribute_tasks(c.comm_mem, full_inds)
            for par_id in range(c.config.nproc_mem):
                obs_inds_pid[obs_rec_id][par_id] = inds[par_id]

        # now each pid_rec has figured out obs_inds for its own list of obs_rec_ids, we
        # gather all obs_rec_id from different pid_rec to form the complete obs_inds dict
        obs_inds = {}
        for entry in c.comm_rec.allgather(obs_inds_pid):
            for obs_rec_id, data in entry.items():
                obs_inds[obs_rec_id] = data

        return obs_inds

    def distribute_partitions(self, c: Context):
        # just assign each partition to each pid, pid==par_id
        par_list = {p:[p] for p in range(c.config.nproc_mem)}
        return par_list

    def assimilation_algorithm(self, c: Context):
        """
        Implementation of the serial assimilation algorithm.

        Notes:
            serial assimilation goes through the list of observations one by one
            for each obs the near by state variables are updated one by one.
            so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
        """
        c.message = 'preparing...'
        c.state.state_post = copy.deepcopy(c.state.state_prior)
        c.obs.lobs_post =copy.deepcopy(c.obs.lobs_prior)

        par_id = c.pid_mem

        state_data = c.state.pack_local_state_data(c, par_id, c.state.state_prior, c.state.state_z, c.state.state_static)

        obs_data = c.obs.pack_local_obs_data(c, par_id, c.obs.lobs, c.obs.lobs_prior, c.obs.lobs_prior_static)
        obs_list = bcast_by_root(c.comm)(c.obs.global_obs_list)(c)

        # ens-complete pre transforms (probit)
        self.transform_ens_state_forward(state_data)
        self.transform_ens_obs_forward(obs_data)

        # go through the entire obs list, indexed by p, one scalar obs at a time
        c.total_tasks = len(obs_list)
        for p in range(len(obs_list)):
            obs_rec_id, v, owner_pid, i = obs_list[p]

            c.debug_message = f"Processing observation obs_rec_id={obs_rec_id:2}, i={i}"
            c.message = f"completed {c.current_task}/{c.total_tasks} observations."
            c.current_task = p

            # 1. if the pid owns this obs, broadcast it to all pid
            if c.pid_mem == owner_pid:
                # collect obs info
                obs_p = {}
                obs_p['prior'] = obs_data['obs_prior'][:, i]
                obs_p['prior_static'] = obs_data['obs_prior_static'][:, i]
                for key in ('obs', 'x', 'y', 'z', 't', 'err_std'):
                    obs_p[key] = obs_data[key][i]
                for key in ('hroi', 'vroi', 'troi', 'impact_on_variable'):
                    obs_p[key] = obs_data[key][obs_rec_id]
                # mark this obs as used
                obs_data['used'][i] = True

            else:
                obs_p = None
            obs_p = c.comm_mem.bcast(obs_p, root=owner_pid)

            if np.isnan(obs_p['prior']).any() or np.isnan(obs_p['prior_static']).any() or np.isnan(obs_p['obs']):
                continue

            # compute obs-space increment
            obs_incr = self.obs_increment(obs_p['prior'], obs_p['prior_static'], obs_p['obs'], obs_p['err_std'])

            # 2. all pid update their own locally stored state:
            state_h_dist = c.grid.distance(obs_p['x'], state_data['x'], obs_p['y'], state_data['y'], p=2)
            state_v_dist = np.abs(obs_p['z'] - state_data['z'])
            state_t_dist = np.abs(obs_p['t'] - state_data['t'])
            impact_per_field = obs_p['impact_on_variable'][state_data['var_id']]
            self.update_local_state(state_data['state_prior'], state_data['state_static'],
                                    obs_p['prior'], obs_p['prior_static'], obs_incr,
                                    state_h_dist, state_v_dist, state_t_dist,
                                    obs_p['hroi'], obs_p['vroi'], obs_p['troi'],
                                    c.localization_funcs['horizontal'], c.localization_funcs['vertical'], c.localization_funcs['temporal'],
                                    impact_per_field)

            # 3. all pid update their own locally stored obs:
            obs_h_dist = c.grid.distance(obs_p['x'], obs_data['x'], obs_p['y'], obs_data['y'], p=2)
            obs_v_dist = np.abs(obs_p['z'] - obs_data['z'])
            obs_t_dist = np.abs(obs_p['t'] - obs_data['t'])
            obs_impact = obs_data['obs_impact']
            self.update_local_obs(obs_data['obs_prior'], obs_data['obs_prior_static'], obs_data['used'],
                                  obs_p['prior'], obs_p['prior_static'], obs_incr,
                                  obs_h_dist, obs_v_dist, obs_t_dist,
                                  obs_p['hroi'], obs_p['vroi'], obs_p['troi'],
                                  c.localization_funcs['horizontal'], c.localization_funcs['vertical'], c.localization_funcs['temporal'],
                                  obs_impact)

        # ens-complete inverse transforms (probit)
        self.transform_ens_state_backward(state_data)
        self.transform_ens_obs_backward(obs_data)

        c.state.unpack_local_state_data(c, par_id, c.state.state_post, state_data)
        c.obs.unpack_local_obs_data(c, par_id, c.obs.lobs, c.obs.lobs_post, obs_data)

    @abstractmethod
    def obs_increment(self, obs_prior, obs_prior_static, obs, obs_err) -> np.ndarray:
        """
        Compute observation-space analysis increments.

        Args:
            obs_prior (np.ndarray): Observation priors, 1-D float array of length nens
            obs_prior_static (np.ndarray): Observation priors of the static members (covariance_def.nens_static)
            obs (float): The real observation value
            obs_err (float): Observation error std

        Returns:
            ndarray: observation-space analysis increments
        """
        pass

    @abstractmethod
    def update_local_state(self, state_prior, state_static, obs_prior, obs_prior_static, obs_incr,
                           state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func,
                           impact_on_variable) -> None:
        """
        Update the local state vector with the analysis increments.

        Args:
            state_prior (np.ndarray): Local state vector, shape (nens, nfld, nloc)
            state_static (np.ndarray): Local state of the static members, shape (nens_static, nfld, nloc), not updated
            obs_prior (np.ndarray): Observation priors, shape (nens,)
            obs_prior_static (np.ndarray): Observation priors of the static members, shape (nens_static,)
            obs_incr (np.ndarray): Analysis increments, shape (nens,)
            impact_on_variable (np.ndarray): Cross-variable localization factor per variable, shape (nfld,)
        """
        pass

    @abstractmethod
    def update_local_obs(self, obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                         h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func,
                         impact_on_variable) -> None:
        """
        Update the local observations with analysis increments.

        Args:
            obs_data (np.ndarray): obs prior ensemble, shape (nens, nlobs)
            obs_data_static (np.ndarray): obs priors of the static members, shape (nens_static, nlobs), not updated
            used (np.ndarray): boolean mask of already-assimilated obs
        """
        pass

    def transform_ens_state_forward(self, state_data):
        pass

    def transform_ens_state_backward(self, state_data):
        pass

    def transform_ens_obs_forward(self, obs_data):
        pass

    def transform_ens_obs_backward(self, obs_data):
        pass

