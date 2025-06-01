import copy
from abc import abstractmethod
import numpy as np
from NEDAS.utils.parallel import bcast_by_root, distribute_tasks
from NEDAS.utils.progress import progress_bar
from NEDAS.assim_tools.assimilators.base import Assimilator

class SerialAssimilator(Assimilator):
    """
    Subclass for serial assimilation algorithms
    """

    def init_partitions(self, c):
        """
        Generate spatial partitioning of the domain
        """
        if len(c.grid.x.shape) == 2:
            ny, nx = c.grid.x.shape
            ##the domain is divided into tiles, each is formed by nproc_mem elements
            ##each element is stored on a different pid_mem
            ##for each pid, its loc points cover the entire domain with some spacing

            ##list of possible factoring of nproc_mem = nx_intv * ny_intv
            ##pick the last factoring that is most 'square', so that the interval
            ##is relatively even in both directions for each pid
            nx_intv, ny_intv = [(i, int(c.nproc_mem / i))
                                for i in range(1, int(np.ceil(np.sqrt(c.nproc_mem))) + 1)
                                if c.nproc_mem % i == 0][-1]

            ##a list of (ist, ied, di, jst, jed, dj) for slicing
            ##note: we have nproc_mem entries in the list
            partitions = [(i, nx, nx_intv, j, ny, ny_intv)
                        for j in range(ny_intv) for i in range(nx_intv) ]

        else:
            npoints = c.grid.x.size
            ##just divide the list of points into nproc_mem parts, each part spanning the entire domain
            nparts = c.nproc_mem
            partitions = [np.arange(i, npoints, nparts) for i in np.arange(nparts)]

        return partitions

    def assign_obs(self, c, state, obs):
        """
        Assign the observation sequence to each partition par_id

        Args:
            c (Config): configuration object
            state (State): state object
            obs (Obs): observation object

        Returns:
            dict[obs_rec_id, dict[par_id, np.ndarray]]:
               Indices in the full obs_seq for the subset of obs that belongs to partition par_id
        """
        obs_inds_pid = {}
        for obs_rec_id in obs.obs_rec_list[c.pid_rec]:
            full_inds = np.arange(obs.obs_seq[obs_rec_id]['obs'].shape[-1])
            obs_inds_pid[obs_rec_id] = {}

            ##locality doesn't matter, we just divide obs_rec into nproc_mem parts
            inds = distribute_tasks(c.comm_mem, full_inds)
            for par_id in range(c.nproc_mem):
                obs_inds_pid[obs_rec_id][par_id] = inds[par_id]

        ##now each pid_rec has figured out obs_inds for its own list of obs_rec_ids, we
        ##gather all obs_rec_id from different pid_rec to form the complete obs_inds dict
        obs_inds = {}
        for entry in c.comm_rec.allgather(obs_inds_pid):
            for obs_rec_id, data in entry.items():
                obs_inds[obs_rec_id] = data

        return obs_inds

    def distribute_partitions(self, c, state, obs):
        ##just assign each partition to each pid, pid==par_id
        par_list = {p:np.array([p]) for p in range(c.nproc_mem)}
        return par_list

    def assimilation_algorithm(self, c, state, obs):
        """
        Implementation of the serial assimilation algorithm.

        Notes:
            serial assimilation goes through the list of observations one by one
            for each obs the near by state variables are updated one by one.
            so each update is a scalar problem, which is solved in 2 steps: obs_increment, update_ensemble
        """
        state.state_post = copy.deepcopy(state.state_prior)
        obs.lobs_post =copy.deepcopy(obs.lobs_prior)

        par_id = c.pid_mem

        state_data = state.pack_local_state_data(c, par_id, state.state_prior, state.z_state)

        obs_data = obs.pack_local_obs_data(c, state, par_id, obs.lobs, obs.lobs_prior)
        obs_list = bcast_by_root(c.comm)(obs.global_obs_list)(c)

        c.print_1p('>>> assimilate in serial mode:\n')
        ##go through the entire obs list, indexed by p, one scalar obs at a time
        for p in range(len(obs_list)):
            obs_rec_id, v, owner_pid, i = obs_list[p]

            if c.debug:
                print(f"Processing observation obs_rec_id={obs_rec_id:2}, i={i}", flush=True)
            else:
                c.print_1p(progress_bar(p, len(obs_list)))

            ##1. if the pid owns this obs, broadcast it to all pid
            if c.pid_mem == owner_pid:
                ##collect obs info
                obs_p = {}
                obs_p['prior'] = obs_data['obs_prior'][:, i]
                for key in ('obs', 'x', 'y', 'z', 't', 'err_std'):
                    obs_p[key] = obs_data[key][i]
                for key in ('hroi', 'vroi', 'troi', 'impact_on_state'):
                    obs_p[key] = obs_data[key][obs_rec_id]
                ##mark this obs as used
                obs_data['used'][i] = True

            else:
                obs_p = None
            obs_p = c.comm_mem.bcast(obs_p, root=owner_pid)

            if np.isnan(obs_p['prior']).any() or np.isnan(obs_p['obs']):
                continue

            ##compute obs-space increment
            obs_incr = self.obs_increment(obs_p['prior'], obs_p['obs'], obs_p['err_std'])

            ##2. all pid update their own locally stored state:
            state_h_dist = c.grid.distance(obs_p['x'], state_data['x'], obs_p['y'], state_data['y'], p=2)
            state_v_dist = np.abs(obs_p['z'] - state_data['z'])
            state_t_dist = np.abs(obs_p['t'] - state_data['t'])
            self.update_local_state(state_data['state_prior'], obs_p['prior'], obs_incr,
                            state_h_dist, state_v_dist, state_t_dist,
                            obs_p['hroi'], obs_p['vroi'], obs_p['troi'],
                            c.localization_funcs['horizontal'], c.localization_funcs['vertical'], c.localization_funcs['temporal'])

            ##3. all pid update their own locally stored obs:
            obs_h_dist = c.grid.distance(obs_p['x'], obs_data['x'], obs_p['y'], obs_data['y'], p=2)
            obs_v_dist = np.abs(obs_p['z'] - obs_data['z'])
            obs_t_dist = np.abs(obs_p['t'] - obs_data['t'])
            self.update_local_obs(obs_data['obs_prior'], obs_data['used'], obs_p['prior'], obs_incr,
                            obs_h_dist, obs_v_dist, obs_t_dist,
                            obs_p['hroi'], obs_p['vroi'], obs_p['troi'],
                            c.localization_funcs['horizontal'], c.localization_funcs['vertical'], c.localization_funcs['temporal'])

        state.unpack_local_state_data(c, par_id, state.state_post, state_data)
        obs.unpack_local_obs_data(c, state, par_id, obs.lobs, obs.lobs_post, obs_data)
        c.print_1p(' done.\n')

    @abstractmethod
    def obs_increment(self, obs_prior, obs, obs_err) -> np.ndarray:
        """
        Compute observation-space analysis increments.

        Args:
            obs_prior (np.ndarray): Observation priors, 1D array of type int size nens
            obs (float): The real observation value
            obs_err (float): Observation error std

        Returns:
            ndarray: observation-space analysis increments
        """
        pass

    @abstractmethod
    def update_local_state(self, state_prior, obs_prior, obs_incr,
                           state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func) -> None:
        """
        Update the local state vector with the analysis increments.

        Args:
            state_data (np.ndarray[(nens, nfld, nloc), float]): Local state vector
            obs_prior (np.ndarray[nens, float]): Observation priors
            obs_incr (np.ndarray[nens, float]): Analysis increments

        """
        pass

    @abstractmethod
    def update_local_obs(self, obs_data_prior, obs_used, obs_prior, obs_incr,
                         obs_h_dist, obs_v_dist, obs_t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func) -> None:
        """
        Update the local observations with analysis increments.

        Args:
            obs_data_prior (np.ndarray):
            obs_used (np.nd
        """
        pass
