import copy
import itertools
from NEDAS.utils.call_cost import CallCost
from abc import abstractmethod
import numpy as np
from NEDAS.utils.parallel import bcast_by_root, distribute_tasks
from NEDAS.core import Context, Assimilator
from NEDAS.assim_tools.localization.distance_based import cutoff

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

    @property
    def loop_cost(self) -> CallCost:
        """
        The three per-observation calls of the serial loop, timed (NEDAS/utils/call_cost.py;
        opt-in with NEDAS_CALL_COST, otherwise the functions are called untouched). Shared by
        every serial assimilator, so the native EAKF and a compiled one are timed at the same
        boundaries; what the loop spends outside them is the broadcast, the distance
        computations and bookkeeping, which are the same Python for every backend.
        """
        cost = getattr(self, '_loop_cost', None)
        if cost is None:
            cost = self._loop_cost = CallCost()
        return cost

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
        # obs without a valid prior (e.g. outside the model domain) are left out of obs_data,
        # so the list is formed from what each owner pid has packed
        valid = c.comm_mem.allgather(c.obs.valid)
        obs_list = bcast_by_root(c.comm)(c.obs.global_obs_list)(c, valid)

        # local state points and obs binned once, for the points within reach of each obs
        h_func = c.localization_funcs['horizontal']
        reach = [cutoff(h_func, r) for r in obs_data['hroi']]
        width = max([r for r in reach if np.isfinite(r)], default=np.inf)
        state_bins = BoxBins(c.grid, state_data['x'], state_data['y'], width)
        obs_bins = BoxBins(c.grid, obs_data['x'], obs_data['y'], width)

        # point-major copies, members contiguous per point (as DART stores its copies), for the update
        # kernels; written back after the loop
        X = np.ascontiguousarray(state_data['state_prior'].transpose(2, 1, 0))
        Xs = np.ascontiguousarray(state_data['state_static'].transpose(2, 1, 0))
        Y = np.ascontiguousarray(obs_data['obs_prior'].T)
        Ys = np.ascontiguousarray(obs_data['obs_prior_static'].T)

        # go through the entire obs list, indexed by p, one scalar obs at a time
        c.total_tasks = len(obs_list)
        cost = self.loop_cost
        obs_increment = cost.wrap(self.obs_increment, 'obs_increment')
        update_local_state = cost.wrap(self.update_local_state, 'update_local_state')
        update_local_obs = cost.wrap(self.update_local_obs, 'update_local_obs')
        for p in range(len(obs_list)):
            obs_rec_id, v, owner_pid, i = obs_list[p]

            c.debug_message = f"Processing observation obs_rec_id={obs_rec_id:2}, i={i}"
            c.message = f"completed {c.current_task}/{c.total_tasks} observations."
            c.current_task = p

            # 1. if the pid owns this obs, broadcast it to all pid
            if c.pid_mem == owner_pid:
                # collect obs info
                obs_p = {}
                obs_p['prior'] = Y[i]
                obs_p['prior_static'] = Ys[i]
                for key in ('obs', 'x', 'y', 'z', 't', 'err_std'):
                    obs_p[key] = obs_data[key][i]
                for key in ('hroi', 'vroi', 'troi', 'impact_on_variable'):
                    obs_p[key] = obs_data[key][obs_rec_id]
                # mark this obs as used
                obs_data['used'][i] = True

            else:
                obs_p = None
            with cost.measure('bcast_obs'):     # where a rank waits for the slowest one
                obs_p = c.comm_mem.bcast(obs_p, root=owner_pid)

            if np.isnan(obs_p['prior']).any() or np.isnan(obs_p['prior_static']).any() or np.isnan(obs_p['obs']):
                continue

            # compute obs-space increment
            obs_incr = obs_increment(obs_p['prior'], obs_p['prior_static'], obs_p['obs'], obs_p['err_std'])

            # 2. all pid update their own locally stored state, at the candidate points ind:
            r = cutoff(h_func, obs_p['hroi'])
            with cost.measure('state_dist'):
                ind = state_bins.candidates(obs_p['x'], obs_p['y'], r)
                state_h_dist = c.grid.distance(obs_p['x'], state_data['x'][ind], obs_p['y'], state_data['y'][ind], p=2)
                state_v_dist = np.abs(obs_p['z'] - state_data['z'][:, ind])
                state_t_dist = np.abs(obs_p['t'] - state_data['t'])
            impact_per_field = obs_p['impact_on_variable'][state_data['var_id']]
            update_local_state(X, Xs,
                                    obs_p['prior'], obs_p['prior_static'], obs_incr,
                                    ind, state_h_dist, state_v_dist, state_t_dist,
                                    obs_p['hroi'], obs_p['vroi'], obs_p['troi'],
                                    c.localization_funcs['horizontal'], c.localization_funcs['vertical'], c.localization_funcs['temporal'],
                                    c.localization_funcs['correlation'], impact_per_field)

            # 3. all pid update their own locally stored obs, the candidates ind:
            with cost.measure('obs_dist'):
                ind = obs_bins.candidates(obs_p['x'], obs_p['y'], r)
                obs_h_dist = c.grid.distance(obs_p['x'], obs_data['x'][ind], obs_p['y'], obs_data['y'][ind], p=2)
                obs_v_dist = np.abs(obs_p['z'] - obs_data['z'][ind])
                obs_t_dist = np.abs(obs_p['t'] - obs_data['t'][ind])
            obs_impact = obs_data['obs_impact'][ind]
            update_local_obs(Y, Ys, obs_data['used'],
                                  obs_p['prior'], obs_p['prior_static'], obs_incr,
                                  ind, obs_h_dist, obs_v_dist, obs_t_dist,
                                  obs_p['hroi'], obs_p['vroi'], obs_p['troi'],
                                  c.localization_funcs['horizontal'], c.localization_funcs['vertical'], c.localization_funcs['temporal'],
                                  c.localization_funcs['correlation'], obs_impact)

        state_data['state_prior'][...] = X.transpose(2, 1, 0)
        obs_data['obs_prior'][...] = Y.T
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
                           ind, state_h_dist, state_v_dist, state_t_dist,
                           hroi, vroi, troi,
                           h_local_func, v_local_func, t_local_func, correlation_local_func,
                           impact_on_variable) -> None:
        """
        Update the local state vector with the analysis increments.

        Args:
            state_prior (np.ndarray): Local state vector, point-major, shape (nloc, nfld, nens)
            state_static (np.ndarray): Local state of the static members, shape (nloc, nfld, nens_static), not updated
            obs_prior (np.ndarray): Observation priors, shape (nens,)
            obs_prior_static (np.ndarray): Observation priors of the static members, shape (nens_static,)
            obs_incr (np.ndarray): Analysis increments, shape (nens,)
            ind (np.ndarray): The candidate points (indices into nloc) the distances are given for
            state_h_dist, state_v_dist, state_t_dist (np.ndarray): Distances, shapes (nind,), (nfld, nind), (nfld,)
            impact_on_variable (np.ndarray): Cross-variable localization factor per variable, shape (nfld,)
        """
        pass

    @abstractmethod
    def update_local_obs(self, obs_data, obs_data_static, used, obs_prior, obs_prior_static, obs_incr,
                         ind, h_dist, v_dist, t_dist,
                         hroi, vroi, troi,
                         h_local_func, v_local_func, t_local_func, correlation_local_func,
                         impact_on_variable) -> None:
        """
        Update the local observations with analysis increments.

        Args:
            obs_data (np.ndarray): obs prior ensemble, point-major, shape (nlobs, nens)
            obs_data_static (np.ndarray): obs priors of the static members, shape (nlobs, nens_static), not updated
            used (np.ndarray): boolean mask of already-assimilated obs
            ind (np.ndarray): The candidate obs (indices into nlobs) the distances and impact_on_variable are given for
        """
        pass


class BoxBins:
    """
    Neighbor search as DART's get_close: the points (x, y) are sorted into bins about w wide once,
    then candidates() returns the points in the bins that overlap grid.search_box, so distances are
    computed for those alone. Bins count from the grid corner; along a cyclic dim they split the
    period evenly and wrap around. Every point is a candidate when the grid gives no box.
    """
    def __init__(self, grid, x, y, w):
        self.grid = grid
        self.all = np.arange(x.size)
        self.bins = None
        if not hasattr(grid, 'search_box') or not np.isfinite(w) or w <= 0:
            return
        cyclic = grid.cyclic_dim or ''
        self.dims = []      # (origin, bin width, bins per period or 0 if not cyclic) for x and y
        for d, origin, period in (('x', grid.xmin, grid.Lx), ('y', grid.ymin, grid.Ly)):
            n = max(1, int(period // w)) if d in cyclic else 0
            self.dims.append((origin, period / n if n else w, n))
        valid = np.where(np.isfinite(x) & np.isfinite(y))[0]
        ij = np.stack((self._index(0, x[valid]), self._index(1, y[valid])), axis=1)
        keys, inv = np.unique(ij, axis=0, return_inverse=True)
        inv = inv.ravel()
        split = np.cumsum(np.bincount(inv, minlength=len(keys)))[:-1]
        self.bins = dict(zip(map(tuple, keys.tolist()), np.split(valid[np.argsort(inv, kind='stable')], split)))

    def _index(self, k, v):
        origin, w, n = self.dims[k]
        i = np.floor((v - origin) / w).astype(int)
        return i % n if n else i

    def candidates(self, ref_x, ref_y, r):
        """Indices of the points that may lie within distance r of (ref_x, ref_y)."""
        box = None if self.bins is None else self.grid.search_box(ref_x, ref_y, r)
        if box is None:
            return self.all
        ranges = []
        for k, lo, hi in ((0, box[0], box[1]), (1, box[2], box[3])):
            origin, w, n = self.dims[k]
            i0, i1 = int(np.floor((lo - origin) / w)), int(np.floor((hi - origin) / w))
            if not n:
                ranges.append(range(i0, i1 + 1))
            elif i1 - i0 + 1 >= n:      # the box spans the whole period
                ranges.append(range(n))
            else:
                ranges.append([i % n for i in range(i0, i1 + 1)])
        parts = [self.bins[ij] for ij in itertools.product(*ranges) if ij in self.bins]
        return np.concatenate(parts) if parts else self.all[:0]
