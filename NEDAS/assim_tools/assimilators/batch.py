import copy
from abc import abstractmethod
import numpy as np
from NEDAS.utils.parallel import distribute_tasks
from NEDAS.utils.progress import progress_bar
from NEDAS.assim_tools.assimilators.base import Assimilator

class BatchAssimilator(Assimilator):
    def init_partitions(self, c):
        """
        Generate spatial partitioning of the domain
        partitions: dict[par_id, tuple(istart, iend, di, jstart, jend, dj)]
        for each partition indexed by par_id, the tuple contains indices for slicing the domain
        Using regular slicing is more efficient than fancy indexing (used in irregular grid)
        """
        if len(c.grid.x.shape) == 2:
            ny, nx = c.grid.x.shape

            ##divide into square tiles with nx_tile grid points in each direction
            ##the workload on each tile is uneven since there are masked points
            ##so we divide into 3*nproc tiles so that they can be distributed
            ##according to their load (number of unmasked points)
            ntile = c.nproc_mem * 3
            nx_tile = np.maximum(int(np.round(np.sqrt(nx * ny / ntile))), 1)

            ##a list of (istart, iend, di, jstart, jend, dj) for tiles
            ##note: we have 3*nproc entries in the list
            partitions = [(i, np.minimum(i+nx_tile, nx), 1,   ##istart, iend, di
                           j, np.minimum(j+nx_tile, ny), 1)   ##jstart, jend, dj
                          for j in np.arange(0, ny, nx_tile)
                          for i in np.arange(0, nx, nx_tile) ]

        else:
            npoints = c.grid.x.size
            ##divide the domain into sqaure tiles, similar to regular_grid case, but collect
            ##the grid points inside each tile and return the indices
            ntile = c.nproc_mem * 3

            if c.grid.Ly==0:
                ##for 1D grid, just divide into equal sections, no y dimension
                Dx = c.grid.Lx / ntile
                partitions = [np.where(np.logical_and(c.grid.x>=x, c.grid.x<x+Dx))[0]
                              for x in np.arange(c.grid.xmin, c.grid.xmax, Dx)]

            else:
                ##for 2D grid, find number of tiles in each direction according to aspect ratio
                ntile_y = max(int(np.sqrt(ntile * c.grid.Ly / c.grid.Lx)), 1)
                ntile_x = max(ntile // ntile_y, 1)
                Dx = c.grid.Lx / ntile_x
                Dy = c.grid.Ly / ntile_y
                partitions = [np.where(np.logical_and(np.logical_and(c.grid.x>=x, c.grid.x<x+Dx),
                                                      np.logical_and(c.grid.y>=y, c.grid.y<y+Dy)))
                              for y in np.arange(c.grid.ymin, c.grid.ymax, Dy)
                              for x in np.arange(c.grid.xmin, c.grid.xmax, Dx)]

        return partitions

    def assign_obs(self, c, state, obs):
        """
        Assign the observation sequence to each partition par_id

        Returns:
        - obs_inds: dict[obs_rec_id, dict[par_id, inds]]
        where inds is np.array with indices in the full obs_seq, for the subset of obs
        that belongs to partition par_id
        """
        ##each pid_rec has a subset of obs_rec_list
        obs_inds_pid = {}
        for obs_rec_id in obs.obs_rec_list[c.pid_rec]:
            ##screen horizontally for obs inside hroi of each partition
            obs_inds_pid[obs_rec_id] = self.assign_obs_to_tiles(c, state, obs, obs_rec_id)

        ##now each pid_rec has figured out obs_inds for its own list of obs_rec_ids, we
        ##gather all obs_rec_id from different pid_rec to form the complete obs_inds dict
        obs_inds = {}
        for entry in c.comm_rec.allgather(obs_inds_pid):
            for obs_rec_id, data in entry.items():
                obs_inds[obs_rec_id] = data

        return obs_inds

    def assign_obs_to_tiles(self, c, state, obs, obs_rec_id):
        hroi = obs.info['records'][obs_rec_id]['hroi']

        xo = np.array(obs.obs_seq[obs_rec_id]['x'])  ##obs x,y
        yo = np.array(obs.obs_seq[obs_rec_id]['y'])

        ##loop over partitions with par_id
        obs_inds = {}
        for par_id in range(len(state.partitions)):
            ##find bounding box for this partition
            if len(c.grid.x.shape)==2:
                ist,ied,di,jst,jed,dj = state.partitions[par_id]
                xmin, xmax, ymin, ymax = c.grid.x[0,ist], c.grid.x[0,ied-1], c.grid.y[jst,0], c.grid.y[jed-1,0]
            else:
                inds = state.partitions[par_id]
                x = c.grid.x[inds]
                y = c.grid.y[inds]
                xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
            Dx = 0.5 * (xmax - xmin)
            Dy = 0.5 * (ymax - ymin)
            xc = xmin + Dx
            yc = ymin + Dy

            ##observations within the bounding box + halo region of width hroi will be assigned to
            ##this partition. Although this will include some observations near the corner that are
            ##not within hroi of any grid points, this is favorable for the efficiency in finding subset
            obs_inds[par_id] = np.where(np.logical_and(c.grid.distance(xc, xo, yc, yc, p=1) <= Dx+hroi,
                                                       c.grid.distance(xc, xc, yc, yo, p=1) <= Dy+hroi))[0]

        return obs_inds

    def distribute_partitions(self, c, state, obs):
        par_list_full = np.arange(len(state.partitions))

        ##distribute the list of par_id according to workload to each pid
        ##number of unmasked grid points in each tile
        if len(c.grid.x.shape) == 2:
            nlpts_loc = np.array([np.sum((~c.grid.mask[jst:jed:dj, ist:ied:di]).astype(int))
                                 for ist,ied,di,jst,jed,dj in state.partitions] )
        else:
            nlpts_loc = np.array([np.sum((~c.grid.mask[inds]).astype(int))
                                 for inds in state.partitions] )

        ##number of observations within the hroi of each tile, at loc,
        ##sum over the len of obs_inds for obs_rec_id over all obs_rec_ids
        nlobs_loc = np.array([np.sum([len(obs.obs_inds[r][p])
                                      for r in obs.info['records'].keys()])
                              for p in par_list_full] )

        workload = np.maximum(nlpts_loc, 1) * np.maximum(nlobs_loc, 1)
        par_list = distribute_tasks(c.comm_mem, par_list_full, workload)

        return par_list

    def assimilation_algorithm(self, c, state, obs):
        """
        batch assimilation solves the matrix version EnKF analysis for each local state,
        the local states in each partition are processed in parallel
        """
        state.state_post = copy.deepcopy(state.state_prior)
        obs.lobs_post = copy.deepcopy(obs.lobs_prior)

        ##pid with the most obs in its task list with show progress message
        obs_count = np.array([np.sum([len(obs.obs_inds[r][p])
                                      for r in obs.info['records'].keys()
                                      for p in lst])
                              for lst in state.par_list.values()])
        c.pid_show = np.argsort(obs_count)[-1]

        ##count number of tasks
        ntask = 0
        for par_id in state.par_list[c.pid_mem]:
            if len(c.grid.x.shape)==2:
                ist,ied,di,jst,jed,dj = state.partitions[par_id]
                msk = c.grid.mask[jst:jed:dj, ist:ied:di]
            else:
                inds = state.partitions[par_id]
                msk = c.grid.mask[inds]
            for loc_id in range(np.sum((~msk).astype(int))):
                ntask += 1

        ##now the actual work starts, loop through partitions stored on pid_mem
        c.print_1p('>>> assimilate in batch mode:\n')
        task = 0
        for par_id in state.par_list[c.pid_mem]:
            state_data = state.pack_local_state_data(c, par_id, state.state_prior, state.z_state)
            nloc = state_data['state_prior'].shape[-1]
            ##skip forward if the partition is empty
            if nloc == 0:
                continue

            obs_data = obs.pack_local_obs_data(c, state, par_id, obs.lobs, obs.lobs_prior)
            nlobs = obs_data['x'].size
            ##if there is no obs to assimilate, update progress message and skip that partition
            if nlobs == 0:
                task += nloc
                if c.debug:
                    print(f"PID {c.pid:4} processed partition {par_id:7} (which is empty)", flush=True)
                else:
                    c.print_1p(progress_bar(task-1, ntask))
                continue

            ##loop through the unmasked grid points in the partition
            for loc_id in range(nloc):
                ##state variable metadata for this location
                state_x = state_data['x'][loc_id]
                state_y = state_data['y'][loc_id]

                ##filter out obs outside the hroi in each direction first (using L1 norm to speed up)
                obs_rec_id = obs_data['obs_rec_id']
                hroi = obs_data['hroi'][obs_rec_id]
                hdist = c.grid.distance(state_x, obs_data['x'], state_y, obs_data['y'], p=1)
                ind = np.where(hdist<=hroi)[0]

                ##TODO: filter out nan in obs_prior
                #obs_data['obs_prior']

                ##compute horizontal localization factor (using L2 norm for distance)
                obs_rec_id = obs_data['obs_rec_id'][ind]
                hroi = obs_data['hroi'][obs_rec_id]
                hdist = c.grid.distance(state_x, obs_data['x'][ind], state_y, obs_data['y'][ind], p=2)
                hlfactor = c.localization_funcs['horizontal'](hdist, hroi)
                ind1 = np.where(hlfactor>0)[0]
                ind = ind[ind1]
                hlfactor = hlfactor[ind1]

                if len(ind1) == 0:
                    if c.debug:
                        print(f"PID {c.pid:4} processed partition {par_id:7} grid point {loc_id} (all local obs outside hroi)", flush=True)
                    else:
                        c.print_1p(progress_bar(task, ntask))
                    continue ##if all obs has no impact on state, just skip to next location

                self.local_analysis(c, loc_id, ind, hlfactor, state_data, obs_data)

                ##add progress message
                if c.debug:
                    print(f"PID {c.pid:4} processed partition {par_id:7} grid point {loc_id}", flush=True)
                else:
                    c.print_1p(progress_bar(task, ntask))
                task += 1

            state.unpack_local_state_data(c, par_id, state.state_post, state_data)
            obs.unpack_local_obs_data(c, state, par_id, obs.lobs, obs.lobs_post, obs_data)
        c.print_1p(' done.\n')

    @abstractmethod
    def local_analysis(self, c, loc_id, ind, hlfactor, state_data, obs_data):
        """Local analysis scheme for each model state variable (grid point)
        to be implemented by derived classes"""
        pass