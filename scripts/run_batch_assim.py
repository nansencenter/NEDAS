import numpy as np
import config as c
import assim_tools
import sys

##setup parallel communicator
comm = assim_tools.parallel_start()
pid = comm.Get_rank()
nproc = comm.Get_size()

##for each scale component, analysis is stored in a separate s_dir
if c.nscale > 1:
    s_dir = f'/scale_{c.scale+1}'
else:
    s_dir = ''

##Pre-processing
##Parse config for state and obs info, generate parallel scheme
##This is done by the first processor and broadcast
if pid == 0:
    assim_tools.message(comm, 'parsing config\n', 0)
    assim_tools.message(comm, 'ensemble size nens={}\n'.format(c.nens), 0)

    state_info = assim_tools.parse_state_info(c)
    assim_tools.message(comm, 'the number of unique field records, nrec={}\n'.format(len(state_info['fields'])), 0)

    obs_info = assim_tools.parse_obs_info(c)
    # for vname in obs_info:
        # assim_tools.message(comm, '\n', 0)

    ##list of uniq field records with (mem_id, rec_id) for each pid to process
    fld_list = [(mem_id, rec_id)
                for mem_id in range(c.nens)
                for rec_id in state_info['fields'].keys()
               ]
    fld_list_proc = assim_tools.distribute_tasks(comm, fld_list)

    ##generate spatial partitioning of the domain
    ##divide into square tiles with nx_tile grid points in each direction
    ##the workload on each tile is uneven since there are masked points
    ##so we divide into 3*nproc tiles so that they can be distributed
    ##according to their load (number of unmasked points)
    nx_tile = int(np.round(np.sqrt(c.nx * c.ny / nproc / 3)))

    ##a list of (istart, iend, di, jstart, jend, dj) for slicing
    loc_list = [(i, np.minimum(i+nx_tile, c.nx), 1,   ##istart, iend, di
                 j, np.minimum(j+nx_tile, c.ny), 1)   ##jstart, jend, dj
                for j in np.arange(0, c.ny, nx_tile)
                for i in np.arange(0, c.nx, nx_tile)
                ]

    ##count the obs in vicinity of each tile loc
    # obs_list = {}
    # for vname in c.obs_def:
    #     obs_list[vname] = {}
    #     hroi = obs_list[vname]['hroi'] / c.dx  ##TODO: do the calculation in meters?
    #     for loc in range(len(loc_list)):
    #         ist,ied,di,jst,jed,dj = loc_list[loc]
    #         ##condition 1: within the four corner points of the tile
    #         dist = np.hypot(np.minimum(np.abs(obs_i-ist), np.abs(obs_i-ied)),
    #                         np.minimum(np.abs(obs_j-jst), np.abs(obs_j-jed)))
    #         cond1 = (dist < hroi)
    #         ##condition 2: within [ist:ied, jst-hroi:jed+hroi]
    #         cond2 = np.logical_and(np.logical_and(obs_i>=ist, obs_i<=ied), np.logical_and(obs_j>jst-hroi, obs_j<jed+hroi))
    #         ##condition 3: within [ist-hroi:ied+hroi, jst:jed]
    #         cond3 = np.logical_and(np.logical_and(obs_i>ist-hroi, obs_i<ied+hroi), np.logical_and(obs_j>=jst, obs_j<=jed))

    #         ##if any of the 3 condition satisfies, the obs is in vicinity of the tile
    #         ind = np.where(np.logical_or(cond1, np.logical_or(cond2, cond3)))
    #         obs_list[loc] = obs_id[ind]


    ##number of unmasked grid points in each loc_list item
    nlpts_loc = np.array([np.sum((~c.mask[jst:jed:dj, ist:ied:di]).astype(int)) for ist,ied,di,jst,jed,dj in loc_list])
    ##number of observations within the roi of each loc_list item
    nlobs_loc = 0

    workload = np.maximum(nlpts_loc, 1) * np.maximum(nlobs_loc, 1)

    ##distribute the loc_list according to workload for each pid
    loc_list_proc = assim_tools.distribute_tasks(comm, loc_list, workload)

else:
    state_info = None
    # obs_info = None
    fld_list_proc = None
    loc_list_proc = None

##broadcast info from pid=0 to all
state_info = comm.bcast(state_info, root=0)
# obs_info = comm.bcast(obs_info, root=0)
fld_list_proc = comm.bcast(fld_list_proc, root=0)
loc_list_proc = comm.bcast(loc_list_proc, root=0)


##2. prepare the ensemble state
fields = assim_tools.prepare_state(c, comm, state_info, fld_list_proc)



##save a copy of prior state to binary file
state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/prior_state.bin'
assim_tools.output_state(c, comm, state_info, fld_list_proc, fields, state_file)


state = assim_tools.transpose_field_to_state(c, comm, state_info, fld_list_proc, loc_list_proc, fields)


fields = assim_tools.transpose_state_to_field(c, comm, state_info, fld_list_proc, loc_list_proc, state)


state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/post_state.bin'
assim_tools.output_state(c, comm, state_info, fld_list_proc, fields, state_file)

##3. prepare obs prior ensemble

##3. sort local observations

##just read in for test now, all proc have same obs list
# obs = np.load(c.work_dir+'/analysis/'+c.time+s_dir+'/obs.npz', allow_pickle=True)['obs'].item()


##2. Assimilation
##2.1
# state = assim_tools.local_analysis(c, comm, state_info, fld_list_proc, tile_list_proc, state, obs)


# for tile_id in range(320, 325):
#     istart,iend,di, jstart,jend,dj = tile_list[tile_id]
#     obs_ind = obs_list[tile_id]

#     for i in range(istart, iend, di):
#         for j in range(jstart, jend, dj):
#             nens = c.nens
#             nlobs = len(obs_ind)
#             if nlobs > 0:
#                 loc_func = local_factor(obs_x[obs_ind], obs_y[obs_ind], c.grid.x[j, i], c.grid.y[j, i], hroi*c.grid.dx, 'GC')
#                 ind = np.where(loc_func>0)
#                 state_post[:, j, i] = update(obs[obs_ind[ind]], obs_err, obs_prior[:, obs_ind[ind]], state_prior[:, j, i], loc_func[ind])



##save the posterior state after assimilation to binary file
# state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/post_state.bin'
# assim_tools.output_state(c, comm, state_info, fld_list_proc, tile_list_proc, state, state_file)


##3. Post-processing
##if c.grid != model_grid
##else
## just copy output state to model restart files

