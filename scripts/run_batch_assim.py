import numpy as np
import config as c
import assim_tools
import sys

c.nens = 16

##setup parallel communicator
c.comm = assim_tools.parallel_start()
c.pid = c.comm.Get_rank()
c.nproc = c.comm.Get_size()

##for each scale component, analysis is stored in a separate s_dir
if c.nscale > 1:
    c.s_dir = f'/scale_{c.scale+1}'
else:
    c.s_dir = ''

c.nproc_mem = 8
assert c.nproc % c.nproc_mem == 0, "nproc should be evenly divided by nproc_mem"
c.pid_mem = c.pid % c.nproc_mem
c.pid_rec = c.pid // c.nproc_mem
c.comm_mem = c.comm.Split(c.pid_rec, c.pid_mem)
c.comm_rec = c.comm.Split(c.pid_mem, c.pid_rec)

##Pre-processing
##analysis grid in 2D is prepared based on config file
##c.grid obj has information on x,y coordinates

##parse state info, for each member there will be a list of uniq fields
if c.pid == 0:
    assim_tools.message(c.comm, 'parsing config\n', 0)
    assim_tools.message(c.comm, 'ensemble size nens={}\n'.format(c.nens), 0)

    c.state_info = assim_tools.parse_state_info(c)
    assim_tools.message(c.comm, 'the number of unique field records, nrec={}\n'.format(len(c.state_info['fields'])), 0)

    mem_list_full = [mem_id for mem_id in range(c.nens)]
    c.mem_list = assim_tools.distribute_tasks(c.comm_mem, mem_list_full)

    rec_list_full = [rec_id for rec_id in c.state_info['fields'].keys()]
    c.rec_list = assim_tools.distribute_tasks(c.comm_rec, rec_list_full)

    c.fld_list = {}
    for p in range(c.nproc):
        c.fld_list[p] = [(mem_id, rec_id)
                        for mem_id in c.mem_list[p%c.nproc_mem]
                        for rec_id in c.rec_list[p//c.nproc_mem] ]

else:
    c.state_info = None
    c.mem_list = None
    c.rec_list = None
    c.fld_list = None

c.state_info = c.comm.bcast(c.state_info, root=0)
c.mem_list = c.comm.bcast(c.mem_list, root=0)
c.rec_list = c.comm.bcast(c.rec_list, root=0)
c.fld_list = c.comm.bcast(c.fld_list, root=0)

##prepare the ensemble state and z coordinates
fields, z_coords = assim_tools.prepare_state(c)

##collect ensemble mean z fields as analysis grid z coordinates
##for obs preprocessing later
zmean = assim_tools.ensemble_mean(c, z_coords)
# if c.pid == 0:
#     zmean_all = c.comm_rec.gather(zmean, root=0)
#     c.zmean = {}
#     for zdict in zmean_all:
#         for rec_id, zfld in zdict.items():
#             c.zmean[rec_id] = zfld
#     print(c.zmean.keys())

exit()
##Parse config for state and obs info, generate parallel scheme
##This is done by the first processor and broadcast
if c.pid == 0:

    c.obs_info = assim_tools.parse_obs_info(c)
    # for vname in obs_info:
        # assim_tools.message(comm, '\n', 0)

    ##generate spatial partitioning of the domain
    ##divide into square tiles with nx_tile grid points in each direction
    ##the workload on each tile is uneven since there are masked points
    ##so we divide into 3*nproc tiles so that they can be distributed
    ##according to their load (number of unmasked points)
    c.nx_tile = int(np.round(np.sqrt(c.nx * c.ny / c.nproc / 3)))

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
    # obs_info = None
    loc_list_proc = None

# obs_info = comm.bcast(obs_info, root=0)
loc_list_proc = comm.bcast(loc_list_proc, root=0)

state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/prior_state.bin'
assim_tools.output_state(c, comm, state_info, fld_list_proc, fields, state_file)

state = assim_tools.transpose_field_to_state(c, comm, state_info, fld_list_proc, loc_list_proc, fields)

obs_prior = assim_tools.prepare_obs_prior(c, comm, obs_info, )

##assimilate obs to update state
##loop through location list on pid
for loc in range(len(loc_list_proc[pid])):

    istart,iend,di,jstart,jend,dj = loc_list_proc[pid][loc]
    ii, jj = np.meshgrid(np.arange(istart,iend,di), np.arange(jstart,jend,dj))
    mask_chk = c.mask[jstart:jend:dj, istart:iend:di]

    ##loop through valid points in the fld_chk
    for l in range(len(ii[~mask_chk])):

        ##l is the index of fld_chk locally stored in state[mem_id,rec_id][loc]
        ##i,j are the indices in the full grid
        i = ii[~mask_chk][l]
        j = jj[~mask_chk][l]
        x = c.grid.x[j, i]
        y = c.grid.y[j, i]

        ##loop through each field record
        for rec_id in range(len(state_info['fields'])):
            rec = state_info['fields'][rec_id]
            name = rec['name']
            t = rec['time']
            # z = c.

            # ens_prior = np.full(c.nens, np.nan)
            # for mem_id in range(c.nens):
            #     ens_prior[mem_id] = state[mem_id, rec_id][loc][l]

            # if np.isnan(ens_prior).any():
            #     ##we don't want to update if any member is missing
            #     continue

            # ##form the obs sequence
            # for obs_rec_id in range(len(obs_info['records'])):

            #     obs_rec = obs_info['records'][obs_rec_id]
            #     ##compute obs distance to state
            #     hdist = 
            #     vdist = 
            #     tdist = 

            #     ##collect local obs seq
            #     lo_ind = 
            #     lobs = 

            #     hlfac = 
            #     vlfac = 
            #     tlfac = 
            #     impact = 

            #     ##the full localization factor
            #     local_factor = hlfac * vlfac

            #     ##obs err and correlation matrix
            #     obs_err = 
                
            #     obs['name']

            #     obs_err_corr = 

            #     ##obs prior ensemble
            #     obs_prior = 

            ##perform the batch assimilation using local_analysis
            ens_post = assim_tools.local_analysis(ens_prior, local_obs, obs_err, obs_prior, local_factor, c.filter_kind, obs_err_corr)

            ##save the posterior ensemble to the state
            for mem_id in range(c.nens):
                state[mem_id, rec_id][loc][l] = ens_post[mem_id]

fields = assim_tools.transpose_state_to_field(c, comm, state_info, fld_list_proc, loc_list_proc, state)

state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/post_state.bin'
assim_tools.output_state(c, comm, state_info, fld_list_proc, fields, state_file)

##3. Post-processing
##if c.grid != model_grid
##else
## just copy output state to model restart files


