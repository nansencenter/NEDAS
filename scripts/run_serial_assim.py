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

assert c.assim_mode == 'serial', 'assimilation mode error, got '+c.assim_mode

##1. Pre-processing
##1.1 parse config and generate state and obs info
##this is done by the first processor and broadcast
assim_tools.message(comm, 'parsing state and obs info from config\n', 0)
if pid == 0:
    state_info = assim_tools.parse_state_info(c, comm)
    obs_info = assim_tools.parse_obs_info(c)
else:
    state_info = None
    obs_info = None
state_info = comm.bcast(state_info, root=0)
obs_info = comm.bcast(obs_info, root=0)

##1.2 list of uniq field records with (mem_id, rec_id) for each pid to process
fld_list = [(mem_id, rec_id) for mem_id in range(c.nens) for rec_id in state_info['fields'].keys()])
fld_list_proc = assim_tools.distribute_tasks(comm, fld_list)

##1.3 generate spatial partitioning of the domain
##the domain is divided into tiles, each tile is formed by all different pids
##for each pid, its subset points still cover the entire domain with some spatial intervals
##try to factor nproc into (nx_intv x ny_intv) so that the tile is relatively 'square' and spacing between points
##for each pid is relatively even in both directions
nx_intv, ny_intv = [(i, int(nproc / i)) for i in range(1, int(np.ceil(np.sqrt(nproc))) + 1) if nproc % i == 0][-1]

##a list of (ist, ied, di, jst, jed, dj) for slicing
loc_list = [(i, nx, nx_intv, j, ny, ny_intv) for j in np.arange(ny_intv) for i in np.arange(nx_intv)]

obs_list = {}


##1.4 prepare observations

#obs_seq_file = c.work_dir+'/analysis/'+c.time+s_dir+'/obs_seq.bin'



##number of valid grid points in each loc_list item
nlpts_loc = np.array([np.sum((~c.mask[jst:jed:dj, ist:ied:di]).astype(int)) for ist,ied,di,jst,jed,dj in loc_list])
##number of observations within the roi of each loc_list item
nlobs_loc = 0

load_loc = np.maximum(nlpts_loc, 1) * np.maximum(nlobs_loc, 1)


##distribute the workload for each pid
loc_list_proc = assim_tools.distribute_tasks(comm, loc_list, load_loc)


##2. prepare state
state = assim_tools.prepare_state(c, comm, state_info)

##save a copy of prior state to binary file
# state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/prior_state.bin'
##if c.grid != model_grid
# assim_tools.output_state(c, comm, state_info, fld_list_proc, tile_list_proc, state, state_file)

##3. sort local observations

##just read in for test now, all proc have same obs list
obs = np.load(c.work_dir+'/analysis/'+c.time+s_dir+'/obs.npz', allow_pickle=True)['obs'].item()


##2. Assimilation
##2.1
state = assim_tools.local_analysis(c, comm, state_info, fld_list_proc, tile_list_proc, state, obs)


##save the posterior state after assimilation to binary file
state_file = c.work_dir+'/analysis/'+c.time+s_dir+'/post_state.bin'
assim_tools.output_state(c, comm, state_info, fld_list_proc, tile_list_proc, state, state_file)


##3. Post-processing
##if c.grid != model_grid
##else
## just copy output state to model restart files

