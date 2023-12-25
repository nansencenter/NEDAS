import numpy as np
import config as c
import sys
import time
from log import message
from parallel import distribute_tasks
from assim_tools import *

# assert c.assim_mode in ('batch', 'serial'), 'unknown assimilation mode: '+c.assim_mode
pstr = '. run_assim in '+c.assim_mode+' mode .'
message(c.comm, len(pstr)*'.'+'\n'+pstr+'\n'+len(pstr)*'.'+'\n\n', 0)

runtime0 = time.time()  ##start the timer

##for each scale component, analysis is stored in a separate s_dir
if c.nscale > 1:
    c.s_dir = f'/scale_{c.scale+1}'
else:
    c.s_dir = ''

##parallel scheme setup
assert c.nproc % c.nproc_mem == 0, "nproc should be evenly divided by nproc_mem"
c.pid_mem = c.pid % c.nproc_mem
c.pid_rec = c.pid // c.nproc_mem
c.comm_mem = c.comm.Split(c.pid_rec, c.pid_mem)
c.comm_rec = c.comm.Split(c.pid_mem, c.pid_rec)
message(c.comm, 'Parallel scheme: nproc = {}, nproc_mem = {}\n\n'.format(c.nproc, c.nproc_mem), 0)

##--------------------------
##1.Prepare state variables
##analysis grid in 2D is prepared based on config file
##c.grid obj has information on x,y coordinates
runtime = time.time()

##parse state info, for each member there will be a list of uniq fields
message(c.comm, '1.Prepare state variables\n', 0)
message(c.comm, 'ensemble size nens={}\n'.format(c.nens), 0)

state_info = parse_state_info(c)

field_list = build_state_tasks(c, state_info)

##prepare the ensemble state and z coordinates
fields, z_fields = prepare_state(c, state_info, field_list)

##save a copy of the prior ens and mean state
state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/prior_state.bin'
output_state(c, state_info, field_list, fields, state_file)
mean_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/prior_mean_state.bin'
output_ens_mean(c, state_info, field_list, fields, mean_file)

##collect ensemble mean z fields as analysis grid z coordinates
##for obs preprocessing later
message(c.comm, 'collect model z coordinates, ', 0)
z_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/z_coords.bin'
output_ens_mean(c, state_info, field_list, z_fields, z_file)
##topaz5 use the first member as mean z coords, so output the full z ens
##mean_z_coords will read mem_id=0 anyway
# output_state(c, state_info, field_list, z_fields, z_file)

c.comm.Barrier()
message(c.comm, 'Step 1 took {} seconds\n\n'.format(time.time()-runtime), 0)
runtime = time.time()

##--------------------------
## 2.Prepare observations
message(c.comm, '2.Prepare obs and obs priors\n', 0)

obs_info = parse_obs_info(c)

obs_list = build_obs_tasks(c, obs_info)

obs_seq = prepare_obs(c, state_info, obs_info, obs_list)

partitions = partition_grid(c)

obs_inds = assign_obs_to_loc(c, partitions, obs_info, obs_list, obs_seq)

par_list = build_loc_tasks(c, partitions, obs_info, obs_inds)

##compute obs prior, each pid compute a subset of obs
obs_prior_seq = prepare_obs_from_state(c, state_info, field_list, obs_info, obs_list, obs_seq, fields, z_fields)

c.comm.Barrier()
message(c.comm, 'Step 2 took {} seconds\n\n'.format(time.time()-runtime), 0)
runtime = time.time()

##--------------------------
##3.Transposing fields to local ensemble-complete states:
message(c.comm, '3.Transpose from field-complete to ensemble-complete\n', 0)

##the local state variables to be updated
message(c.comm, 'state variable fields: ', 0)
state_prior = transpose_field_to_state(c, state_info, field_list, partitions, par_list, fields)

##z_coords for state variables
message(c.comm, 'z coords fields: ', 0)
z_state = transpose_field_to_state(c, state_info, field_list, partitions, par_list, z_fields)

##global scalar state variables to be updated

##obs and obs_prior
message(c.comm, 'obs sequences: ', 0)
lobs = transpose_obs_to_lobs(c, obs_list, obs_inds, par_list, obs_seq)

message(c.comm, 'obs prior sequences: ', 0)
lobs_prior = transpose_obs_to_lobs(c, obs_list, obs_inds, par_list, obs_prior_seq, ensemble=True)

c.comm.Barrier()
message(c.comm, 'Step 3 took {} seconds\n\n'.format(time.time()-runtime), 0)
runtime = time.time()

##--------------------------
##4.Assimilate obs to update state variables
message(c.comm, '4.Assimilation\n', 0)
if c.assim_mode == 'batch':
    assim = batch_assim
elif c.assim_mode == 'serial':
    assim = serial_assim

state_post = assim(c, state_info, obs_info, obs_inds, partitions, par_list, state_prior, z_state, lobs, lobs_prior)

c.comm.Barrier()
message(c.comm, 'Step 4 took {} seconds\n\n'.format(time.time()-runtime), 0)
runtime = time.time()

##--------------------------
##5.Transposing state back to field-complete
message(c.comm, '5.Transpose state from ensemble-complete to field-complete\n', 0)

message(c.comm, 'state variable fields: ', 0)
fields = transpose_state_to_field(c, state_info, field_list, partitions, par_list, state_post)

state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/post_state.bin'
output_state(c, state_info, field_list, fields, state_file)

mean_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/post_mean_state.bin'
output_ens_mean(c, state_info, field_list, fields, mean_file)

c.comm.Barrier()
message(c.comm, 'Step 5 took {} seconds\n\n'.format(time.time()-runtime), 0)
runtime = time.time()

##--------------------------
##6.Post-processing
message(c.comm, '6.Post-processing\n', 0)

##if c.grid != model_grid
##else
## just copy output state to model restart files
##optional: output posterior obs for diag
# obs_post_seq = prepare_obs_from_state(c, state_info, field_list, obs_info, obs_list, obs_seq, fields, z_fields)
# output_obs(c, obs_info, obs_post_seq)

c.comm.Barrier()
message(c.comm, 'Step 6 took {} seconds\n\n'.format(time.time()-runtime), 0)

message(c.comm, 'Completed successfully. All took {} seconds\n'.format(time.time()-runtime0), 0)

