import numpy as np
import config as c
import sys
import time
from log import message
from parallel import distribute_tasks
from assim_tools import *

debug = False

pstr = '. module_assim .'
message(c.comm, len(pstr)*'.'+'\n'+pstr+'\n'+len(pstr)*'.'+'\n\n', c.pid_show)

runtime0 = time.time()  ##start the timer

##parallel scheme setup
assert c.nproc % c.nproc_mem == 0, "nproc should be evenly divided by nproc_mem"
c.pid_mem = c.pid % c.nproc_mem
c.pid_rec = c.pid // c.nproc_mem
c.comm_mem = c.comm.Split(c.pid_rec, c.pid_mem)
c.comm_rec = c.comm.Split(c.pid_mem, c.pid_rec)
message(c.comm, 'Parallel scheme: nproc = {}, nproc_mem = {}\n\n'.format(c.nproc, c.nproc_mem), c.pid_show)

##--------------------------
##1.Prepare state variables
##analysis grid in 2D is prepared based on config file
##c.grid obj has information on x,y coordinates
runtime = time.time()

message(c.comm, '1.Prepare state variables\n', c.pid_show)
message(c.comm, 'ensemble size nens={}\n'.format(c.nens), c.pid_show)

state_info = parse_state_info(c)

mem_list, rec_list = build_state_tasks(c, state_info)
# if c.pid_rec == 0:
#     print('mem', c.pid_mem, mem_list[c.pid_mem])
# if c.pid_mem == 0:
#     print('rec', c.pid_rec, rec_list[c.pid_rec])

fields_prior, z_fields = prepare_state(c, state_info, mem_list, rec_list)

state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/prior_state.bin'
output_state(c, state_info, mem_list, rec_list, fields_prior, state_file)
mean_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/prior_mean_state.bin'
output_ens_mean(c, state_info, mem_list, rec_list, fields_prior, mean_file)

message(c.comm, 'collect model z coordinates, ', c.pid_show)
z_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/z_coords.bin'
output_ens_mean(c, state_info, mem_list, rec_list, z_fields, z_file)
##topaz5 use the first member as mean z coords, so output the full z ens
##mean_z_coords will read mem_id=0 anyway
# output_state(c, state_info, mem_list, rec_list, z_fields, z_file)

message(c.comm, 'Step 1 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
runtime = time.time()

##--------------------------
## 2.Prepare observations
message(c.comm, '2.Prepare obs and obs priors\n', c.pid_show)

obs_info = parse_obs_info(c)

obs_rec_list = build_obs_tasks(c, obs_info)

obs_seq = prepare_obs(c, state_info, obs_info, obs_rec_list)

if c.pid_mem == 0 and debug:
    np.save('obs_seq.{}.npy'.format(c.pid_rec), obs_seq)

partitions = partition_grid(c)

obs_inds = assign_obs(c, state_info, obs_info, partitions, obs_rec_list, obs_seq)

par_list = build_par_tasks(c, partitions, obs_info, obs_inds)

if c.pid == 0 and debug:
    np.save('obs_inds.npy', obs_inds)
    np.save('partitions.npy', partitions)
    np.save('par_list.npy', par_list)

obs_prior_seq = prepare_obs_from_state(c, state_info, mem_list, rec_list, obs_info, obs_rec_list, obs_seq, fields_prior, z_fields)

message(c.comm, 'Step 2 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
runtime = time.time()

##--------------------------
##3.Transposing fields to local ensemble-complete states:
message(c.comm, '3.Transpose from field-complete to ensemble-complete\n', c.pid_show)

message(c.comm, 'state variable fields: ', c.pid_show)
state_prior = transpose_field_to_state(c, state_info, mem_list, rec_list, partitions, par_list, fields_prior)

message(c.comm, 'z coords fields: ', c.pid_show)
z_state = transpose_field_to_state(c, state_info, mem_list, rec_list, partitions, par_list, z_fields)

#global scalar state variables to be updated

lobs = transpose_obs_to_lobs(c, mem_list, rec_list, obs_rec_list, par_list, obs_inds, obs_seq)

lobs_prior = transpose_obs_to_lobs(c, mem_list, rec_list, obs_rec_list, par_list, obs_inds, obs_prior_seq, ensemble=True)

message(c.comm, 'Step 3 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
runtime = time.time()

if debug:
    np.save('state_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), state_prior)
    np.save('z_state.{}.{}.npy'.format(c.pid_mem, c.pid_rec), z_state)
    np.save('lobs.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs)
    np.save('lobs_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs_prior)

##--------------------------
##4.Assimilate obs to update state variables
message(c.comm, '4.Assimilation\n', c.pid_show)
if c.assim_mode == 'batch':
    assimilate = batch_assim
elif c.assim_mode == 'serial':
    assimilate = serial_assim

state_post = assimilate(c, state_info, obs_info, obs_inds, partitions, par_list, rec_list, state_prior, z_state, lobs, lobs_prior)

message(c.comm, 'Step 4 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
runtime = time.time()

##--------------------------
##5.Transposing state back to field-complete
message(c.comm, '5.Transpose state from ensemble-complete to field-complete\n', c.pid_show)

message(c.comm, 'state variable fields: ', c.pid_show)
fields_post = transpose_state_to_field(c, state_info, mem_list, rec_list, partitions, par_list, state_post)

state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/post_state.bin'
output_state(c, state_info, mem_list, rec_list, fields_post, state_file)
mean_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/post_mean_state.bin'
output_ens_mean(c, state_info, mem_list, rec_list, fields_post, mean_file)

message(c.comm, 'Step 5 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
runtime = time.time()

##--------------------------
##6.Post-processing
message(c.comm, '6.Post-processing\n', c.pid_show)

update_restart(c, state_info, mem_list, rec_list, fields_prior_save, fields_post)

##optional: output posterior obs for diag
# obs_post_seq = prepare_obs_from_state(c, state_info, mem_list, rec_list, obs_info, obs_list, obs_seq, fields_post, z_fields)
# output_obs(c, obs_info, obs_post_seq)

message(c.comm, 'Step 6 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)

message(c.comm, 'Completed successfully. All took {} seconds\n'.format(time.time()-runtime0), c.pid_show)

