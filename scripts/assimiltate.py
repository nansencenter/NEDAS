import os
import numpy as np
from config import Config
from utils.conversion import t2s, s2t, dt1h
from utils.parallel import by_rank, bcast_by_root
from utils.progress import timer
from assim_tools.state import parse_state_info, distribute_state_tasks, partition_grid, prepare_state, output_state, output_ens_mean
from assim_tools.obs import parse_obs_info, distribute_obs_tasks, prepare_obs, prepare_obs_from_state
# from assim_tools.transpose import transpose
# from assim_tools.analysis import batch_assim, serial_assim
# from assim_tools.update import update

c = Config(parse_args=True)

##rank 0 is going to print messages
print_by_one=by_rank(c.comm,0)(print)

print_by_one('start assimilation', flush=True)

##the algorithm can be iterated over several scale components
##for s = 0, ..., nscale
##if nscale = 1, then just
c.s_dir = ''

assim_dir = os.path.join(c.work_dir,'cycle', t2s(c.time), 'analysis', c.s_dir)

c.state_info = bcast_by_root(c.comm)(parse_state_info)(c)
c.mem_list, c.rec_list = bcast_by_root(c.comm)(distribute_state_tasks)(c)
c.partitions = bcast_by_root(c.comm)(partition_grid)(c)
fields_prior, z_fields = timer(c)(prepare_state)(c)

# mem_list, rec_list = build_state_tasks(c, state_info)
# # if c.pid_rec == 0:
# #     print('mem', c.pid_mem, mem_list[c.pid_mem])
# # if c.pid_mem == 0:
# #     print('rec', c.pid_rec, rec_list[c.pid_rec])

# if c.debug:
#     np.save(assim_dir+'/fields_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), fields_prior)

timer(c)(output_state)(c, fields_prior, os.path.join(assim_dir,'prior_state.bin'))
timer(c)(output_ens_mean)(c, fields_prior, os.path.join(assim_dir,'prior_mean_state.bin'))

# message(c.comm, 'collect model z coordinates, ', c.pid_show)
timer(c)(output_ens_mean)(c, z_fields, os.path.join(assim_dir,'z_coords.bin'))

c.obs_info = bcast_by_root(c.comm)(parse_obs_info)(c)
c.obs_rec_list = bcast_by_root(c.comm)(distribute_obs_tasks)(c)
obs_seq = timer(c)(bcast_by_root(c.comm_mem)(prepare_obs))(c)

# if c.pid_mem == 0 and c.debug:
#     np.save(assim_dir+'/obs_seq.{}.npy'.format(c.pid_rec), obs_seq)

# obs_inds = assign_obs(c, state_info, obs_info, partitions, obs_rec_list, obs_seq)

# par_list = build_par_tasks(c, partitions, obs_info, obs_inds)

# if c.pid == 0 and c.debug:
#     np.save(assim_dir+'/obs_inds.npy', obs_inds)
#     np.save(assim_dir+'/partitions.npy', partitions)
#     np.save(assim_dir+'/par_list.npy', par_list)

obs_prior_seq = timer(c)(prepare_obs_from_state)(c, obs_seq, fields_prior, z_fields)

# if c.debug:
#     np.save(assim_dir+'/obs_prior_seq.{}.{}.npy'.format(c.pid_mem, c.pid_rec), obs_prior_seq)


# message(c.comm, '3.Transpose from field-complete to ensemble-complete\n', c.pid_show)

# message(c.comm, 'state variable fields: ', c.pid_show)
# state_prior = transpose_field_to_state(c, state_info, mem_list, rec_list, partitions, par_list, fields_prior)

# message(c.comm, 'z coords fields: ', c.pid_show)
# z_state = transpose_field_to_state(c, state_info, mem_list, rec_list, partitions, par_list, z_fields)

# #global scalar state variables to be updated

# lobs = transpose_obs_to_lobs(c, mem_list, rec_list, obs_rec_list, par_list, obs_inds, obs_seq)

# lobs_prior = transpose_obs_to_lobs(c, mem_list, rec_list, obs_rec_list, par_list, obs_inds, obs_prior_seq, ensemble=True)

# message(c.comm, 'Step 3 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
# runtime = time.time()

# if c.debug:
#     np.save(assim_dir+'/state_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), state_prior)
#     np.save(assim_dir+'/z_state.{}.{}.npy'.format(c.pid_mem, c.pid_rec), z_state)
#     np.save(assim_dir+'/lobs.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs)
#     np.save(assim_dir+'/lobs_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs_prior)

# ##--------------------------
# ##4.Assimilate obs to update state variables
# message(c.comm, '4.Assimilation\n', c.pid_show)
# if c.assim_mode == 'batch':
#     assimilate = batch_assim
# elif c.assim_mode == 'serial':
#     assimilate = serial_assim

# state_post = assimilate(c, state_info, obs_info, obs_inds, partitions, par_list, rec_list, state_prior, z_state, lobs, lobs_prior)

# message(c.comm, 'Step 4 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)

##--------------------------
##5.Transposing state back to field-complete
# message(c.comm, '5.Transpose state from ensemble-complete to field-complete\n', c.pid_show)

# message(c.comm, 'state variable fields: ', c.pid_show)
# fields_post = transpose_state_to_field(c, state_info, mem_list, rec_list, partitions, par_list, state_post)

# state_file = assim_dir+'/post_state.bin'
# output_state(c, state_info, mem_list, rec_list, fields_post, state_file)
# mean_file = assim_dir+'/post_mean_state.bin'
# output_ens_mean(c, state_info, mem_list, rec_list, fields_post, mean_file)

# message(c.comm, 'Step 5 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)
# runtime = time.time()

# ##--------------------------
# ##6.Post-processing
# message(c.comm, '6.Post-processing\n', c.pid_show)

# update_restart(c, state_info, mem_list, rec_list, fields_prior, fields_post)

# ##optional: output posterior obs for diag
# # obs_post_seq = prepare_obs_from_state(c, state_info, mem_list, rec_list, obs_info, obs_list, obs_seq, fields_post, z_fields)
# # output_obs(c, obs_info, obs_post_seq)

# message(c.comm, 'Step 6 took {} seconds\n\n'.format(time.time()-runtime), c.pid_show)

# message(c.comm, 'Completed successfully. All took {} seconds\n'.format(time.time()-runtime0), c.pid_show)


