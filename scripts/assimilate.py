import numpy as np
import os
from config import Config
from utils.conversion import t2s, s2t, dt1h
from utils.parallel import bcast_by_root
from utils.progress import timer
from assim_tools.state import parse_state_info, distribute_state_tasks, partition_grid, prepare_state, output_state, output_ens_mean
from assim_tools.obs import parse_obs_info, distribute_obs_tasks, prepare_obs, prepare_obs_from_state, assign_obs, distribute_partitions
from assim_tools.transpose import transpose, transpose_state_to_field
from assim_tools.analysis import batch_assim, serial_assim
from assim_tools.inflation import inflate_prior_state
from assim_tools.update import update_restart

c = Config(parse_args=True)

##the algorithm can be iterated over several scale components
##for s = 0, ..., nscale
##if nscale = 1, then just
c.s_dir = ''
analysis_dir = os.path.join(c.work_dir,'cycle', t2s(c.time), 'analysis', c.s_dir)

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
#     np.save(analysis_dir+'/fields_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), fields_prior)

timer(c)(output_state)(c, fields_prior, os.path.join(analysis_dir,'prior_state.bin'))
timer(c)(output_ens_mean)(c, fields_prior, os.path.join(analysis_dir,'prior_mean_state.bin'))

timer(c)(output_ens_mean)(c, z_fields, os.path.join(analysis_dir,'z_coords.bin'))

c.obs_info = bcast_by_root(c.comm)(parse_obs_info)(c)
c.obs_rec_list = bcast_by_root(c.comm)(distribute_obs_tasks)(c)
obs_seq = timer(c)(bcast_by_root(c.comm_mem)(prepare_obs))(c)

if c.pid_mem == 0:
    np.save(analysis_dir+'/obs_seq.{}.npy'.format(c.pid_rec), obs_seq)

c.obs_inds = bcast_by_root(c.comm_mem)(assign_obs)(c, obs_seq)
c.par_list = bcast_by_root(c.comm)(distribute_partitions)(c)

# if c.pid == 0 and c.debug:
#     np.save(analysis_dir+'/obs_inds.npy', obs_inds)
#     np.save(analysis_dir+'/partitions.npy', partitions)
#     np.save(analysis_dir+'/par_list.npy', par_list)

obs_prior_seq = timer(c)(prepare_obs_from_state)(c, obs_seq, fields_prior, z_fields)

# if c.debug:
#     np.save(analysis_dir+'/obs_prior_seq.{}.{}.npy'.format(c.pid_mem, c.pid_rec), obs_prior_seq)

state_prior, z_state, lobs, lobs_prior = transpose(c, fields_prior, z_fields, obs_seq, obs_prior_seq)

# if c.debug:
#     np.save(analysis_dir+'/state_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), state_prior)
#     np.save(analysis_dir+'/z_state.{}.{}.npy'.format(c.pid_mem, c.pid_rec), z_state)
#     np.save(analysis_dir+'/lobs.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs)
#     np.save(analysis_dir+'/lobs_prior.{}.{}.npy'.format(c.pid_mem, c.pid_rec), lobs_prior)

if c.assim_mode == 'batch':
    assim = timer(c)(batch_assim)
elif c.assim_mode == 'serial':
    assim = timer(c)(serial_assim)

state_post = assim(c, state_prior, z_state, lobs, lobs_prior)

fields_post = transpose_state_to_field(c, state_post)

timer(c)(output_state)(c, fields_post, os.path.join(analysis_dir,'post_state.bin'))
timer(c)(output_ens_mean)(c, fields_post, os.path.join(analysis_dir,'post_mean_state.bin'))

timer(c)(update_restart)(c, fields_prior, fields_post)

# ##optional: output posterior obs for diag
# # obs_post_seq = prepare_obs_from_state(c, state_info, mem_list, rec_list, obs_info, obs_list, obs_seq, fields_post, z_fields)
# # output_obs(c, obs_info, obs_post_seq)


