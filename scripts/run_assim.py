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

##for each scale component, analysis is stored in a separate s_dir
if c.nscale > 1:
    c.s_dir = f'/scale_{c.scale+1}'
else:
    c.s_dir = ''

##--------------------------
##1.Prepare state variables
##analysis grid in 2D is prepared based on config file
##c.grid obj has information on x,y coordinates
start = time.time()

##parse state info, for each member there will be a list of uniq fields
message(c.comm, '1.Prepare state variables\n', 0)
message(c.comm, 'ensemble size nens={}\n'.format(c.nens), 0)

state_info = parse_state_info(c)

field_list = build_state_tasks(c, state_info)

##prepare the ensemble state and z coordinates
fields, z_fields = prepare_state(c, state_info, field_list)

##save a copy of the prior state
state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/prior_state.bin'
output_state(c, state_info, field_list, fields, state_file)

##collect ensemble mean z fields as analysis grid z coordinates
##for obs preprocessing later
message(c.comm, 'collect model z coordinates, ', 0)
z_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/z_coords.bin'
output_ens_mean(c, state_info, field_list, z_fields, z_file)

message(c.comm, 'Step 1 took {} seconds\n\n'.format(time.time()-start), 0)
start = time.time()

##--------------------------
## 2.Prepare observations
message(c.comm, '2.Prepare obs and obs priors\n', 0)

obs_info = parse_obs_info(c)

obs_list = build_obs_tasks(c, obs_info)

obs_seq = prepare_obs(c, state_info, obs_info, obs_list)

loc_list_full = partition_grid(c)

obs_inds = assign_obs_to_loc(c, loc_list_full, obs_info, obs_list, obs_seq)

loc_list = build_loc_tasks(c, loc_list_full, obs_info, obs_inds)

##compute obs prior, each pid compute a subset of obs
obs_prior_seq = prepare_obs_prior(c, state_info, field_list, obs_info, obs_list, obs_seq, fields, z_fields)

message(c.comm, 'Step 2 took {} seconds\n\n'.format(time.time()-start), 0)
start = time.time()

##--------------------------
##3.Transposing fields to local ensemble-complete states:
message(c.comm, '3.Transpose from field-complete to ensemble-complete\n', 0)

##the local state variables to be updated
message(c.comm, 'state variable fields: ', 0)
state = transpose_field_to_state(c, state_info, field_list, loc_list, fields)

##z_coords for state variables
message(c.comm, 'z coords fields: ', 0)
z_state = transpose_field_to_state(c, state_info, field_list, loc_list, z_fields)

##global scalar state variables to be updated

##obs and obs_prior
message(c.comm, 'obs sequences: ', 0)
lobs = transpose_obs_to_lobs(c, obs_list, obs_inds, loc_list, obs_seq)

message(c.comm, 'obs prior sequences: ', 0)
lobs_prior = transpose_obs_to_lobs(c, obs_list, obs_inds, loc_list, obs_prior_seq, ensemble=True)

message(c.comm, 'Step 3 took {} seconds\n\n'.format(time.time()-start), 0)
start = time.time()


##--------------------------
##4.Assimilate obs to update state variables
message(c.comm, '4.Assimilation\n', 0)

if c.assim_mode == 'batch':

    ##loop through tiles stored on pid
    for loc in range(len(loc_list[c.pid])):
        ist,ied,di,jst,jed,dj = loc_list[c.pid][loc]
        ii, jj = np.meshgrid(np.arange(ist,ied,di), np.arange(jst,jed,dj))
        mask_chk = c.mask[jst:jed:dj, ist:ied:di]

        ##fetch local obs seq and obs_prior seq on tile
        # obs = []
        # obs_prior = []
        # obs_err = []
        # nlobs = 0
        # for obs_rec_id, obs_rec in obs_info['records'].items():
        #     if obs_rec['is_vector']:
        #         for v in range(2):
        #             obs.append(lobs[obs_rec_id][loc]['obs'][v, :])
        #     else:
        #         obs.append(lobs[obs_rec_id][loc]['obs'])

        #     #obs err and correlation matrix
        #     obs_err.append([obs_rec['err']['std'] for i in range(len(obs))])
        #     nlobs += len(obs)
        # obs_err_corr = np.eye(nlobs)
        # print(obs)

        ##loop through unmasked grid points in the tile
        for l in range(len(ii[~mask_chk])):
            ##loop through each field record
            for rec_id, rec in state_info['fields'].items():
                keys = [(0, l), (1, l)] if rec['is_vector'] else [l]

                # for key in keys:
                    # ens_prior = np.array([state[m, rec_id][loc][key] for m in range(c.nens)])

                    ##localization factor
                    # local_factor = np.ones(nlobs)

                    # ens_post = local_analysis(ens_prior, local_obs, obs_err, obs_prior,
                                              # local_factor, c.filter_kind, obs_err_corr)

                    ##save the posterior ensemble to the state
                    # for m in range(c.nens):
                        # state[m, rec_id][loc][key] = ens_post[m]

            # message(c.comm, progress_bar(), 0)

    message(c.comm, ' done.\n', 0)

elif c.assim_mode == 'serial':
    pass

message(c.comm, 'Step 4 took {} seconds\n\n'.format(time.time()-start), 0)
start = time.time()

##--------------------------
##5.Transposing state back to field-complete
message(c.comm, '5.Transpose state from ensemble-complete to field-complete\n', 0)

message(c.comm, 'state variable fields: ', 0)
fields = transpose_state_to_field(c, state_info, field_list, loc_list, state)

state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/post_state.bin'
output_state(c, state_info, field_list, fields, state_file)

message(c.comm, 'Step 5 took {} seconds\n\n'.format(time.time()-start), 0)
start = time.time()

##--------------------------
##6.Post-processing
message(c.comm, '6.Post-processing\n', 0)

##if c.grid != model_grid
##else
## just copy output state to model restart files

message(c.comm, 'Step 6 took {} seconds\n\n'.format(time.time()-start), 0)
start = time.time()

