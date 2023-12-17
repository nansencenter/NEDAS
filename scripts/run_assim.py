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

##analysis grid in 2D is prepared based on config file
##c.grid obj has information on x,y coordinates
timer1 = time.time()

##parse state info, for each member there will be a list of uniq fields
message(c.comm, '1.Prepare state variables\n', 0)
message(c.comm, 'ensemble size nens={}\n'.format(c.nens), 0)

c.state_info = parse_state_info(c)

build_state_tasks(c)

##prepare the ensemble state and z coordinates
# fields, z_fields = prepare_state(c)

##save a copy of the prior state
state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/prior_state.bin'
# output_state(c, fields, state_file)

##collect ensemble mean z fields as analysis grid z coordinates
##for obs preprocessing later
message(c.comm, 'collect model z coordinates, ', 0)
# z_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/z_coords.bin'
# output_ens_mean(c, z_fields, z_file)

timer2 = time.time()
message(c.comm, 'Step 1 took {} seconds\n\n'.format(timer2-timer1), 0)

##Prepare observations
message(c.comm, '2.Prepare obs and obs priors\n', 0)

c.obs_info = parse_obs_info(c)

build_obs_tasks(c)

obs_seq = prepare_obs(c)

c.loc_list_full = partition_grid(c)

c.obs_inds = assign_obs_to_loc(c, obs_seq)

c.loc_list = build_loc_tasks(c)
exit()

##compute obs prior, each pid compute a subset of obs
obs_prior = prepare_obs_prior(c)

timer3 = time.time()
message(c.comm, 'Step 2 took {} seconds\n\n'.format(timer3-timer2), 0)

##transposing fields to local ensemble-complete states:
message(c.comm, '3.Transpose state from field-complete to ensemble-complete\n', 0)

##the local state variables to be updated
message(c.comm, 'state variable fields: ', 0)
state = transpose_field_to_state(c, fields)

##z_coords for state variables
message(c.comm, 'z coords fields: ', 0)
z_state = transpose_field_to_state(c, z_fields)

##global scalar state variables to be updated

timer4 = time.time()
message(c.comm, 'Step 3 took {} seconds\n\n'.format(timer4-timer3), 0)

exit()
##assimilate obs to update state variables
message(c.comm, '4.Assimilation\n', 0)

##loop through location list on pid
for loc in range(len(c.loc_list[c.pid])):

    istart,iend,di,jstart,jend,dj = c.loc_list[pid][loc]
    ii, jj = np.meshgrid(np.arange(istart,iend,di), np.arange(jstart,jend,dj))
    mask_chk = c.mask[jstart:jend:dj, istart:iend:di]

    ##loop through unmasked grid points in the tile fld_chk
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
            ens_post = local_analysis(ens_prior, local_obs, obs_err, obs_prior, local_factor, c.filter_kind, obs_err_corr)

            ##save the posterior ensemble to the state
            for mem_id in range(c.nens):
                state[mem_id, rec_id][loc][l] = ens_post[mem_id]

timer5 = time.time()
message(c.comm, 'Step 4 took {} seconds\n\n'.format(timer5-timer4), 0)

##transposing state back to field-complete
message(c.comm, '5.Transpose state from ensemble-complete to field-complete\n', 0)

message(c.comm, 'state variable fields: ', 0)
fields = transpose_state_to_field(c, state)

state_file = c.work_dir+'/analysis/'+c.time+c.s_dir+'/post_state.bin'
output_state(c, fields, state_file)

timer6 = time.time()
message(c.comm, 'Step 5 took {} seconds\n\n'.format(timer6-timer5), 0)

##Post-processing
message(c.comm, '6.Post-processing\n', 0)

##if c.grid != model_grid
##else
## just copy output state to model restart files

timer7 = time.time()
message(c.comm, 'Step 6 took {} seconds\n\n'.format(timer7-timer6), 0)

