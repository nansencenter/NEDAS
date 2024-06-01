import numpy as np
import sys
import struct
import importlib

from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t, dt1h
from utils.progress import show_progress
from utils.parallel import distribute_tasks, bcast_by_root, by_rank

import numpy as np
import struct
import importlib
from datetime import datetime, timedelta

from perturb import random_field_gaussian
from grid import Grid
from utils.conversion import type_convert, type_dic, type_size, t2h, h2t, t2s, s2t
from utils.log import message, show_progress
from utils.parallel import bcast_by_root, distribute_tasks

from .state import read_field

"""
Note: The entire state is distributed across the memory of many processors,
at any moment, a processor only stores a subset of state in its memory:
either having all the mem_id,rec_id but only a subset of par_id (we call this
ensemble-complete), or having all the par_id but a subset of mem_id,rec_id
(we call this field-complete).
It is easier to perform i/o and pre/post processing on field-complete state,
while easier to run assimilation algorithms with ensemble-complete state.
"""

def transpose_field_to_state(state_info, mem_list, rec_list, partitions, par_list, fields):
    """
    transpose_field_to_state send chunks of field owned by a pid to other pid
    so that the field-complete fields get transposed into ensemble-complete state
    with keys (mem_id, rec_id) pointing to the partition in par_list

    Inputs:
    - c: config module
    - state_info: from parse_state_info()
    - mem_list, rec_list: from build_state_tasks()
    - partitions: from partition_grid()
    - par_list: from build_par_tasks()

    - fields: dict[(mem_id, rec_id), fld]
      The locally stored field-complete fields with subset of mem_id,rec_id

    Returns:
    - state: dict[(mem_id, rec_id), dict[par_id, fld_chk]]
      The locally stored ensemble-complete field chunks on partitions.
    """

    message(c.comm, 'transpose field to state\n', c.pid_show)
    state = {}

    nr = len(rec_list[c.pid_rec])
    for r, rec_id in enumerate(rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in mem_list.items()])

        for m in range(nm_max):
            show_progress(c.comm, r*nm_max+m, nr*nm_max, c.pid_show)

            ##prepare the fld for sending if not at the end of mem_list
            if m < len(mem_list[c.pid_mem]):
                mem_id = mem_list[c.pid_mem][m]
                rec = state_info['fields'][rec_id]
                fld = fields[mem_id, rec_id].copy()

            ## - for each source pid_mem (src_pid) with fields[mem_id, rec_id],
            ##   send chunk of fld[..., jstart:jend:dj, istart:iend:di] to
            ##   destination pid_mem (dst_pid) with its partition in par_list
            ## - every pid needs to send/recv to/from every pid, so we use cyclic
            ##   coreography here to prevent deadlock

            ## 1) receive fld_chk from src_pid, for src_pid<pid first
            for src_pid in np.arange(0, c.pid_mem):
                if m < len(mem_list[src_pid]):
                    src_mem_id = mem_list[src_pid][m]
                    state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

            ## 2) send my fld chunk to a list of dst_pid, send to dst_pid>=pid first
            ##    because they wait to receive before able to send their own stuff;
            ##    when finished with dst_pid>=pid, cycle back to send to dst_pid<pid,
            ##    i.e., dst_pid list = [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
            if m < len(mem_list[c.pid_mem]):
                for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    fld_chk = {}
                    for par_id in par_list[dst_pid]:
                        ##slice for this par_id
                        istart,iend,di,jstart,jend,dj = partitions[par_id]
                        ##save the unmasked points in slice to fld_chk for this par_id
                        mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                        if rec['is_vector']:
                            fld_chk[par_id] = fld[:, jstart:jend:dj, istart:iend:di][:, ~mask_chk]
                        else:
                            fld_chk[par_id] = fld[jstart:jend:dj, istart:iend:di][~mask_chk]

                    if dst_pid == c.pid_mem:
                        ##same pid, so just write to state
                        state[mem_id, rec_id] = fld_chk
                    else:
                        ##send fld_chk to dst_pid's state
                        c.comm_mem.send(fld_chk, dest=dst_pid, tag=m)

            ## 3) finish receiving fld_chk from src_pid, for src_pid>pid now
            for src_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                if m < len(mem_list[src_pid]):
                    src_mem_id = mem_list[src_pid][m]
                    state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

    message(c.comm, ' done.\n', c.pid_show)

    return state


def transpose_state_to_field(state_info, mem_list, rec_list, partitions, par_list, state):
    """
    transpose_state_to_field transposes back the state to field-complete fields

    Inputs:
    - c: config module
    - state_info: from parse_state_info()
    - mem_list, rec_list: from build_state_tasks()
    - partitions: from partition_grid()
    - par_list: from build_par_tasks()

    - state: dict[(mem_id, rec_id), dict[par_id, fld_chk]]
      the locally stored ensemble-complete field chunks for subset of par_id

    Returns:
    - fields: dict[(mem_id, rec_id), fld]
      the locally stored field-complete fields for subset of mem_id,rec_id.
    """

    message(c.comm, 'transpose state to field\n', c.pid_show)
    fields = {}

    ##all pid goes through their own task list simultaneously
    nr = len(rec_list[c.pid_rec])
    for r, rec_id in enumerate(rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in mem_list.items()])

        for m in range(nm_max):
            show_progress(c.comm, r*nm_max+m, nr*nm_max, c.pid_show)

            ##prepare an empty fld for receiving if not at the end of mem_list
            if m < len(mem_list[c.pid_mem]):
                mem_id = mem_list[c.pid_mem][m]
                rec = state_info['fields'][rec_id]
                if rec['is_vector']:
                    fld = np.full((2, c.ny, c.nx), np.nan)
                else:
                    fld = np.full((c.ny, c.nx), np.nan)

            ##this is just the reverse of transpose_field_to_state
            ## we take the exact steps, but swap send and recv operations here
            ##
            ## 1) send my fld_chk to dst_pid, for dst_pid<pid first
            for dst_pid in np.arange(0, c.pid_mem):
                if m < len(mem_list[dst_pid]):
                    dst_mem_id = mem_list[dst_pid][m]
                    c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                    del state[dst_mem_id, rec_id]   ##free up memory

            ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
            ##    because they wait to send stuff before able to receive themselves,
            ##    cycle back to receive from src_pid<pid then.
            if m < len(mem_list[c.pid_mem]):
                for src_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    if src_pid == c.pid_mem:
                        ##same pid, so just copy fld_chk from state
                        fld_chk = state[mem_id, rec_id].copy()
                    else:
                        ##receive fld_chk from src_pid's state
                        fld_chk = c.comm_mem.recv(source=src_pid, tag=m)

                    ##unpack the fld_chk to form a complete field
                    for par_id in par_list[src_pid]:
                        istart,iend,di,jstart,jend,dj = partitions[par_id]
                        mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                        fld[..., jstart:jend:dj, istart:iend:di][..., ~mask_chk] = fld_chk[par_id]

                    fields[mem_id, rec_id] = fld

            ## 3) finish sending fld_chk to dst_pid, for dst_pid>pid now
            for dst_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                if m < len(mem_list[dst_pid]):
                    dst_mem_id = mem_list[dst_pid][m]
                    c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                    del state[dst_mem_id, rec_id]   ##free up memory

    message(c.comm, ' done.\n', c.pid_show)

    return fields


def transpose_obs_to_lobs(mem_list, rec_list, obs_rec_list, par_list, obs_inds, input_obs, ensemble=False):
    """
    Transpose obs from field-complete to ensemble-complete

    Step 1: Within comm_mem, send the subset of input_obs with mem_id and par_id
            from the source proc (src_pid) to the destination proc (dst_pid), store the
            result in tmp_obs with all the mem_id (ensemble-complete)
    Step 2: Gather all obs_rec_id within comm_rec, so that each pid_rec will have the
            entire obs record for assimilation

    Requires attributes in config:
    - mem_list, rec_list: from build_state_tasks()
    - obs_rec_list: from build_obs_tasks()
    - par_list: from build_par_tasks()
    - obs_inds: from assign_obs()
    - input_obs: obs_seq from process_all_obs() or obs_prior_seq from process_all_obs_priors()
    - ensemble: bool

    Returns:
    - output_obs:
      If ensemble: the input_obs is the obs_prior_seq: dict[(mem_id, obs_rec_id), np.array]
      output_obs: dict[(mem_id, obs_rec_id), dict[par_id, np.array]]
      is the local observation priors sequence

      If not ensemble: the input_obs is the obs_seq: dict[obs_rec_id, dict[key, np.array]]
      output_obs: dict[obs_rec_id, dict[par_id, dict[key, np.array]]]
      is the local observation sequence, key = 'obs','x','y','z','t'...
    """

    pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in obs_rec_list.items() if len(lst)>0][0]
    c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show

    if ensemble:
        message(c.comm, 'obs prior sequences: ', c.pid_show)
    else:
        message(c.comm, 'obs sequences: ', c.pid_show)
    message(c.comm, 'transpose obs to local obs\n', c.pid_show)

    ##Step 1: transpose to ensemble-complete by exchanging mem_id, par_id in comm_mem
    ##        input_obs -> tmp_obs
    tmp_obs = {}  ##local obs at intermediate stage

    nr = len(obs_rec_list[c.pid_rec])
    for r, obs_rec_id in enumerate(obs_rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in mem_list.items()])
        for m in range(nm_max):

            show_progress(c.comm, r*nm_max+m, nr*nm_max, c.pid_show)

            ##prepare the obs seq for sending if not at the end of mem_list
            if m < len(mem_list[c.pid_mem]):
                mem_id = mem_list[c.pid_mem][m]
                if ensemble:  ##this is the obs prior seq
                    seq = input_obs[mem_id, obs_rec_id].copy()
                else:
                    if mem_id == 0:  ##this is the obs seq, just let mem_id=0 send it
                        seq = input_obs[obs_rec_id].copy()

            ##the collective send/recv follows the same idea under state.transpose_field_to_state
            ##1) receive lobs_seq from src_pid, for src_pid<pid first
            for src_pid in np.arange(0, c.pid_mem):
                if m < len(mem_list[src_pid]):
                    src_mem_id = mem_list[src_pid][m]
                    if ensemble:
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                    else:
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

            ##2) send my obs chunk to a list of dst_pid, send to dst_pid>=pid first
            ##   then cycle back to send to dst_pid<pid. i.e. the dst_pid sequence is
            ##   [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
            if m < len(mem_list[c.pid_mem]):
                for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    if ensemble:
                        ##this is the obs prior seq for mem_id, obs_rec_id
                        ##for each par_id, assemble the subset lobs_seq using obs_inds
                        lobs_seq = {}
                        for par_id in par_list[dst_pid]:
                            inds = obs_inds[obs_rec_id][par_id]
                            lobs_seq[par_id] = seq[..., inds]

                        if dst_pid == c.pid_mem:
                            ##pid already stores the lobs_seq, just copy
                            tmp_obs[mem_id, obs_rec_id] = lobs_seq
                        else:
                            ##send lobs_seq to dst_pid
                            c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

                    else:
                        if mem_id == 0:
                            ##this is the obs seq with keys 'obs','err_std','x','y','z','t'
                            ##assemble the lobs_seq dict with same keys but subset obs_inds
                            ##do this for each par_id to get the full lobs_seq
                            lobs_seq = {}
                            for par_id in par_list[dst_pid]:
                                lobs_seq[par_id] = {}
                                inds = obs_inds[obs_rec_id][par_id]
                                for key in ('obs', 'err_std', 'x', 'y', 'z', 't'):
                                    lobs_seq[par_id][key] = seq[key][..., inds]

                            if dst_pid == c.pid_mem:
                                ##pid already stores the lobs_seq, just copy
                                tmp_obs[obs_rec_id] = lobs_seq
                            else:
                                ##send lobs_seq to dst_pid's lobs
                                c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

            ##3) finish receiving lobs_seq from src_pid, for src_pid>pid now
            for src_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                if m < len(mem_list[src_pid]):
                    src_mem_id = mem_list[src_pid][m]
                    if ensemble:
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                    else:
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

    message(c.comm, ' done.\n', c.pid_show)

    ##Step 2: collect all obs records (all obs_rec_ids) on pid_rec
    ##        tmp_obs -> output_obs
    output_obs = {}
    for entry in c.comm_rec.allgather(tmp_obs):
        for key, data in entry.items():
            output_obs[key] = data

    return output_obs


def transpose(c):


