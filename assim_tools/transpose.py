import numpy as np
from utils.progress import print_with_cache, progress_bar
from utils.parallel import by_rank

"""
Note: The entire state is distributed across the memory of many processors,
at any moment, a processor only stores a subset of state in its memory:
either having all the mem_id,rec_id but only a subset of par_id (we call this
ensemble-complete), or having all the par_id but a subset of mem_id,rec_id
(we call this field-complete).
It is easier to perform i/o and pre/post processing on field-complete state,
while easier to run assimilation algorithms with ensemble-complete state.
"""

def transpose_field_to_state(c, fields):
    """
    transpose_field_to_state send chunks of field owned by a pid to other pid
    so that the field-complete fields get transposed into ensemble-complete state
    with keys (mem_id, rec_id) pointing to the partition in par_list

    Inputs:
    - c: config module
    - fields: dict[(mem_id, rec_id), fld]
      The locally stored field-complete fields with subset of mem_id,rec_id

    Returns:
    - state: dict[(mem_id, rec_id), dict[par_id, fld_chk]]
      The locally stored ensemble-complete field chunks on partitions.
    """

    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('transpose field-complete to ensemble-complete\n')
    state = {}

    nr = len(c.rec_list[c.pid_rec])
    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in c.mem_list.items()])
        for m in range(nm_max):
            if c.debug:
                print(progress_bar(r*nm_max+m, nr*nm_max))

            ##prepare the fld for sending if not at the end of mem_list
            if m < len(c.mem_list[c.pid_mem]):
                mem_id = c.mem_list[c.pid_mem][m]
                rec = c.state_info['fields'][rec_id]
                fld = fields[mem_id, rec_id].copy()

            ## - for each source pid_mem (src_pid) with fields[mem_id, rec_id],
            ##   send chunk of fld[..., jstart:jend:dj, istart:iend:di] to
            ##   destination pid_mem (dst_pid) with its partition in par_list
            ## - every pid needs to send/recv to/from every pid, so we use cyclic
            ##   coreography here to prevent deadlock

            ## 1) receive fld_chk from src_pid, for src_pid<pid first
            for src_pid in np.arange(0, c.pid_mem):
                if m < len(c.mem_list[src_pid]):
                    src_mem_id = c.mem_list[src_pid][m]
                    state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

            ## 2) send my fld chunk to a list of dst_pid, send to dst_pid>=pid first
            ##    because they wait to receive before able to send their own stuff;
            ##    when finished with dst_pid>=pid, cycle back to send to dst_pid<pid,
            ##    i.e., dst_pid list = [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
            if m < len(c.mem_list[c.pid_mem]):
                for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    fld_chk = {}
                    for par_id in c.par_list[dst_pid]:
                        ##slice for this par_id
                        istart,iend,di,jstart,jend,dj = c.partitions[par_id]
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
                if m < len(c.mem_list[src_pid]):
                    src_mem_id = c.mem_list[src_pid][m]
                    state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
    if c.debug:
        print(' done.\n')
    return state


def transpose_state_to_field(c, state):
    """
    transpose_state_to_field transposes back the state to field-complete fields

    Inputs:
    - c: config module
    - state: dict[(mem_id, rec_id), dict[par_id, fld_chk]]
      the locally stored ensemble-complete field chunks for subset of par_id

    Returns:
    - fields: dict[(mem_id, rec_id), fld]
      the locally stored field-complete fields for subset of mem_id,rec_id.
    """

    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('transpose ensemble-complete to field-complete\n')
    fields = {}

    ##all pid goes through their own task list simultaneously
    nr = len(c.rec_list[c.pid_rec])
    for r, rec_id in enumerate(c.rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in c.mem_list.items()])

        for m in range(nm_max):
            if c.debug:
                print(progress_bar(r*nm_max+m, nr*nm_max))

            ##prepare an empty fld for receiving if not at the end of mem_list
            if m < len(c.mem_list[c.pid_mem]):
                mem_id = c.mem_list[c.pid_mem][m]
                rec = c.state_info['fields'][rec_id]
                if rec['is_vector']:
                    fld = np.full((2, c.ny, c.nx), np.nan)
                else:
                    fld = np.full((c.ny, c.nx), np.nan)

            ##this is just the reverse of transpose_field_to_state
            ## we take the exact steps, but swap send and recv operations here
            ##
            ## 1) send my fld_chk to dst_pid, for dst_pid<pid first
            for dst_pid in np.arange(0, c.pid_mem):
                if m < len(c.mem_list[dst_pid]):
                    dst_mem_id = c.mem_list[dst_pid][m]
                    c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                    del state[dst_mem_id, rec_id]   ##free up memory

            ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
            ##    because they wait to send stuff before able to receive themselves,
            ##    cycle back to receive from src_pid<pid then.
            if m < len(c.mem_list[c.pid_mem]):
                for src_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    if src_pid == c.pid_mem:
                        ##same pid, so just copy fld_chk from state
                        fld_chk = state[mem_id, rec_id].copy()
                    else:
                        ##receive fld_chk from src_pid's state
                        fld_chk = c.comm_mem.recv(source=src_pid, tag=m)

                    ##unpack the fld_chk to form a complete field
                    for par_id in c.par_list[src_pid]:
                        istart,iend,di,jstart,jend,dj = c.partitions[par_id]
                        mask_chk = c.mask[jstart:jend:dj, istart:iend:di]
                        fld[..., jstart:jend:dj, istart:iend:di][..., ~mask_chk] = fld_chk[par_id]

                    fields[mem_id, rec_id] = fld

            ## 3) finish sending fld_chk to dst_pid, for dst_pid>pid now
            for dst_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                if m < len(c.mem_list[dst_pid]):
                    dst_mem_id = c.mem_list[dst_pid][m]
                    c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                    del state[dst_mem_id, rec_id]   ##free up memory
    if c.debug:
        print(' done.\n')
    return fields


def transpose_obs_to_lobs(c, input_obs, ensemble=False):
    """
    Transpose obs from field-complete to ensemble-complete

    Step 1: Within comm_mem, send the subset of input_obs with mem_id and par_id
            from the source proc (src_pid) to the destination proc (dst_pid), store the
            result in tmp_obs with all the mem_id (ensemble-complete)
    Step 2: Gather all obs_rec_id within comm_rec, so that each pid_rec will have the
            entire obs record for assimilation

    Requires attributes in config:
    - c: config obj
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

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.obs_rec_list.items() if len(lst)>0][0]
    c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    if c.debug:
        if ensemble:
            print('obs prior sequences: ')
        else:
            print('obs sequences: ')
        print('transpose obs to local obs\n')

    ##Step 1: transpose to ensemble-complete by exchanging mem_id, par_id in comm_mem
    ##        input_obs -> tmp_obs
    tmp_obs = {}  ##local obs at intermediate stage

    nr = len(c.obs_rec_list[c.pid_rec])
    for r, obs_rec_id in enumerate(c.obs_rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in c.mem_list.items()])
        for m in range(nm_max):
            if c.debug:
                print(progress_bar(r*nm_max+m, nr*nm_max))

            ##prepare the obs seq for sending if not at the end of mem_list
            if m < len(c.mem_list[c.pid_mem]):
                mem_id = c.mem_list[c.pid_mem][m]
                if ensemble:  ##this is the obs prior seq
                    seq = input_obs[mem_id, obs_rec_id].copy()
                else:
                    if mem_id == 0:  ##this is the obs seq, just let mem_id=0 send it
                        seq = input_obs[obs_rec_id].copy()

            ##the collective send/recv follows the same idea under state.transpose_field_to_state
            ##1) receive lobs_seq from src_pid, for src_pid<pid first
            for src_pid in np.arange(0, c.pid_mem):
                if m < len(c.mem_list[src_pid]):
                    src_mem_id = c.mem_list[src_pid][m]
                    if ensemble:
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                    else:
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

            ##2) send my obs chunk to a list of dst_pid, send to dst_pid>=pid first
            ##   then cycle back to send to dst_pid<pid. i.e. the dst_pid sequence is
            ##   [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
            if m < len(c.mem_list[c.pid_mem]):
                for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                    if ensemble:
                        ##this is the obs prior seq for mem_id, obs_rec_id
                        ##for each par_id, assemble the subset lobs_seq using obs_inds
                        lobs_seq = {}
                        for par_id in c.par_list[dst_pid]:
                            inds = c.obs_inds[obs_rec_id][par_id]
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
                            for par_id in c.par_list[dst_pid]:
                                lobs_seq[par_id] = {}
                                inds = c.obs_inds[obs_rec_id][par_id]
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
                if m < len(c.mem_list[src_pid]):
                    src_mem_id = c.mem_list[src_pid][m]
                    if ensemble:
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
                    else:
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
    if c.debug:
        print(' done.\n')

    ##Step 2: collect all obs records (all obs_rec_ids) on pid_rec
    ##        tmp_obs -> output_obs
    output_obs = {}
    for entry in c.comm_rec.allgather(tmp_obs):
        for key, data in entry.items():
            output_obs[key] = data
    return output_obs


def transpose_lobs_to_obs(c, lobs):
    """
    Transpose obs from ensemble-complete to field-complete

    Requires attributes in config:
    - c: config obj
    - lobs_post after analysis

    Returns:
    - obs_post_seq:
      dict[(mem_id, obs_rec_id), np.array]
    """

    pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
    pid_rec_show = [p for p,lst in c.obs_rec_list.items() if len(lst)>0][0]
    c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show
    print = by_rank(c.comm, c.pid_show)(print_with_cache)

    if c.debug:
        print('obs post sequences: ')
        print('transpose local obs to obs\n')

    obs_seq = {}
    nr = len(c.obs_rec_list[c.pid_rec])
    for r, obs_rec_id in enumerate(c.obs_rec_list[c.pid_rec]):

        ##all pid goes through their own mem_list simultaneously
        nm_max = np.max([len(lst) for p,lst in c.mem_list.items()])
        for m in range(nm_max):
            if c.debug:
                print(progress_bar(r*nm_max+m, nr*nm_max))

            ##prepare an empty obs_seq for receiving if not at the end of mem_list
            if m < len(c.mem_list[c.pid_mem]):
                mem_id = c.mem_list[c.pid_mem][m]
                rec = c.obs_info['records'][obs_rec_id]
                if rec['is_vector']:
                    seq = np.full((2, rec['nobs']), np.nan)
                else:
                    seq = np.full((rec['nobs'],), np.nan)

            ##this is just the reverse of transpose_obs_to_lobs
            ## we take the exact steps, but swap send and recv operations here
            ##
            ## 1) send my lobs to dst_pid, for dst_pid<pid first
            for dst_pid in np.arange(0, c.pid_mem):
                if m < len(c.mem_list[dst_pid]):
                    dst_mem_id = c.mem_list[dst_pid][m]
                    c.comm_mem.send(lobs[dst_mem_id, obs_rec_id], dest=dst_pid, tag=m)

            ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
            ##    because they wait to send stuff before able to receive themselves,
            ##    cycle back to receive from src_pid<pid then.
            if m < len(c.mem_list[c.pid_mem]):
                for src_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):

                    if src_pid == c.pid_mem:
                        ##pid already stores the lobs_seq, just copy
                        lobs_seq = lobs[mem_id, obs_rec_id].copy()
                    else:
                        ##send lobs_seq to dst_pid
                        lobs_seq = c.comm_mem.recv(source=src_pid, tag=m)

                    ##unpack the lobs_seq to form a complete seq
                    for par_id in c.par_list[src_pid]:
                        inds = c.obs_inds[obs_rec_id][par_id]
                        seq[..., inds] = lobs_seq[par_id]

                    obs_seq[mem_id, obs_rec_id] = seq

            ## 3) finish sending lobs_seq to dst_pid, for dst_pid>pid now
            for dst_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                if m < len(c.mem_list[dst_pid]):
                    dst_mem_id = c.mem_list[dst_pid][m]
                    c.comm_mem.send(lobs[dst_mem_id, obs_rec_id], dest=dst_pid, tag=m)
    if c.debug:
        print(' done.\n')
    return obs_seq


def transpose_forward(c, fields_prior, z_fields, obs_seq, obs_prior_seq):
    """
    transpose funcs called by assimilate.py
    """
    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('tranpose:\n')

    if c.debug:
        print('state variable fields: ')
    state_prior = transpose_field_to_state(c, fields_prior)

    if c.debug:
        print('z coords fields: ')
    z_state = transpose_field_to_state(c, z_fields)

    lobs = transpose_obs_to_lobs(c, obs_seq)
    lobs_prior = transpose_obs_to_lobs(c, obs_prior_seq, ensemble=True)
    return state_prior, z_state, lobs, lobs_prior


def transpose_backward(c, state_post, lobs_post):
    print = by_rank(c.comm, c.pid_show)(print_with_cache)
    if c.debug:
        print('transpose back:\n')

    fields_post = transpose_state_to_field(c, state_post)
    obs_post_seq = transpose_lobs_to_obs(c, lobs_post)

    return fields_post, obs_post_seq


