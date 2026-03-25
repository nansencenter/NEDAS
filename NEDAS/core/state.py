import numpy as np
from NEDAS.utils.conversion import t2h, h2t, dt1h
from NEDAS.utils.parallel import distribute_tasks, bcast_by_root
from .context import Context
from .types import ProcIDMem, ProcIDRec, FieldRecordID, PartitionID, FieldRecord, FieldEns, StateEns
from .state_info import StateInfo

class State:
    """
    The State class manages the state variables for the assimilation system.
    
    The analysis is performed on a regular grid.
    
    The entire state has dimensions: member, variable, time,  z,  y,  x
    indexed by:                      mem_id,        v,    t,  k,  j,  i
    with size:                         nens,       nv,   nt, nz, ny, nx

    To parallelize workload, we group the dimensions into 3 indices:
    
    mem_id indexes the ensemble members
    
    rec_id indexes the uniq 2D fields with (v, t, k), since nz and nt may vary
    for different variables, we stack these dimensions in the 'record'
    dimension with size nrec
    
    par_id indexes the spatial partitions, which are subset of the 2D grid
    given by (ist, ied, di, jst, jed, dj), for a complete field fld[j,i]
    the processor with par_id stores fld[ist:ied:di, jst:jed:dj] locally.

    The entire state is distributed across the memory of many processors,
    at any moment, a processor only stores a subset of state in its memory:
    either having all the mem_id,rec_id but only a subset of par_id (we call this
    ensemble-complete), or having all the par_id but a subset of mem_id,rec_id
    (we call this field-complete).
    It is easier to perform i/o and pre/post processing on field-complete state,
    while easier to run assimilation algorithms with ensemble-complete state.
    """
    info: StateInfo
    rec_list: dict[ProcIDRec, list[FieldRecordID]]
    partitions: list = []  # will be created by assimilator.partition_grid()
    par_list: dict[ProcIDMem, list[PartitionID]] = {}
    fields_prior: FieldEns = {}   # will be created by self.prepare_state()
    fields_z: FieldEns = {}
    state_prior: StateEns = {}    # will be created by self.transpose_to_ensemble_complete() 
    state_z: StateEns = {}
    state_post: StateEns = {}     # will be created by assimilator.assimilate()
    fields_post: FieldEns = {}    # will be created by self.transpose_to_field_complete()
    data = {}                     # will be created by self.pack_state_data(), for use in assmilator.assimilate()

    def __init__(self, c: Context):
        self.info = bcast_by_root(c.comm)(StateInfo)(c)
        self.rec_list = bcast_by_root(c.comm)(self.distribute_state_tasks)(c)

    def distribute_state_tasks(self, c: Context) -> dict[int, list[int]]:
        """
        Distribute rec_id across processors
        """
        ##list rec_id as tasks
        rec_list_full = [i for i in self.info.fields.keys()]
        rec_size = np.array([2 if r.is_vector else 1 for i,r in self.info.fields.items()])
        rec_list = distribute_tasks(c.comm_rec, rec_list_full, rec_size)

        return rec_list

    def prepare_state(self, c: Context) -> None:
        """
        Main method to collect fields from model to form the complete state (field-complete distributed)
        """
        c.logger('Collect prior fields')(self.collect_prior_fields)(c)
        #self.scalars_prior = self.collect_scalars(c)

        c.logger('Collect reference z coords')(self.output_ref_z)(c)

        # compute and save the prior ensemble and mean fields
        c.logger('Output prior ensemble members')(self.output_state)(c, 'prior')
        c.logger('Output prior ensemble mean')(self.output_ens_mean)(c, 'prior')

    def collect_prior_fields(self, c: Context) -> None:
        """
        Collect fields from prior model state, convert them to the analysis grid,
        preprocess (coarse-graining etc), save to fields[mem_id, rec_id] pointing to the uniq fields

        Args:
            c (Context): context object

        Returns:
            dict: fields dictionary [(mem_id, rec_id), fld]
                where fld is np.array defined on c.grid, it's one of the state variable field
            dict: fields_z dictionary [(mem_id, rec_id), zfld]
                where zfld is same shape as fld, it's he z coordinates corresponding to each field
        """
        pid_mem_show = [p for p,lst in c.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.rec_list.items() if len(lst)>0][0]
        ##pid_show has some workload, it will print progress message
        c.pid_show =  pid_rec_show * c.config.nproc_mem + pid_mem_show

        ##process the fields, each proc gets its own workload as a subset of
        ##mem_id,rec_id; all pid goes through their own task list simultaneously
        nm = len(c.mem_list[c.pid_mem])
        nr = len(self.rec_list[c.pid_rec])
        c.total_tasks = nm*nr
        for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
            for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
                rec = self.info.fields[rec_id]

                c.debug_message = f"prepare_state mem{mem_id+1:03} '{rec.name:20}' {rec.time} k={rec.k}"
                c.current_task = m*nr+r

                model_name = rec.model_src
                model = c.models[model_name]
                model_fld = c.io.call_method(c, 'prior', model.read_var, member=mem_id, **rec.asdict())
                model.grid.set_destination_grid(c.grid)
                fld = model.grid.convert(model_fld, is_vector=rec.is_vector, method='linear', coarse_grain=True)
                if rec.is_vector:
                    fld[:, c.grid.mask] = np.nan
                else:
                    fld[c.grid.mask] = np.nan

                ##misc. transform can be added here
                for transform_func in c.transform_funcs:
                    fld = transform_func.forward_state(c, rec, fld)
                ##save field to dict
                self.fields_prior[mem_id, rec_id] = fld

                ##read z_coords for the field
                ##only need to generate the uniq z coords, store in bank
                model_z = c.io.call_method(c, 'prior', model.z_coords, member=mem_id, **rec.asdict())
                z = model.grid.convert(model_z, is_vector=False, method='linear', coarse_grain=True)
                if rec.is_vector:
                    self.fields_z[mem_id, rec_id] = np.array([z, z])
                else:
                    self.fields_z[mem_id, rec_id] = z
        c.comm.Barrier()

        ##additonal output of debugging
        if c.debug:
            c.io.save_debug_data(c, f"fields_prior_{c.pid_mem}_{c.pid_rec}", self.fields_prior)
            ##TODO: data is (mem, rec) -> ndarray, but savez needs str keys

    def collect_scalar_variables(self, c):
        pass
        # TODO: implement scalars here for simultaneous state parameter estimation (SSPE)

    def output_state(self, c: Context, tag: str, mem_id_out: int|None=None, rec_id_out: int|None=None) -> None:
        """
        Parallel output the fields to the binary state_file

        Args:
            c (Context): the runtime context obj
            tag (str): which version of state this is: 'prior', 'post' or 'z' coords?
            mem_id_out (int, optional): member id to be output, if None all available ids will output.
            rec_id_out (int, optional): record id to be output, if None all available ids will output.
        """
        c.io.prepare_fields_storage(c, tag)

        nm = len(c.mem_list[c.pid_mem])
        nr = len(self.rec_list[c.pid_rec])
        c.total_tasks = nm*nr
        for m, mem_id in enumerate(c.mem_list[c.pid_mem]):
            if mem_id_out is not None and mem_id != mem_id_out:
                continue
            for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
                if rec_id_out is not None and rec_id != rec_id_out:
                    continue
                rec = self.info.fields[rec_id]
                c.debug_message = f"saving field: mem{mem_id+1:03} '{rec.name:20}' {rec.time} k={rec.k}"
                c.current_task = m*nr+r

                ##get the field record for output
                fields = getattr(self, f"fields_{tag}")
                fld = fields[mem_id, rec_id]
                c.io.write_field(fld, c, tag, rec_id, mem_id)
        c.comm.Barrier()

    def output_ens_mean(self, c: Context, tag: str) -> None:
        """
        Compute ensemble mean of a field stored distributively on all pid_mem
        collect means on pid_mem=0, and output to mean_file

        Args:
            c (Context): the runtime context obj
            tag (str): which version of state this is: 'prior_mean', 'post_mean', or 'z'
            mean_file (str): path to the output binary file for the ensemble mean
        """
        fields = getattr(self, f"fields_{tag}")
        c.io.prepare_fields_storage(c, f"{tag}_mean")

        c.total_tasks = len(self.rec_list[c.pid_rec])
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
            rec = self.info.fields[rec_id]
            c.debug_message = f"saving mean field '{rec.name:20}' {rec.time} k={rec.k}"
            c.current_task = r

            ##initialize a zero field with right dimensions for rec_id
            fld_shape = (2,)+self.info.shape if rec.is_vector else self.info.shape
            sum_fld_pid = np.zeros(fld_shape)

            ##sum over all fields locally stored on pid
            for mem_id in c.mem_list[c.pid_mem]:
                sum_fld_pid += fields[mem_id, rec_id]

            # sum over all field sums on different pids together to get the total sum
            # TODO:reduce is expensive if only sparse pid holds state in memory, so in runtime should try to
            # populate the comm_mem with members as much as possible.
            sum_fld = c.comm_mem.allreduce(sum_fld_pid)

            mean_fld = sum_fld / c.nens
            c.io.write_field(mean_fld, c, f"{tag}_mean", rec_id, mem_id=0)

        c.comm.Barrier()

    def output_ref_z(self, c: Context):
        # topaz uses the first ensemble member z coords as the reference z for obs
        # include this here for backward compatibility
        # there is no need for choosing which member also, will just use the first one
        if c.config.z_coords_from == 'member':
            self.output_state(c, 'z', mem_id_out=0)

        # we use by default the ensemble mean z coords as the reference z for obs
        if c.config.z_coords_from == 'mean':
            self.output_ens_mean(c, 'z')

    def pack_field_chunk(self, c: Context, fld, is_vector, dst_pid):
        fld_chk = {}
        for par_id in self.par_list[dst_pid]:
            if len(c.grid.x.shape) == 2:
                ##slice for this par_id
                istart,iend,di,jstart,jend,dj = self.partitions[par_id]
                ##save the unmasked points in slice to fld_chk for this par_id
                mask_chk = c.grid.mask[jstart:jend:dj, istart:iend:di]
                if is_vector:
                    fld_chk[par_id] = fld[:, jstart:jend:dj, istart:iend:di][:, ~mask_chk]
                else:
                    fld_chk[par_id] = fld[jstart:jend:dj, istart:iend:di][~mask_chk]
            else:
                inds = self.partitions[par_id]
                mask_chk = c.grid.mask[inds]
                if is_vector:
                    fld_chk[par_id] = fld[:, inds][:, ~mask_chk]
                else:
                    fld_chk[par_id] = fld[inds][~mask_chk]
        return fld_chk

    def unpack_field_chunk(self, c, fld, fld_chk, src_pid):
        for par_id in self.par_list[src_pid]:
            if len(c.grid.x.shape) == 2:
                istart,iend,di,jstart,jend,dj = self.partitions[par_id]
                mask_chk = c.grid.mask[jstart:jend:dj, istart:iend:di]
                fld[..., jstart:jend:dj, istart:iend:di][..., ~mask_chk] = fld_chk[par_id]
            else:
                inds = self.partitions[par_id]
                mask_chk = c.grid.mask[inds]
                fld[..., inds[~mask_chk]] = fld_chk[par_id]

    def transpose_to_ensemble_complete(self, c: Context, fields: FieldEns) -> StateEns:
        """
        Send chunks of field owned by a pid to other pid
        so that the field-complete fields get transposed into ensemble-complete state
        with keys (mem_id, rec_id) pointing to the partition in par_list

        Args:
            c (Context): the runtime context
            fields (FieldEns): The locally stored field-complete fields with subset of mem_id,rec_id

        Returns:
            StateEns: The locally stored ensemble-complete field chunks on partitions, dict[(mem_id, rec_id), dict[par_id, fld_chk]]
        """
        state = {}

        nr = len(self.rec_list[c.pid_rec])
        nm_max = np.max([len(lst) for p,lst in c.mem_list.items()])
        c.total_tasks = nr * nm_max
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            mem_list_own = c.mem_list[c.pid_mem]
            for m in range(nm_max):
                status = f"processing mem{mem_list_own[m]+1:03} rec{rec_id}" if m < len(mem_list_own) else "waiting"
                c.debug_message = f"transposing field: {status}"
                c.current_task = r*nm_max+m

                ##prepare the fld for sending if not at the end of mem_list
                fld = None
                mem_id = None
                rec = None
                if m < len(c.mem_list[c.pid_mem]):
                    mem_id = c.mem_list[c.pid_mem][m]
                    rec = self.info.fields[rec_id]
                    fld = fields[mem_id, rec_id].copy()

                ## - for each source pid_mem (src_pid) with fields[mem_id, rec_id],
                ##   send chunk of fld to destination pid_mem (dst_pid) with its partition in par_list
                ## - every pid needs to send/recv to/from every pid, so we use cyclic
                ##   coreography here to prevent deadlock

                ## 1) receive fld_chk from src_pid, for src_pid<pid first
                for src_pid in range(0, c.pid_mem):
                    if m < len(c.mem_list[src_pid]):
                        src_mem_id = c.mem_list[src_pid][m]
                        state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

                ## 2) send my fld chunk to a list of dst_pid, send to dst_pid>=pid first
                ##    because they wait to receive before able to send their own stuff;
                ##    when finished with dst_pid>=pid, cycle back to send to dst_pid<pid,
                ##    i.e., dst_pid list = [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
                if m < len(c.mem_list[c.pid_mem]):
                    assert isinstance(rec, FieldRecord)
                    for dst_pid in np.mod(np.arange(c.config.nproc_mem)+c.pid_mem, c.config.nproc_mem):
                        fld_chk = self.pack_field_chunk(c, fld, rec.is_vector, dst_pid)
                        if dst_pid == c.pid_mem:
                            ##same pid, so just write to state
                            state[mem_id, rec_id] = fld_chk
                        else:
                            ##send fld_chk to dst_pid's state
                            c.comm_mem.send(fld_chk, dest=dst_pid, tag=m)

                ## 3) finish receiving fld_chk from src_pid, for src_pid>pid now
                for src_pid in range(c.pid_mem+1, c.config.nproc_mem):
                    if m < len(c.mem_list[src_pid]):
                        src_mem_id = c.mem_list[src_pid][m]
                        state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
        c.comm.Barrier()    
        return state

    def transpose_to_field_complete(self, c: Context, state: StateEns) -> FieldEns:
        """
        Transposes back the state to field-complete fields

        Args:
            c (Context): the runtime context
            state (StateEns): the locally stored ensemble-complete field chunks for subset of par_id

        Returns:
            FieldEns: the locally stored field-complete fields for subset of mem_id,rec_id.
        """
        fields = {}

        ##all pid goes through their own task list simultaneously
        nr = len(self.rec_list[c.pid_rec])
        nm_max = np.max([len(lst) for p,lst in c.mem_list.items()])
        c.total_tasks = nr * nm_max
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            mem_list_own = c.mem_list[c.pid_mem]

            for m in range(nm_max):
                status = f"processing mem{mem_list_own[m]} rec{rec_id}" if m < len(mem_list_own) else "waiting"
                c.debug_message = f"transposing field: {status}"
                c.current_task = r*nm_max+m

                ##prepare an empty fld for receiving if not at the end of mem_list
                mem_id = None
                fld = None
                if m < len(c.mem_list[c.pid_mem]):
                    mem_id = c.mem_list[c.pid_mem][m]
                    rec = self.info.fields[rec_id]
                    if rec.is_vector:
                        fld = np.full((2,)+c.grid.x.shape, np.nan)
                    else:
                        fld = np.full(c.grid.x.shape, np.nan)
                    fields[mem_id, rec_id] = fld

                ##this is just the reverse of transpose_field_to_state
                ## we take the exact steps, but swap send and recv operations here
                ##
                ## 1) send my fld_chk to dst_pid, for dst_pid<pid first
                for dst_pid in range(0, c.pid_mem):
                    if m < len(c.mem_list[dst_pid]):
                        dst_mem_id = c.mem_list[dst_pid][m]
                        c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                        del state[dst_mem_id, rec_id]   ##free up memory

                ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
                ##    because they wait to send stuff before able to receive themselves,
                ##    cycle back to receive from src_pid<pid then.
                if m < len(c.mem_list[c.pid_mem]):
                    assert mem_id is not None
                    assert fld is not None
                    for src_pid in np.mod(np.arange(c.config.nproc_mem)+c.pid_mem, c.config.nproc_mem):
                        if src_pid == c.pid_mem:
                            ##same pid, so just copy fld_chk from state
                            fld_chk = state[mem_id, rec_id].copy()
                        else:
                            ##receive fld_chk from src_pid's state
                            fld_chk = c.comm_mem.recv(source=src_pid, tag=m)

                        ##unpack the fld_chk to form a complete field
                        self.unpack_field_chunk(c, fld, fld_chk, src_pid)

                ## 3) finish sending fld_chk to dst_pid, for dst_pid>pid now
                for dst_pid in range(c.pid_mem+1, c.config.nproc_mem):
                    if m < len(c.mem_list[dst_pid]):
                        dst_mem_id = c.mem_list[dst_pid][m]
                        c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                        del state[dst_mem_id, rec_id]   ##free up memory
        c.comm.Barrier()
        return fields

    def pack_local_state_data(self, c: Context, par_id: PartitionID, state_prior: StateEns, state_z: StateEns) -> dict:
        """pack state dict into arrays to be more easily handled by jitted funcs"""
        data = {}

        ##x,y coordinates for local state variables on pid
        if len(c.grid.x.shape) == 2:  ##regular grid
            ist,ied,di,jst,jed,dj = self.partitions[par_id]
            msk = c.grid.mask[jst:jed:dj, ist:ied:di]
            data['x'] = c.grid.x[jst:jed:dj, ist:ied:di][~msk]
            data['y'] = c.grid.y[jst:jed:dj, ist:ied:di][~msk]

        else:
            inds = self.partitions[par_id]
            msk = c.grid.mask[inds]
            data['x'] = c.grid.x[inds][~msk]
            data['y'] = c.grid.y[inds][~msk]

        data['field_ids'] = []
        for rec_id in self.rec_list[c.pid_rec]:
            rec = self.info.fields[rec_id]
            v_list = [0, 1] if rec.is_vector else [None]
            for v in v_list:
                data['field_ids'].append((rec_id, v))

        nfld = len(data['field_ids'])
        nloc = len(data['x'])
        data['t'] = np.full(nfld, np.nan)
        data['z'] = np.zeros((nfld, nloc))
        data['var_id'] = np.full(nfld, 0)
        data['err_type'] = np.full(nfld, 0)
        data['state_prior'] = np.full((c.nens, nfld, nloc), np.nan)
        for n in range(nfld):
            rec_id, v = data['field_ids'][n]
            rec = self.info.fields[rec_id]
            data['t'][n] = t2h(rec.time)
            data['err_type'][n] = self.info.err_types.index(rec.err_type)
            data['var_id'][n] = self.info.variables.index(rec.name)
            for m in range(c.nens):
                data['z'][n, :] += np.squeeze(state_z[m, rec_id][par_id][v, :]).astype(np.float32) / c.nens  ##ens mean z
                data['state_prior'][m, n, :] = np.squeeze(state_prior[m, rec_id][par_id][v, :].copy())

        return data

    def unpack_local_state_data(self, c: Context, par_id: PartitionID, state_prior: StateEns, data: dict) -> None:
        """unpack data and write back to the state dict"""
        nfld = len(data['field_ids'])
        for m in range(c.nens):
            for n in range(nfld):
                rec_id, v = data['field_ids'][n]
                state_prior[m, rec_id][par_id][v, :] = data['state_prior'][m, n, :]
