import numpy as np
from NEDAS.utils.conversion import t2h, h2t
from NEDAS.utils.progress import progress_bar
from NEDAS.utils.parallel import distribute_tasks, bcast_by_root
from .context import Context
from .types import ProcIDMem, ProcIDRec, MemID, FieldRecordID, PartitionID, FieldRecord, FieldEns, StateEns
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
    mem_list: dict[ProcIDMem, list[MemID]]
    rec_list: dict[ProcIDRec, list[FieldRecordID]]
    partitions: list[tuple] = []  # will be created by assimilator.partition_grid()
    par_list: dict[ProcIDMem, list[PartitionID]] = {}
    fields_prior: FieldEns = {}   # will be created by self.prepare_state()
    z_fields: FieldEns = {}
    state_prior: StateEns = {}    # will be created by self.transpose_to_ensemble_complete() 
    z_state: StateEns = {}
    state_post: StateEns = {}     # will be created by assimilator.assimilate()
    fields_post: FieldEns = {}    # will be created by self.transpose_to_field_complete()
    data = {}                     # will be created by self.pack_state_data(), for use in assmilator.assimilate()

    def __init__(self, c: Context):
        self.info = bcast_by_root(c.comm)(StateInfo)(c)
        self.mem_list, self.rec_list = bcast_by_root(c.comm)(self.distribute_state_tasks)(c)

    def distribute_state_tasks(self, c: Context) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """
        Distribute mem_id and rec_id across processors
        """
        ##list of mem_id as tasks
        mem_list = distribute_tasks(c.comm_mem, [m for m in range(c.config.nens)])

        ##list rec_id as tasks
        rec_list_full = [i for i in self.info.fields.keys()]
        rec_size = np.array([2 if r.is_vector else 1 for i,r in self.info.fields.items()])
        rec_list = distribute_tasks(c.comm_rec, rec_list_full, rec_size)

        return mem_list, rec_list

    def prepare_state(self, c: Context) -> None:
        """
        Main method to collect fields from model to form the complete state (field-complete distributed)
        """
        self.collect_prior_fields(c)
        #self.output_z_coords(c) !!!need this?
        #self.scalars_prior = self.collect_scalars(c)

    def collect_prior_fields(self, c: Context) -> None:
        """
        Collect fields from prior model state, convert them to the analysis grid,
        preprocess (coarse-graining etc), save to fields[mem_id, rec_id] pointing to the uniq fields

        Args:
            c (Context): context object

        Returns:
            dict: fields dictionary [(mem_id, rec_id), fld]
                where fld is np.array defined on c.grid, it's one of the state variable field
            dict: z_fields dictionary [(mem_id, rec_id), zfld]
                where zfld is same shape as fld, it's he z coordinates corresponding to each field
        """
        pid_mem_show = [p for p,lst in self.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * c.config.nproc_mem + pid_mem_show

        ##pid_show has some workload, it will print progress message
        c.print_1p('>>> prepare state by reading fields from model restart\n')
        #if self.fields_prior and self.z_fields are not empty, warning that they will be overwritten

        ##process the fields, each proc gets its own workload as a subset of
        ##mem_id,rec_id; all pid goes through their own task list simultaneously
        nm = len(self.mem_list[c.pid_mem])
        nr = len(self.rec_list[c.pid_rec])

        for m, mem_id in enumerate(self.mem_list[c.pid_mem]):
            for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
                rec = self.info.fields[rec_id]
                if c.config.debug:
                    print(f"PID {c.pid:4}: prepare_state mem{mem_id+1:03} '{rec.name:20}' {rec.time} k={rec.k}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nm*nr))

                model_name = rec.model_src
                model = c.models[model_name]
                model_fld = c.io.call_model_io(c, model_name, 'read_var', member=mem_id, **rec.asdict())
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
                model_z = c.io.call_model_io(c, model_name, 'z_coords', member=mem_id, **rec.asdict())
                z = model.grid.convert(model_z, is_vector=False, method='linear', coarse_grain=True)
                if rec.is_vector:
                    self.z_fields[mem_id, rec_id] = np.array([z, z])
                else:
                    self.z_fields[mem_id, rec_id] = z

        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##additonal output of debugging
        if c.config.debug:
            c.io.save_debug_data(c, f"fields_prior_{c.pid_mem}_{c.pid_rec}", self.fields_prior)

    def collect_scalar_variables(self, c):
        pass
        # TODO: implement scalars here for simultaneous state parameter estimation (SSPE)

    # def save_state(self, c,
    #                state_file: Optional[str]=None,
    #                mem_id_out: Optional[int]=None,
    #                rec_id_out: Optional[int]=None) -> None:
    #     """
    #     Parallel output the fields to the binary state_file

    #     Args:
    #         c (Config): config obj
    #         fields (dict): the locally-stored field-complete fields for output, [(mem_id, rec_id), fld]
    #         state_file (str, optional): path to the output binary file.
    #         mem_id_out (int, optional): member id to be output, if None all available ids will output.
    #         rec_id_out (int, optional): record id to be output, if None all available ids will output.
    #     """
    #     c.print_1p('>>> save state to '+state_file+'\n')

    #     if c.pid == 0:
    #         ##if file doesn't exist, create the file
    #         open(state_file, 'wb')
    #         ##write state_info to the accompanying .dat file
    #         self.write_state_info(state_file)
    #     c.comm.Barrier()

    #     nm = len(self.mem_list[c.pid_mem])
    #     nr = len(self.rec_list[c.pid_rec])
    #     for m, mem_id in enumerate(self.mem_list[c.pid_mem]):
    #         if mem_id_out is not None and mem_id != mem_id_out:
    #             continue
    #         for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
    #             if rec_id_out is not None and rec_id != rec_id_out:
    #                 continue

    #             if c.debug:
    #                 rec = self.info['fields'][rec_id]
    #                 print(f"PID {c.pid:4}: saving field: mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
    #             else:
    #                 c.print_1p(progress_bar(m*nr+r, nm*nr))

    #             ##get the field record for output
    #             fld = fields[mem_id, rec_id]

    #             ##write the data to binary file
    #             self.io.write_field(state_file, self.info, c.grid.mask, mem_id, rec_id, fld)

    #     c.comm.Barrier()
    #     c.print_1p(' done.\n')

    # def output_ens_mean(self, c, fields, mean_file):
    #     """
    #     Compute ensemble mean of a field stored distributively on all pid_mem
    #     collect means on pid_mem=0, and output to mean_file

    #     Args:
    #         c (Config): config obj
    #         fields (dict): the locally stored field-complete fields for output
    #         mean_file (str): path to the output binary file for the ensemble mean
    #     """
    #     c.print_1p('>>> compute ensemble mean, save to '+mean_file+'\n')

    #     if c.pid == 0:
    #         ##if file doesn't exist, create the file, write state_info
    #         open(mean_file, 'wb')
    #         self.write_state_info(mean_file)
    #     c.comm.Barrier()

    #     for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
    #         rec = self.info['fields'][rec_id]
    #         if c.debug:
    #             print(f"PID {c.pid:4}: saving mean field '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
    #         else:
    #             c.print_1p(progress_bar(r, len(self.rec_list[c.pid_rec])))

    #         ##initialize a zero field with right dimensions for rec_id
    #         fld_shape = (2,)+self.info['shape'] if rec['is_vector'] else self.info['shape']
    #         sum_fld_pid = np.zeros(fld_shape)

    #         ##sum over all fields locally stored on pid
    #         for mem_id in self.mem_list[c.pid_mem]:
    #             sum_fld_pid += fields[mem_id, rec_id]

    #         ##sum over all field sums on different pids together to get the total sum
    #         ##TODO:reduce is expensive if only part of pid holds state in memory
    #         sum_fld = c.comm_mem.reduce(sum_fld_pid, root=0)

    #         if c.pid_mem == 0:
    #             mean_fld = sum_fld / c.nens
    #             self.write_field(mean_file, c.grid.mask, 0, rec_id, mean_fld)

    #     c.comm.Barrier()
    #     c.print_1p(' done.\n')

    # def output_z_coords(self, c):
    #     ##topaz uses the first ensemble member z coords as the reference z for obs
    #     ##include this here for backward compatibility
    #     ##there is no need for choosing which member also, just use the first one
    #     if c.z_coords_from == 'member':
    #         self.output_state(c, self.z_fields, self.z_coords_file, mem_id_out=0)

    #     ##we use by default the ensemble mean z coords as the reference z for obs
    #     if c.z_coords_from == 'mean':
    #         self.output_ens_mean(c, self.z_fields, self.z_coords_file)

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
                fld[..., inds][..., ~mask_chk] = fld_chk[par_id]

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
        c.print_1p('transpose field-complete to ensemble-complete\n')
        state = {}

        nr = len(self.rec_list[c.pid_rec])
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in self.mem_list.items()])
            for m in range(nm_max):
                if c.config.debug:
                    if m < len(self.mem_list[c.pid_mem]):
                        mem_id = self.mem_list[c.pid_mem][m]
                        print(f"PID {c.pid:4}: transposing field: mem{mem_id+1:03} rec{rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing field: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare the fld for sending if not at the end of mem_list
                fld = None
                mem_id = None
                rec = None
                if m < len(self.mem_list[c.pid_mem]):
                    mem_id = self.mem_list[c.pid_mem][m]
                    rec = self.info.fields[rec_id]
                    fld = fields[mem_id, rec_id].copy()

                ## - for each source pid_mem (src_pid) with fields[mem_id, rec_id],
                ##   send chunk of fld to destination pid_mem (dst_pid) with its partition in par_list
                ## - every pid needs to send/recv to/from every pid, so we use cyclic
                ##   coreography here to prevent deadlock

                ## 1) receive fld_chk from src_pid, for src_pid<pid first
                for src_pid in np.arange(0, c.pid_mem):
                    if m < len(self.mem_list[src_pid]):
                        src_mem_id = self.mem_list[src_pid][m]
                        state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

                ## 2) send my fld chunk to a list of dst_pid, send to dst_pid>=pid first
                ##    because they wait to receive before able to send their own stuff;
                ##    when finished with dst_pid>=pid, cycle back to send to dst_pid<pid,
                ##    i.e., dst_pid list = [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
                if m < len(self.mem_list[c.pid_mem]):
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
                for src_pid in np.arange(c.pid_mem+1, c.config.nproc_mem):
                    if m < len(self.mem_list[src_pid]):
                        src_mem_id = self.mem_list[src_pid][m]
                        state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
        c.comm.Barrier()
        c.print_1p(' done.\n')
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
        c.print_1p('transpose ensemble-complete to field-complete\n')
        fields = {}

        ##all pid goes through their own task list simultaneously
        nr = len(self.rec_list[c.pid_rec])
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in self.mem_list.items()])

            for m in range(nm_max):
                if c.config.debug:
                    if m < len(self.mem_list[c.pid_mem]):
                        mem_id = self.mem_list[c.pid_mem][m]
                        print(f"PID {c.pid:4}: transposing field: mem{mem_id+1:03} rec{rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing field: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare an empty fld for receiving if not at the end of mem_list
                mem_id = None
                fld = None
                if m < len(self.mem_list[c.pid_mem]):
                    mem_id = self.mem_list[c.pid_mem][m]
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
                for dst_pid in np.arange(0, c.pid_mem):
                    if m < len(self.mem_list[dst_pid]):
                        dst_mem_id = self.mem_list[dst_pid][m]
                        c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                        del state[dst_mem_id, rec_id]   ##free up memory

                ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
                ##    because they wait to send stuff before able to receive themselves,
                ##    cycle back to receive from src_pid<pid then.
                if m < len(self.mem_list[c.pid_mem]):
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
                for dst_pid in np.arange(c.pid_mem+1, c.config.nproc_mem):
                    if m < len(self.mem_list[dst_pid]):
                        dst_mem_id = self.mem_list[dst_pid][m]
                        c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                        del state[dst_mem_id, rec_id]   ##free up memory
        c.comm.Barrier()
        c.print_1p(' done.\n')
        return fields

    def pack_local_state_data(self, c: Context, par_id: PartitionID, state_prior: StateEns, z_state: StateEns) -> dict:
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
        data['state_prior'] = np.full((c.config.nens, nfld, nloc), np.nan)
        for n in range(nfld):
            rec_id, v = data['field_ids'][n]
            rec = self.info.fields[rec_id]
            data['t'][n] = t2h(rec.time)
            data['err_type'][n] = self.info.err_types.index(rec.err_type)
            data['var_id'][n] = self.info.variables.index(rec.name)
            for m in range(c.config.nens):
                data['z'][n, :] += np.squeeze(z_state[m, rec_id][par_id][v, :]).astype(np.float32) / c.config.nens  ##ens mean z
                data['state_prior'][m, n, :] = np.squeeze(state_prior[m, rec_id][par_id][v, :].copy())

        return data

    def unpack_local_state_data(self, c: Context, par_id: PartitionID, state_prior: StateEns, data: dict) -> None:
        """unpack data and write back to the state dict"""
        nfld = len(data['field_ids'])
        for m in range(c.config.nens):
            for n in range(nfld):
                rec_id, v = data['field_ids'][n]
                state_prior[m, rec_id][par_id][v, :] = data['state_prior'][m, n, :]
