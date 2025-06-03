import os
import struct
import numpy as np
from NEDAS.utils.conversion import type_dic, type_size, t2h, h2t, dt1h, ensure_list
from NEDAS.utils.parallel import distribute_tasks, bcast_by_root
from NEDAS.utils.progress import progress_bar

class State:
    """
    The State class manages the state variables for the assimilation system.
    
    The analysis is performed on a regular grid.
    
    The entire state has dimensions: member, variable, time,  z,  y,  x
    indexed by: mem_id,        v,    t,  k,  j,  i
    with size:   nens,       nv,   nt, nz, ny, nx

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
    def __init__(self, c):
        self.analysis_dir = c.analysis_dir(c.time, c.iter)
        self.prior_file = os.path.join(self.analysis_dir, 'prior_state.bin')
        self.prior_mean_file = os.path.join(self.analysis_dir, 'prior_mean_state.bin')
        self.post_file = os.path.join(self.analysis_dir, 'post_state.bin')
        self.post_mean_file = os.path.join(self.analysis_dir, 'post_mean_state.bin')
        self.z_coords_file = os.path.join(self.analysis_dir, 'z_coords.bin')

        self.info = bcast_by_root(c.comm)(self.parse_state_info)(c)

        self.mem_list, self.rec_list = bcast_by_root(c.comm)(self.distribute_state_tasks)(c)

        self.partitions = {}  ##will be setup by assimilator.partition_grid()
        self.par_list = {}

        self.fields_prior = {}  ##will be created by prepare_obs()
        self.z_fields = {}
        self.state_prior = {}   ##will be created by transpose_to_ensemble_complete()
        self.z_state = {}
        self.state_post = {}    ##will be created by assimilator.assimilate()
        self.fields_post = {}   ##will be created by transpose_to_field_complete()
        self.data = {}  ##will be created by pack_state_data, for use in assimilate()

    def prepare_state(self, c):
        """
        Main method from state object to be called in the analysis script
        """
        self.fields_prior, self.z_fields = self.collect_fields_from_restartfiles(c)
        self.output_z_coords(c)

    def parse_state_info(self, c) -> dict:
        """
        Parses info for the nrec fields in the state.

        Args:
            c (Config): config obj

        Returns:
            dict: A dictionary with some dimensions and list of unique field records
        """
        info = {}
        info['size'] = 0
        info['shape'] = c.grid.x.shape
        info['fields'] = {}
        info['scalars'] = {}
        rec_id = 0   ##record id for a 2D field
        pos = 0      ##seek position for rec
        variables = set()
        err_types = set()

        ##loop through variables in state_def
        for vrec in ensure_list(c.state_def):
            vname = vrec['name']
            variables.add(vname)
            err_types.add(vrec['err_type'])

            if vrec['var_type'] == 'field':
                ##this is a state variable 'field' with dimensions t, z, y, x
                ##some properties of the variable is defined in its source module
                src = c.models[vrec['model_src']]
                assert vname in src.variables, 'variable '+vname+' not defined in '+vrec['model_src']+' Model.variables'

                #now go through time and zlevels to form a uniq field record
                for time in c.time + np.array(c.state_time_steps)*dt1h:
                    for k in src.variables[vname]['levels']:
                        rec = { 'name': vname,
                                'model_src': vrec['model_src'],
                                'dtype': src.variables[vname]['dtype'],
                                'is_vector': src.variables[vname]['is_vector'],
                                'units': src.variables[vname]['units'],
                                'err_type': vrec['err_type'],
                                'time': time,
                                'dt': c.state_time_scale,
                                'k': k,
                                'pos': pos, }
                        info['fields'][rec_id] = rec

                        ##update seek position
                        nv = 2 if rec['is_vector'] else 1
                        fld_size = np.sum((~c.grid.mask).astype(int))
                        pos += nv * fld_size * type_size[rec['dtype']]
                        rec_id += 1

            elif vrec['var_type'] == 'scalar':
                pass

            else:
                raise NotImplementedError(f"{vrec['var_type']} is not supported in the state vector.")

        if c.debug:
            print(f"number of ensemble members, nens={c.nens}", flush=True)
            print(f"number of unique field records, nrec={len(info['fields'])}", flush=True)
            print(f"variables: {variables}", flush=True)

        info['size'] = pos ##size of a complete state (fields) for 1 memeber
        info['variables'] = list(variables)
        info['err_types'] = list(err_types)

        return info

    def write_state_info(self, binfile: str):
        """
        Write state_info to a .dat file accompanying the .bin file

        Args:
            binfile (str): File path for the .bin file
        """
        with open(binfile.replace('.bin','.dat'), 'wt') as f:
            ##first line: grid dimension
            if len(self.info['shape']) == 1:
                f.write('{}\n'.format(self.info['shape'][0]))
            else:
                f.write('{} {}\n'.format(self.info['shape'][0], self.info['shape'][1]))

            ##second line: total size of the state
            f.write('{}\n'.format(self.info['size']))

            ##followed by nfield lines: each for a field record
            for i, rec in self.info['fields'].items():
                name = rec['name']
                model_src = rec['model_src']
                dtype = rec['dtype']
                is_vector = int(rec['is_vector'])
                units = rec['units']
                err_type = rec['err_type']
                time = t2h(rec['time'])
                dt = rec['dt']
                k = rec['k']
                pos = rec['pos']
                f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(name, model_src, dtype, is_vector, units, err_type, time, dt, k, pos))

    def read_state_info(self, binfile: str) -> dict:
        """
        Read .dat file accompanying the .bin file and obtain state_info

        Args:
            binfile (str): File path for the .bin file

        Returns:
            dict: state info dictionary
        """
        with open(binfile.replace('.bin','.dat'), 'r') as f:
            lines = f.readlines()
            info = {}

            ss = lines[0].split()
            if len(ss)==1:
                info['shape'] = (int(ss),)
            else:
                info['shape'] = (int(ss[0]), int(ss[1]))

            info['size'] = int(lines[1])

            ##records for uniq fields
            info['fields'] = {}
            rec_id = 0
            for lin in lines[2:]:
                ss = lin.split()
                rec = {'name': ss[0],
                    'model_src': ss[1],
                    'dtype': ss[2],
                    'is_vector': bool(int(ss[3])),
                    'units': ss[4],
                    'err_type': ss[5],
                    'time': h2t(np.float32(ss[6])),
                    'dt': np.float32(ss[7]),
                    'k': np.float32(ss[8]),
                    'pos': int(ss[9]), }
                info['fields'][rec_id] = rec
                rec_id += 1
            return info

    def write_field(self, binfile: str, mask: np.ndarray, mem_id: int, rec_id: int, fld: np.ndarray) -> None:
        """
        Write a field to a binary file

        Args:
            binfile (str): File path for the .bin file
            mask (np.ndarray):  Array of bool with same shape as grid.x.
                True if the grid point is masked (for example land grid point in ocean models).
                The masked points will not be stored in the binfile to reduce disk usage.
            mem_id (int): Member index, from 0 to nens-1
            rec_id (int): Field record index, info['fields'][rec_id] gives the record information
            fld (np.ndarray):  The field to be written to the file
        """
        rec = self.info['fields'][rec_id]

        fld_shape = (2,)+self.info['shape'] if rec['is_vector'] else self.info['shape']
        assert fld.shape == fld_shape, f'fld shape incorrect: expected {fld_shape}, got {fld.shape}'

        if rec['is_vector']:
            fld_ = fld[:, ~mask].flatten()
        else:
            fld_ = fld[~mask]

        with open(binfile, 'r+b') as f:
            f.seek(mem_id*self.info['size'] + rec['pos'])
            f.write(struct.pack(fld_.size*type_dic[rec['dtype']], *fld_))

    def read_field(self, binfile: str, mask: np.ndarray, mem_id: int, rec_id: int) -> np.ndarray:
        """
        Read a field from a binary file

        Args:
        binfile (str): File path for the .bin file
        mask (np.ndarray): Mask of grid points.
        mem_id (int): Member index from 0 to nens-1
        rec_id (int): Field record index, info['fields'][rec_id] gives the record information

        Returns:
            np.ndarray: The field read from the file
        """
        rec = self.info['fields'][rec_id]

        nv = 2 if rec['is_vector'] else 1
        fld_shape = (2,)+self.info['shape'] if rec['is_vector'] else self.info['shape']
        fld_size = np.sum((~mask).astype(int))

        with open(binfile, 'rb') as f:
            f.seek(mem_id*self.info['size'] + rec['pos'])
            fld_ = np.array(struct.unpack((nv*fld_size*type_dic[rec['dtype']]),
                            f.read(nv*fld_size*type_size[rec['dtype']])))
            fld = np.full(fld_shape, np.nan)
            if rec['is_vector']:
                fld[:, ~mask] = fld_.reshape((2, -1))
            else:
                fld[~mask] = fld_
            return fld

    def distribute_state_tasks(self, c):
        """
        Distribute mem_id and rec_id across processors

        Args:
        c (Config): config obj

        Returns:
            dict: mem_list {pid_mem: list[mem_id]}
            dict: rec_list {pid_rec: list[rec_id]}
        """
        ##list of mem_id as tasks
        mem_list = distribute_tasks(c.comm_mem, [m for m in range(c.nens)])

        ##list rec_id as tasks
        rec_list_full = [i for i in self.info['fields'].keys()]
        rec_size = np.array([2 if r['is_vector'] else 1 for i,r in self.info['fields'].items()])
        rec_list = distribute_tasks(c.comm_rec, rec_list_full, rec_size)

        return mem_list, rec_list

    def output_state(self, c, fields, state_file, mem_id_out=None, rec_id_out=None):
        """
        Parallel output the fields to the binary state_file

        Args:
            c (Config): config obj
            fields (dict): the locally-stored field-complete fields for output, [(mem_id, rec_id), fld]
            state_file (str): path to the output binary file.
            mem_id_out (int, optional): member id to be output, if None all available ids will output.
            rec_id_out (int, optional): record id to be output, if None all available ids will output.
        """
        c.print_1p('>>> save state to '+state_file+'\n')

        if c.pid == 0:
            ##if file doesn't exist, create the file
            open(state_file, 'wb')
            ##write state_info to the accompanying .dat file
            self.write_state_info(state_file)
        c.comm.Barrier()

        nm = len(self.mem_list[c.pid_mem])
        nr = len(self.rec_list[c.pid_rec])
        for m, mem_id in enumerate(self.mem_list[c.pid_mem]):
            if mem_id_out is not None and mem_id != mem_id_out:
                continue
            for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
                if rec_id_out is not None and rec_id != rec_id_out:
                    continue

                if c.debug:
                    rec = self.info['fields'][rec_id]
                    print(f"PID {c.pid:4}: saving field: mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nm*nr))

                ##get the field record for output
                fld = fields[mem_id, rec_id]

                ##write the data to binary file
                self.write_field(state_file, c.grid.mask, mem_id, rec_id, fld)

        c.comm.Barrier()
        c.print_1p(' done.\n')

    def output_ens_mean(self, c, fields, mean_file):
        """
        Compute ensemble mean of a field stored distributively on all pid_mem
        collect means on pid_mem=0, and output to mean_file

        Args:
            c (Config): config obj
            fields (dict): the locally stored field-complete fields for output
            mean_file (str): path to the output binary file for the ensemble mean
        """
        c.print_1p('>>> compute ensemble mean, save to '+mean_file+'\n')

        if c.pid == 0:
            ##if file doesn't exist, create the file, write state_info
            open(mean_file, 'wb')
            self.write_state_info(mean_file)
        c.comm.Barrier()

        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
            rec = self.info['fields'][rec_id]
            if c.debug:
                print(f"PID {c.pid:4}: saving mean field '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
            else:
                c.print_1p(progress_bar(r, len(self.rec_list[c.pid_rec])))

            ##initialize a zero field with right dimensions for rec_id
            fld_shape = (2,)+self.info['shape'] if rec['is_vector'] else self.info['shape']
            sum_fld_pid = np.zeros(fld_shape)

            ##sum over all fields locally stored on pid
            for mem_id in self.mem_list[c.pid_mem]:
                sum_fld_pid += fields[mem_id, rec_id]

            ##sum over all field sums on different pids together to get the total sum
            ##TODO:reduce is expensive if only part of pid holds state in memory
            sum_fld = c.comm_mem.reduce(sum_fld_pid, root=0)

            if c.pid_mem == 0:
                mean_fld = sum_fld / c.nens
                self.write_field(mean_file, c.grid.mask, 0, rec_id, mean_fld)

        c.comm.Barrier()
        c.print_1p(' done.\n')

    def output_z_coords(self, c):
        ##topaz uses the first ensemble member z coords as the reference z for obs
        ##include this here for backward compatibility
        ##there is no need for choosing which member also, just use the first one
        if c.z_coords_from == 'member':
            self.output_state(c, self.z_fields, self.z_coords_file, mem_id_out=0)

        ##we use by default the ensemble mean z coords as the reference z for obs
        if c.z_coords_from == 'mean':
            self.output_ens_mean(c, self.z_fields, self.z_coords_file)

    def collect_fields_from_restartfiles(self, c):
        """
        Collects fields from model restart files, convert them to the analysis grid,
        preprocess (coarse-graining etc), save to fields[mem_id, rec_id] pointing to the uniq fields

        Args:
            c (Config): config object

        Returns:
            dict: fields dictionary [(mem_id, rec_id), fld]
                where fld is np.array defined on c.grid, it's one of the state variable field
            dict: z_fields dictionary [(mem_id, rec_id), zfld]
                where zfld is same shape as fld, it's he z coordinates corresponding to each field
        """
        pid_mem_show = [p for p,lst in self.mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * c.nproc_mem + pid_mem_show

        ##pid_show has some workload, it will print progress message
        c.print_1p('>>> prepare state by reading fields from model restart\n')
        fields = {}
        z_fields = {}

        ##process the fields, each proc gets its own workload as a subset of
        ##mem_id,rec_id; all pid goes through their own task list simultaneously
        nm = len(self.mem_list[c.pid_mem])
        nr = len(self.rec_list[c.pid_rec])

        for m, mem_id in enumerate(self.mem_list[c.pid_mem]):
            for r, rec_id in enumerate(self.rec_list[c.pid_rec]):
                rec = self.info['fields'][rec_id]

                if c.debug:
                    print(f"PID {c.pid:4}: prepare_state mem{mem_id+1:03} '{rec['name']:20}' {rec['time']} k={rec['k']}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nm*nr))

                ##directory storing model output
                path = c.forecast_dir(rec['time'], rec['model_src'])

                ##the model object for handling this variable
                model = c.models[rec['model_src']]

                ##read field from restart file
                model.read_grid(path=path, member=mem_id, **rec)
                var = model.read_var(path=path, member=mem_id, **rec)

                model.grid.set_destination_grid(c.grid)
                fld = model.grid.convert(var, is_vector=rec['is_vector'], method='linear', coarse_grain=True)

                if rec['is_vector']:
                    fld[:, c.grid.mask] = np.nan
                else:
                    fld[c.grid.mask] = np.nan

                ##misc. transform can be added here
                for transform_func in c.transform_funcs:
                    fld = transform_func.forward_state(c, rec, fld)

                ##save field to dict
                fields[mem_id, rec_id] = fld

                ##read z_coords for the field
                ##only need to generate the uniq z coords, store in bank
                zvar = model.z_coords(path=path, member=mem_id, **rec)
                z = model.grid.convert(zvar, is_vector=False, method='linear', coarse_grain=True)
                if rec['is_vector']:
                    z_fields[mem_id, rec_id] = np.array([z, z])
                else:
                    z_fields[mem_id, rec_id] = z

        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##additonal output of debugging
        if c.debug:
            np.save(os.path.join(self.analysis_dir, f'fields_prior.{c.pid_mem}.{c.pid_rec}.npy'), fields)

        return fields, z_fields

    def collect_scalar_variables(self, c):
        pass
        # TODO: implement scalars here for simultaneous state parameter estimation (SSPE)

    def pack_field_chunk(self, c, fld, is_vector, dst_pid):
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

    def transpose_to_ensemble_complete(self, c, fields):
        """
        Send chunks of field owned by a pid to other pid
        so that the field-complete fields get transposed into ensemble-complete state
        with keys (mem_id, rec_id) pointing to the partition in par_list

        Args:
            c (Config): config obj
            fields (dict): The locally stored field-complete fields with subset of mem_id,rec_id

        Returns:
            dict: The locally stored ensemble-complete field chunks on partitions, dict[(mem_id, rec_id), dict[par_id, fld_chk]]
        """
        c.print_1p('transpose field-complete to ensemble-complete\n')
        state = {}

        nr = len(self.rec_list[c.pid_rec])
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in self.mem_list.items()])
            for m in range(nm_max):
                if c.debug:
                    if m < len(self.mem_list[c.pid_mem]):
                        mem_id = self.mem_list[c.pid_mem][m]
                        print(f"PID {c.pid:4}: transposing field: mem{mem_id+1:03} rec{rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing field: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare the fld for sending if not at the end of mem_list
                if m < len(self.mem_list[c.pid_mem]):
                    mem_id = self.mem_list[c.pid_mem][m]
                    rec = self.info['fields'][rec_id]
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
                    for dst_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                        fld_chk = self.pack_field_chunk(c, fld, rec['is_vector'], dst_pid)
                        if dst_pid == c.pid_mem:
                            ##same pid, so just write to state
                            state[mem_id, rec_id] = fld_chk
                        else:
                            ##send fld_chk to dst_pid's state
                            c.comm_mem.send(fld_chk, dest=dst_pid, tag=m)

                ## 3) finish receiving fld_chk from src_pid, for src_pid>pid now
                for src_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                    if m < len(self.mem_list[src_pid]):
                        src_mem_id = self.mem_list[src_pid][m]
                        state[src_mem_id, rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
        c.comm.Barrier()
        c.print_1p(' done.\n')
        return state

    def transpose_to_field_complete(self, c, state):
        """
        Transposes back the state to field-complete fields

        Args:
            c (Config): config obj
            state (dict): the locally stored ensemble-complete field chunks for subset of par_id

        Returns:
            dict: the locally stored field-complete fields for subset of mem_id,rec_id.
        """
        c.print_1p('transpose ensemble-complete to field-complete\n')
        fields = {}

        ##all pid goes through their own task list simultaneously
        nr = len(self.rec_list[c.pid_rec])
        for r, rec_id in enumerate(self.rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in self.mem_list.items()])

            for m in range(nm_max):
                if c.debug:
                    if m < len(self.mem_list[c.pid_mem]):
                        mem_id = self.mem_list[c.pid_mem][m]
                        print(f"PID {c.pid:4}: transposing field: mem{mem_id+1:03} rec{rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing field: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare an empty fld for receiving if not at the end of mem_list
                if m < len(self.mem_list[c.pid_mem]):
                    mem_id = self.mem_list[c.pid_mem][m]
                    rec = self.info['fields'][rec_id]
                    if rec['is_vector']:
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
                    for src_pid in np.mod(np.arange(c.nproc_mem)+c.pid_mem, c.nproc_mem):
                        if src_pid == c.pid_mem:
                            ##same pid, so just copy fld_chk from state
                            fld_chk = state[mem_id, rec_id].copy()
                        else:
                            ##receive fld_chk from src_pid's state
                            fld_chk = c.comm_mem.recv(source=src_pid, tag=m)

                        ##unpack the fld_chk to form a complete field
                        self.unpack_field_chunk(c, fld, fld_chk, src_pid)

                ## 3) finish sending fld_chk to dst_pid, for dst_pid>pid now
                for dst_pid in np.arange(c.pid_mem+1, c.nproc_mem):
                    if m < len(self.mem_list[dst_pid]):
                        dst_mem_id = self.mem_list[dst_pid][m]
                        c.comm_mem.send(state[dst_mem_id, rec_id], dest=dst_pid, tag=m)
                        del state[dst_mem_id, rec_id]   ##free up memory
        c.comm.Barrier()
        c.print_1p(' done.\n')
        return fields

    def pack_local_state_data(self, c, par_id, state_prior, z_state):
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
            rec = self.info['fields'][rec_id]
            v_list = [0, 1] if rec['is_vector'] else [None]
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
            rec = self.info['fields'][rec_id]
            data['t'][n] = t2h(rec['time'])
            data['err_type'][n] = self.info['err_types'].index(rec['err_type'])
            data['var_id'][n] = self.info['variables'].index(rec['name'])
            for m in range(c.nens):
                data['z'][n, :] += np.squeeze(z_state[m, rec_id][par_id][v, :]).astype(np.float32) / c.nens  ##ens mean z
                data['state_prior'][m, n, :] = np.squeeze(state_prior[m, rec_id][par_id][v, :].copy())

        return data

    def unpack_local_state_data(self, c, par_id, state_prior, data):
        """unpack data and write back to the state dict"""
        nfld = len(data['field_ids'])
        nloc = len(data['x'])

        for m in range(c.nens):
            for n in range(nfld):
                rec_id, v = data['field_ids'][n]
                state_prior[m, rec_id][par_id][v, :] = data['state_prior'][m, n, :]
