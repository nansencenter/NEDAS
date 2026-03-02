from typing import Optional
import numpy as np
from datetime import datetime
from NEDAS.utils.conversion import t2h, ensure_list
from NEDAS.utils.progress import progress_bar
from NEDAS.utils.parallel import bcast_by_root, distribute_tasks
from NEDAS.datasets.synthetic import SyntheticObs
from .context import Context
from .types import ProcID, ProcIDRec, PartitionID, ObsRecordID, ObsSeq, ObsEns, LocalObsEns, LocalObsSeq
from .obs_info import ObsInfo

class Obs:
    """
    Class for handling observations.
    
    The observation has dimensions: variable, time, z, y, x
    Since the observation network is typically irregular, we store the obs record
    for each variable in a 1d sequence, with coordinates (t,z,y,x), and size nobs

    To parallelize workload, we distribute each obs record over all the processors

    for batch assimilation mode, each pid stores the list of local obs within the
    hroi of its tiles, with size nlobs (number of local obs)
    
    for serial mode, each pid stores a non-overlapping subset of the obs list,
    here 'local' obs (in storage sense) is broadcast to all pid before computing
    its update to the state/obs near that obs.

    The hroi is separately defined for each obs record.
    For very large hroi, the serial mode is more parallel efficient option, since
    in batch mode the same obs may need to be stored in multiple pids

    To compare to the observation, obs_prior simulated by the model needs to be
    computed, they have dimension [nens, nlobs], indexed by (mem_id, obs_id)
    """
    obs_rec_list: dict[ProcIDRec, list[ObsRecordID]] = {}
    obs_inds: dict = {}            # will be created by assimilator.assign_obs()
    obs_seq: ObsSeq = {}           # will be created by self.prepare_obs()
    obs_prior: ObsEns = {}         # will be created by self.prepare_obs_from_state()
    lobs: LocalObsSeq = {}         # will be created by self.transpose_to_ensemble_complete()
    lobs_prior: LocalObsEns = {}
    lobs_post: LocalObsEns = {}    # will be created by assimilator.assimilate()
    obs_post: ObsEns = {}          # will be created by self.transpose_to_field_complete()
    data: dict = {}                # will be created by self.pack_obs_data, for use in assimilator.assimilate()

    def __init__(self, c: Context):
        self.info = bcast_by_root(c.comm)(ObsInfo)(c)
        self.obs_rec_list = bcast_by_root(c.comm)(self.distribute_obs_tasks)(c)

    def distribute_obs_tasks(self, c: Context):
        """
        Distribute obs_rec_id across processors

        Args:
            c (Context): the runtime context object.

        Returns:
            dict: Dictionary {pid_rec (int): list[obs_rec_id (int)]}
        """
        obs_rec_list_full = [i for i in self.info.records.keys()]
        obs_rec_size = np.array([2 if r.is_vector else 1 for i,r in self.info.records.items()])
        obs_rec_list = distribute_tasks(c.comm_rec, obs_rec_list_full, obs_rec_size)

        return obs_rec_list

    def state_to_obs(self, c: Context, tag: str, **kwargs) -> np.ndarray:
        """
        Compute the corresponding obs value given the state variable(s), namely the "obs_prior"
        This function includes several ways to compute the obs_prior:

        1, If obs_name is one of the variables provided by the model_src module, then
        model_src.read_var shall be able to provide the obs field defined on model native grid.
        Then we convert the obs field to the analysis grid and do vertical interpolation.

        2, If obs_name is one of the variables provided by obs.obs_operator, we call it to
        obtain the obs seq. Typically the obs_operator performs more complex computation, such
        as path integration, radiative transfer model, etc. (slowest)

        Args:
            c (Context): the runtime context object
            tag (str): 'prior' or 'post', or 'truth' if generating synthetic obs
            **kwargs: Additional parameters
                - member: int, member index; or None if dealing with synthetic obs
                - name: str, obs variable name
                - time: datetime obj, time of the obs window
                - is_vector: bool, if True the obs is a vector measurement
                - dataset_src: str, dataset source module name providing the obs
                - model_src: str, model source module name providing the state
                - x, y, z, t: np.array, coordinates from obs_seq

        Returns:
            np.ndarray: Values corresponding to the obs_seq but from the state identified by kwargs
        """
        # obs dataset source module
        dataset = c.datasets[kwargs['dataset_src']]

        # model source module
        model_name = kwargs['model_src']
        model = c.models[model_name]

        # z coords
        z = {}
        

        if kwargs['name'] in model.variables:
            # option 1:
            # if obs variable is one of the state variable, or can be computed by the model,
            # then we just need to collect the 3D variable and interpolate in x,y,z

            obs_x = np.array(kwargs['x'])
            obs_y = np.array(kwargs['y'])
            obs_z = np.array(kwargs['z'])
            nobs = len(obs_x)
            if kwargs['is_vector']:
                seq = np.full((2, nobs), np.nan)
            else:
                seq = np.full(nobs, np.nan)

            levels = model.variables[kwargs['name']].levels
            zp = None; vp = None; dzp = None; dzc = None;
            for k in range(len(levels)):
                kwargs['k'] = k

                fld_k = self.get_field(c, tag, **kwargs)
                z_k = self.get_z(c, **kwargs)
                
                horizontal_interp()
                seq, vp, zp, dzp = self.vertical_interp()

        elif kwargs['name'] in dataset.obs_operator:
            # option 2:
            # if dataset module provides an obs_operator, use it to compute obs
            operator = dataset.obs_operator[kwargs['name']]

            # get the obs seq from operator
            seq = c.io.call_io_method(c, tag, operator, model=model, grid=c.grid, mask=c.grid.mask, z=z, **kwargs)

        else:
            raise ValueError(f"unable to obtain obs prior for '{kwargs['name']}'")

        return seq

    def get_field(self, c: Context, tag: str, **kwargs) -> np.ndarray:
        """ Get a variable (2D field) from a model and conver to the analysis grid """
        name = kwargs['name']
        time = kwargs['time']
        k = kwargs['k']
        is_vector = kwargs['is_vector']
        model_name = kwargs['model_src']
        model = c.models[model_name]
        mem_id = kwargs['member']

        if name in [r['name'] for r in ensure_list(c.config.state_def)]:
            # the variable (name) is one of the state variables
            # we can find its corresponding rec_id and call io.read_field to get it
            rec_id_found = [i for i,r in c.state.info.fields.items() if r.name==name and r.time==time and r.k==k]
            if len(rec_id_found) == 0:
                raise RuntimeError("field '{name}' is defined in state_def but not found in state.info.fields")
            rec_id = rec_id_found[0]
            fld = c.io.read_field(c, tag, rec_id, mem_id)
        else:
            # otherwise, we get the field from by calling model.read_var
            model_fld = c.io.call_io_method(c, tag, model.read_var, **kwargs)

            # convert the model field to the analysis c.grid
            model.grid.set_destination_grid(c.grid)
            fld = model.grid.convert(model_fld, is_vector=is_vector, method=c.config.interp_method)
        return fld

    def get_z(self, c: Context, tag: str, **kwargs) -> np.ndarray:
        """ Get the z coords at level k on the analysis grid for a model """
        name = kwargs['name']
        time = kwargs['time']
        k = kwargs['k']
        model_name = kwargs['model_src']
        model = c.models[model_name]

        if c.config.z_coords_from == 'mean':
            ztag = 'z_mean'
        elif c.config.z_coords_from == 'member':
            ztag = 'z'
        else:
            raise ValueError("unknown config.z_coords_from: {c.config.z_coords_from}")

        if name in [r['name'] for r in ensure_list(c.config.state_def)]:
            # the variable is one of the state variables
            # we can find its corresponding rec_id and call io.read_field to get z coords
            # there can be multiple records (different state variables), we just take the first one
            rec_id = [i for i,r in c.state.info.fields.items() if r.time==time and r.model_src==model_name and r.k==k][0]
            rec = c.state.info.fields[rec_id]

            # read the z field with (mem_id=0, rec_id) from fields_z
            z_fld = c.io.read_field(c, ztag, rec_id, mem_id=0)
            if rec.is_vector:
                return z_fld[0, ...]
            else:
                return z_fld
        else:
            # otherwise, we get the z coords by calling model.z_coords
            if model.z and k in model.z:
                # just return the cached z coords from model
                model_z_fld = model.z[k]
            else:
                # TODO: ensemble mean of z calculation may take a lot of cost here? just use member=0 now
                model_z_fld = c.io.call_io_method(c, tag, model.z_coords, **kwargs)

            # convert to analysis c.grid
            model.grid.set_destination_grid(c.grid)
            z_fld = model.grid.convert(model_z_fld, is_vector=False, method=c.config.interp_method)
            return z_fld

    def horizontal_interp(self, ):

        ##horizontal interp field to obs_x,y, for current layer k
        ##TODO: move vertical interp to a separate function
        if is_vector:
            z = c.grid.interp(z[0, ...], obs_x, obs_y, method='nearest')
            zc = np.array([z, z])
            v1 = c.grid.interp(fld[0, ...], obs_x, obs_y, method='nearest')
            v2 = c.grid.interp(fld[1, ...], obs_x, obs_y, method='nearest')
            vc = np.array([v1, v2])
        else:
            zc = c.grid.interp(z, obs_x, obs_y, method='nearest')
            vc = c.grid.interp(fld, obs_x, obs_y, method='nearest')
        return vc, zc


    def vertical_interp(self,):
        ##vertical interp to obs_z, take ocean depth as example:
        ##    -------------------------------------------
        ##k-1 -  -  -  -  v[k-1], vp  }dzp  prevous layer
        ##    ----------  z[k-1], zp --------------------
        ##k   -  -  -  -  v[k],   vc  }dzc  current layer
        ##    ----------  z[k],   zc --------------------
        ##k+1 -  -  -  -  v[k+1]
        ##
        ##z at current level k denoted as zc, previous level as zp
        ##variables are considered layer averages, so they are at layer centers
        
        if k == 0:
            dzc = zc  ##layer thickness for first level

            ##the first level: constant vc from z=0 to dzc/2
            inds = (obs_z >= np.minimum(0, 0.5*dzc)) & (obs_z <= np.maximum(0, 0.5*dzc))
            seq[..., inds] = vc[..., inds]

        if k > 0:
            dzc = zc - zp  ##layer thickness for level k

            ##in between levels: linear interp between vp and vc
            assert dzp is not None
            assert vp is not None
            z_vp = zp - 0.5*dzp
            z_vc = zp + 0.5*dzc
            inds = (obs_z >= np.minimum(z_vp, z_vc)) & (obs_z <= np.maximum(z_vp, z_vc))
            ##there can be collapsed layers if z_vc=z_vp
            zdiff = z_vc - z_vp
            collapsed = (zdiff == 0)
            zdiff = np.where(collapsed, 1, zdiff)
            vi = ((z_vc - obs_z)*vc + (obs_z - z_vp)*vp) / zdiff
            vi = np.where(collapsed, vp, vi)

            ##TODO: still got some warnings here
            #with np.errstate(divide='ignore'):
            #    vi = np.where(z_vp==z_vc,
            #                vp,   ##for collapsed layers just use previous value
            #                ((z_vc - obs_z)*vc + (obs_z - z_vp)*vp)/(z_vc - z_vp) )  ##otherwise linear interp between layer
            seq[..., inds] = vi[..., inds]

        if k == len(levels)-1:
            ##the last level: constant vc from z=zc-dzc/2 to zc
            assert dzc is not None
            inds = (obs_z >= np.minimum(zc-0.5*dzc, zc)) & (obs_z <= np.maximum(zc-0.5*dzc, zc))
            seq[..., inds] = vc[..., inds]

        if k < len(levels)-1:
            assert dzc is not None
            ##make a copy of current layer as 'previous' for next k
            zp = zc.copy()
            vp = vc.copy()
            dzp = dzc.copy()
        return vp, zp, dzp

    def validate_seq_shape(self, seq: np.ndarray, is_vector: bool) -> None:
        """
        Validate the shape of an observation sequence. 
        Allowed shape: (nobs,) for scalar obs seq; (2, nobs) for vector obs seq.
        """
        if not isinstance(seq, np.ndarray):
            raise TypeError(f"obs sequence must be a numpy array, got {type(seq)}")

        shape = seq.shape
        if is_vector:
            if len(shape) != 2:
                raise ValueError(f"vector obs sequence must have shape (2, nobs), got {shape}")
            if shape[0] != 2:
                raise ValueError(f"vector obs sequence first dimension must be 2, got {shape[0]}")
        else:
            if len(shape) != 1:
                raise ValueError(f"scalar obs sequence must have shape (nobs,), got {shape}")

    def collect_obs_seq(self, c: Context) -> ObsSeq:
        """
        Process the obs in parallel, read dataset files and convert to obs_seq
        which contains obs value, coordinates and other info

        Since this is the actual obs (1 copy), only 1 processor needs to do the work

        Argss:
            c (Context): The runtime context object.

        Returns:
            ObsSeq: observation sequence. Dictionary {obs_rec_id (int): record}
                where each record is a dictionary {key: np.ndarray}, the mandatory keys are
                'obs' the observed values (measurements)
                'x', 'y', 'z', 't' the coordinates for each measurement
                'err_std' the uncertainties for each measurement
                there can be other optional keys provided by read_obs() but we don't use them
        """
        if c.pid == 0:
            print('>>> read observation sequence from datasets', flush=True)

        ##get obs_seq from dataset module, each pid_rec gets its own workload as a subset of obs_rec_list
        obs_seq = {}
        for obs_rec_id in self.obs_rec_list[c.pid_rec]:
            obs_rec = self.info.records[obs_rec_id]

            ##load the dataset module
            dataset = c.datasets[obs_rec.dataset_src]
            if obs_rec.name not in dataset.variables:
                raise ValueError(f"variable '{obs_rec.name}' not defined in dataset.{obs_rec.dataset_src}.variables")

            model = c.models[obs_rec.model_src]

            # get z coords on all z levels (to be sent to dataset class methods)
            k_list = list(set([r.k for i,r in c.state.info.fields.items() if r.time==obs_rec.time and r.model_src==obs_rec.model_src]))
            z = {}
            for k in k_list:
                z[k] = self.get_z(c, k=k, **obs_rec.asdict())

            if isinstance(dataset, SyntheticObs):  #using synthetic observation
                ##generate synthetic obs network
                seq = dataset.random_network(model=model, grid=c.grid, mask=c.grid.mask, z=z, **obs_rec.asdict())

                ##compute obs values
                seq['obs'] = self.state_to_obs(c, 'truth', member=None, **obs_rec.asdict(), **seq)

                ##perturb with obs err
                seq['obs'] += np.random.normal(0, 1, seq['obs'].shape) * obs_rec.err.std

            else:
                ##read dataset files and obtain obs sequence
                seq = dataset.read_obs(model=model, grid=c.grid, mask=c.grid.mask, z=z, **obs_rec.asdict())

            self.validate_seq_shape(seq['obs'], obs_rec.is_vector)

            if c.pid_mem == 0:
                print(f"number of '{obs_rec.name}' obs from '{obs_rec.dataset_src}': {seq['obs'].shape[-1]}", flush=True)

            ##misc. transform here
            for transform_func in c.transform_funcs:
                seq = transform_func.forward_obs(c, obs_rec, seq)

            obs_seq[obs_rec_id] = seq
            obs_rec.nobs = seq['obs'].shape[-1]  ##update nobs in obs_rec

        ##output obs sequence for debugging
        if c.config.debug and c.pid_mem == 0:
            for obs_rec_id, rec in obs_seq.items():
                c.io.save_debug_data(c, f'obs_seq.rec{obs_rec_id}', rec)

        return obs_seq

    def prepare_obs(self, c: Context) -> None:
        self.obs_seq = bcast_by_root(c.comm_mem)(self.collect_obs_seq)(c)

    def prepare_obs_from_state(self, c: Context, flag: str) -> None:
        """
        Compute the obs priors in parallel, run state_to_obs to obtain obs_prior_seq

        Args:
            c (Context): the runtime context object
            flag (str): 'prior' or 'post', or 'truth' if we are generating synthetic obs
        """
        mem_list = c.state.mem_list
        pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * c.config.nproc_mem + pid_mem_show
        c.print_1p('>>> compute observation priors\n')

        ##process the obs, each proc gets its own workload as a subset of
        ##all proc goes through their own task list simultaneously
        nr = len(self.obs_rec_list[c.pid_rec])
        nm = len(mem_list[c.pid_mem])
        for m, mem_id in enumerate(mem_list[c.pid_mem]):
            for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):
                ##this is the obs record to process
                obs_rec = self.info.records[obs_rec_id]

                if c.config.debug:
                    print(f"PID {c.pid:4}: obs_prior mem{mem_id+1:03} {obs_rec.name:20}", flush=True)
                else:
                    c.print_1p(progress_bar(m*nr+r, nr*nm))

                seq = {}
                ##need the coordinates for transform later
                for key in ['x', 'y', 'z', 't', 'err_std']:
                    seq[key] = self.obs_seq[obs_rec_id][key]
                ##obtain obs_prior values from model state
                seq['obs'] = self.state_to_obs(c, flag, member=mem_id, **obs_rec.asdict(), **self.obs_seq[obs_rec_id])

                ##misc. transform here
                for transform_func in c.transform_funcs:
                    seq = transform_func.forward_obs(c, obs_rec, seq)

                ##collect obs priors together
                if flag == 'prior':
                    self.obs_prior[mem_id, obs_rec_id] = seq['obs']
                elif flag == 'posterior':
                    self.obs_post[mem_id, obs_rec_id] = seq['obs']
                else:
                    raise ValueError
        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##output obs_prior sequeneces
        if c.config.debug:
            for key, seq in self.obs_prior.items():
                mem_id, obs_rec_id = key
                file = f'obs_prior_seq.rec{obs_rec_id}.mem{mem_id:03}'
                c.io.save_debug_data(c, file, {'obs_prior':seq})

    def global_obs_list(self, c: Context) -> list[tuple[ObsRecordID, Optional[int], ProcID, int]]:
        ##form the global list of obs (in serial mode the main loop is over this list)
        n_obs_rec = len(self.info.records)

        i = {}  ##location in full obs vector on owner pid
        for owner_pid in range(c.config.nproc_mem):
            i[owner_pid] = 0

        obs_list = []
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info.records[obs_rec_id]
            v_list = [0, 1] if obs_rec.is_vector else [None]
            for owner_pid in self.obs_inds[obs_rec_id].keys():
                for _ in self.obs_inds[obs_rec_id][owner_pid]:
                    for v in v_list:
                        obs_list.append((obs_rec_id, v, owner_pid, i[owner_pid]))
                        i[owner_pid] += 1

        if getattr(c, 'shuffle_obs', False):
            np.random.shuffle(obs_list)  ##randomize the order of obs (this is optional)

        return obs_list

    def transpose_obs_seq(self, c: Context, input_obs: ObsSeq) -> LocalObsSeq:
        """
        Transpose the obs sequence from field-complete to ensemble-complete

        Args:
            c (Context): the runtime context
            input_obs (ObsSeq): obs_seq from process_all_obs(), dict[obs_rec_id, dict[key, np.array]]     

        Returns,
            LocalObsSeq: the lobs dict[obs_rec_id, dict[par_id, dict[key, np.array]]], key = 'obs','x','y','z','t'...
        """
        mem_list = c.state.mem_list
        nproc_mem = c.config.nproc_mem
        pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * nproc_mem + pid_mem_show

        c.print_1p('transpose obs_seq to local obs\n')

        ##Step 1: transpose to ensemble-complete by exchanging mem_id, par_id in comm_mem
        ##        input_obs -> tmp_obs
        tmp_obs = {}  ##local obs at intermediate stage

        nr = len(self.obs_rec_list[c.pid_rec])
        for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in mem_list.items()])
            for m in range(nm_max):
                mem_id = None
                seq = None
                if m < len(mem_list[c.pid_mem]):
                    mem_id = mem_list[c.pid_mem][m]

                if c.config.debug:
                    if mem_id:
                        print(f"PID {c.pid:4}: transposing obs: mem{mem_id+1:03} obs_rec{obs_rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing obs: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##prepare the obs seq for sending if not at the end of mem_list
                if m < len(mem_list[c.pid_mem]):
                    mem_id = mem_list[c.pid_mem][m]
                    if mem_id == 0:  ##this is the obs seq, just let mem_id=0 send it
                        seq = input_obs[obs_rec_id].copy()

                ##the collective send/recv follows the same idea under state.transpose_field_to_state
                ##1) receive lobs_seq from src_pid, for src_pid<pid first
                for src_pid in np.arange(0, c.pid_mem):
                    if m < len(mem_list[src_pid]):
                        src_mem_id = mem_list[src_pid][m]
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

                ##2) send my obs chunk to a list of dst_pid, send to dst_pid>=pid first
                ##   then cycle back to send to dst_pid<pid. i.e. the dst_pid sequence is
                ##   [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
                if m < len(mem_list[c.pid_mem]):
                    for dst_pid in np.mod(np.arange(nproc_mem)+c.pid_mem, nproc_mem):
                        if mem_id == 0:
                            ##this is the obs seq with keys 'obs','err_std','x','y','z','t'
                            ##assemble the lobs_seq dict with same keys but subset obs_inds
                            ##do this for each par_id to get the full lobs_seq
                            lobs_seq = {}
                            for par_id in c.state.par_list[dst_pid]:
                                lobs_seq[par_id] = {}
                                inds = self.obs_inds[obs_rec_id][par_id]
                                assert seq is not None
                                for key in ('obs', 'err_std', 'x', 'y', 'z', 't'):
                                    lobs_seq[par_id][key] = seq[key][..., inds]

                            if dst_pid == c.pid_mem:
                                ##pid already stores the lobs_seq, just copy
                                tmp_obs[obs_rec_id] = lobs_seq
                            else:
                                ##send lobs_seq to dst_pid's lobs
                                c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

                ##3) finish receiving lobs_seq from src_pid, for src_pid>pid now
                for src_pid in np.arange(c.pid_mem+1, nproc_mem):
                    if m < len(mem_list[src_pid]):
                        src_mem_id = mem_list[src_pid][m]
                        if src_mem_id == 0:
                            tmp_obs[obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)
        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##Step 2: collect all obs records (all obs_rec_ids) on pid_rec
        ##        tmp_obs -> output_obs
        output_obs = {}
        for entry in c.comm_rec.allgather(tmp_obs):
            for key, data in entry.items():
                output_obs[key] = data
        c.comm.Barrier()

        return output_obs

    def transpose_to_ensemble_complete(self, c: Context, input_obs: ObsEns) -> LocalObsEns:
        """
        Transpose obs from field-complete to ensemble-complete

        Step 1, Within comm_mem, send the subset of input_obs with mem_id and par_id
        from the source proc (src_pid) to the destination proc (dst_pid), store the
        result in tmp_obs with all the mem_id (ensemble-complete)

        Step 2, Gather all obs_rec_id within comm_rec, so that each pid_rec will have the
        entire obs record for assimilation

        Args:
            c (Context): the runtime context
            input_obs (ObsEns): obs_prior from process_all_obs_priors(), dict[(mem_id, obs_rec_id), np.array];

        Returns,
            LocalObsEns: the lobs_prior dict[(mem_id, obs_rec_id), dict[par_id, np.array]]
        """
        mem_list = c.state.mem_list
        nproc_mem = c.config.nproc_mem
        pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * nproc_mem + pid_mem_show

        c.print_1p('transpose obs prior ensemble to local obs priors\n')

        ##Step 1: transpose to ensemble-complete by exchanging mem_id, par_id in comm_mem
        ##        input_obs -> tmp_obs
        tmp_obs = {}  ##local obs at intermediate stage

        nr = len(self.obs_rec_list[c.pid_rec])
        for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in mem_list.items()])
            for m in range(nm_max):
                mem_id = None
                seq = None
                ##prepare the obs seq for sending if not at the end of mem_list
                if m < len(mem_list[c.pid_mem]):
                    mem_id = mem_list[c.pid_mem][m]
                    seq = input_obs[mem_id, obs_rec_id].copy()

                if c.config.debug:
                    if mem_id:
                        print(f"PID {c.pid:4}: transposing obs: mem{mem_id+1:03} obs_rec{obs_rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing obs: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##the collective send/recv follows the same idea under state.transpose_field_to_state
                ##1) receive lobs_seq from src_pid, for src_pid<pid first
                for src_pid in np.arange(0, c.pid_mem):
                    if m < len(mem_list[src_pid]):
                        src_mem_id = mem_list[src_pid][m]
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

                ##2) send my obs chunk to a list of dst_pid, send to dst_pid>=pid first
                ##   then cycle back to send to dst_pid<pid. i.e. the dst_pid sequence is
                ##   [pid, pid+1, ..., nproc-1, 0, 1, ..., pid-1]
                if m < len(mem_list[c.pid_mem]):
                    for dst_pid in np.mod(np.arange(nproc_mem)+c.pid_mem, nproc_mem):
                        ##this is the obs prior seq for mem_id, obs_rec_id
                        ##for each par_id, assemble the subset lobs_seq using obs_inds
                        lobs_seq = {}
                        for par_id in c.state.par_list[dst_pid]:
                            inds = self.obs_inds[obs_rec_id][par_id]
                            assert seq is not None
                            lobs_seq[par_id] = seq[..., inds]

                        if dst_pid == c.pid_mem:
                            ##pid already stores the lobs_seq, just copy
                            tmp_obs[mem_id, obs_rec_id] = lobs_seq
                        else:
                            ##send lobs_seq to dst_pid
                            c.comm_mem.send(lobs_seq, dest=dst_pid, tag=m)

                ##3) finish receiving lobs_seq from src_pid, for src_pid>pid now
                for src_pid in np.arange(c.pid_mem+1, nproc_mem):
                    if m < len(mem_list[src_pid]):
                        src_mem_id = mem_list[src_pid][m]
                        tmp_obs[src_mem_id, obs_rec_id] = c.comm_mem.recv(source=src_pid, tag=m)

        c.comm.Barrier()
        c.print_1p(' done.\n')

        ##Step 2: collect all obs records (all obs_rec_ids) on pid_rec
        ##        tmp_obs -> output_obs
        output_obs = {}
        for entry in c.comm_rec.allgather(tmp_obs):
            for key, data in entry.items():
                output_obs[key] = data
        c.comm.Barrier()

        return output_obs

    def transpose_to_field_complete(self, c: Context, lobs: LocalObsEns) -> ObsEns:
        """
        Transpose obs from ensemble-complete to field-complete

        Args:
            c (Context): the runtime context
            lobs (LocalObsEns): ensemble-complete local obs

        Returns:
            ObsEns: field-complete obs_seq ensemble
        """
        mem_list = c.state.mem_list
        nproc_mem = c.config.nproc_mem
        pid_mem_show = [p for p,lst in mem_list.items() if len(lst)>0][0]
        pid_rec_show = [p for p,lst in self.obs_rec_list.items() if len(lst)>0][0]
        c.pid_show =  pid_rec_show * nproc_mem + pid_mem_show

        c.print_1p('obs post sequences: ')
        c.print_1p('transpose local obs to obs\n')

        obs_seq = {}
        nr = len(self.obs_rec_list[c.pid_rec])
        for r, obs_rec_id in enumerate(self.obs_rec_list[c.pid_rec]):

            ##all pid goes through their own mem_list simultaneously
            nm_max = np.max([len(lst) for p,lst in mem_list.items()])
            for m in range(nm_max):
                mem_id = None
                seq = None
                if m < len(mem_list[c.pid_mem]):
                    mem_id = mem_list[c.pid_mem][m]
                    rec = self.info.records[obs_rec_id]
                    ##prepare an empty obs_seq for receiving if not at the end of mem_list
                    if rec.is_vector:
                        seq = np.full((2, rec.nobs), np.nan)
                    else:
                        seq = np.full((rec.nobs,), np.nan)

                if c.config.debug:
                    if mem_id:
                        print(f"PID {c.pid:4}: transposing obs: mem{mem_id+1:03} obs_rec{obs_rec_id}")
                    else:
                        print(f"PID {c.pid:4}: transposing obs: waiting")
                else:
                    c.print_1p(progress_bar(r*nm_max+m, nr*nm_max))

                ##this is just the reverse of transpose_obs_to_lobs
                ## we take the exact steps, but swap send and recv operations here
                ##
                ## 1) send my lobs to dst_pid, for dst_pid<pid first
                for dst_pid in np.arange(0, c.pid_mem):
                    if m < len(mem_list[dst_pid]):
                        dst_mem_id = mem_list[dst_pid][m]
                        c.comm_mem.send(lobs[dst_mem_id, obs_rec_id], dest=dst_pid, tag=m)

                ## 2) receive fld_chk from a list of src_pid, from src_pid>=pid first
                ##    because they wait to send stuff before able to receive themselves,
                ##    cycle back to receive from src_pid<pid then.
                if m < len(mem_list[c.pid_mem]):
                    assert mem_id is not None
                    for src_pid in np.mod(np.arange(nproc_mem)+c.pid_mem, nproc_mem):
                        if src_pid == c.pid_mem:
                            ##pid already stores the lobs_seq, just copy
                            lobs_seq = lobs[mem_id, obs_rec_id].copy()
                        else:
                            ##send lobs_seq to dst_pid
                            lobs_seq = c.comm_mem.recv(source=src_pid, tag=m)

                        ##unpack the lobs_seq to form a complete seq
                        for par_id in c.state.par_list[src_pid]:
                            inds = self.obs_inds[obs_rec_id][par_id]
                            assert seq is not None
                            seq[..., inds] = lobs_seq[par_id]

                        obs_seq[mem_id, obs_rec_id] = seq

                ## 3) finish sending lobs_seq to dst_pid, for dst_pid>pid now
                for dst_pid in np.arange(c.pid_mem+1, nproc_mem):
                    if m < len(mem_list[dst_pid]):
                        dst_mem_id = mem_list[dst_pid][m]
                        c.comm_mem.send(lobs[dst_mem_id, obs_rec_id], dest=dst_pid, tag=m)
        c.comm.Barrier()
        c.print_1p(' done.\n')
        return obs_seq

    def pack_local_obs_data(self, c: Context, par_id: PartitionID, lobs: LocalObsSeq, lobs_prior: LocalObsEns) -> dict:
        """pack lobs and lobs_prior into arrays for the jitted functions"""
        n_obs_rec = len(self.info.records)        ##number of obs records
        n_state_var = len(c.state.info.variables)   ##number of state variable names

        ##filter out obs with nan in obs_prior, valid index stored as subset of local_inds
        nlobs = 0  ##number of local obs on partition
        self.valid = {}
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info.records[obs_rec_id]
            v_list = [0, 1] if obs_rec.is_vector else [None]
            values = np.stack([lobs_prior[m, obs_rec_id][par_id][v, :].flatten() for m in range(c.config.nens) for v in v_list], axis=0)
            no_nan_mask = ~np.isnan(values).any(axis=0)
            self.valid[obs_rec_id] = np.where(no_nan_mask)[0].tolist()
            nlobs += len(self.valid[obs_rec_id]) * len(v_list)

        data = {}
        data['obs_rec_id'] = np.zeros(nlobs, dtype=int)
        data['obs'] = np.full(nlobs, np.nan)
        data['x'] = np.full(nlobs, np.nan)
        data['y'] = np.full(nlobs, np.nan)
        data['z'] = np.full(nlobs, np.nan)
        data['t'] = np.full(nlobs, np.nan)
        data['err_std'] = np.full(nlobs, np.nan)
        data['obs_prior'] = np.full((c.config.nens, nlobs), np.nan)
        data['used'] = np.full(nlobs, False)
        data['hroi'] = np.ones(n_obs_rec)
        data['vroi'] = np.ones(n_obs_rec)
        data['troi'] = np.ones(n_obs_rec)
        data['impact_on_state'] = np.ones((n_obs_rec, n_state_var))

        i = 0
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info.records[obs_rec_id]
            v_list = [0, 1] if obs_rec.is_vector else [None]

            data['hroi'][obs_rec_id] = obs_rec.hroi
            data['vroi'][obs_rec_id] = obs_rec.vroi
            data['troi'][obs_rec_id] = obs_rec.troi
            for state_var_id in range(len(c.state.info.variables)):
                state_vname = c.state.info.variables[state_var_id]
                data['impact_on_state'][obs_rec_id, state_var_id] = obs_rec.impact_on_state[state_vname]

            valid = self.valid[obs_rec_id]
            local_inds = self.obs_inds[obs_rec_id][par_id]
            d = len(local_inds[valid])
            ##append obs and obs prior records to the full array
            for v in v_list:
                data['obs_rec_id'][i:i+d] = obs_rec_id
                data['obs'][i:i+d] = np.squeeze(lobs[obs_rec_id][par_id]['obs'][v, valid])
                data['x'][i:i+d] = lobs[obs_rec_id][par_id]['x'][valid]
                data['y'][i:i+d] = lobs[obs_rec_id][par_id]['y'][valid]
                data['z'][i:i+d] = lobs[obs_rec_id][par_id]['z'][valid].astype(np.float32)
                data['t'][i:i+d] = np.array([t2h(t) for t in lobs[obs_rec_id][par_id]['t'][valid]])
                data['err_std'][i:i+d] = lobs[obs_rec_id][par_id]['err_std'][valid]
                for m in range(c.config.nens):
                    data['obs_prior'][m, i:i+d] = np.squeeze(lobs_prior[m, obs_rec_id][par_id][v, valid].copy())
                i += d

        return data

    def unpack_local_obs_data(self, c: Context, par_id: PartitionID, lobs: LocalObsSeq, lobs_prior: LocalObsEns, data: dict) -> None:
        """unpack data and write back to the original lobs_prior dict"""
        n_obs_rec = len(self.info.records)
        i = 0
        for obs_rec_id in range(n_obs_rec):
            obs_rec = self.info.records[obs_rec_id]

            valid = self.valid[obs_rec_id]
            local_inds = self.obs_inds[obs_rec_id][par_id]
            d = len(local_inds[valid])
            v_list = [0, 1] if obs_rec.is_vector else [None]
            for v in v_list:
                for m in range(c.config.nens):
                    lobs_prior[m, obs_rec_id][par_id][v, valid] = data['obs_prior'][m, i:i+d]
                i += d
