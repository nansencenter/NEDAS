"""
NEDAS offline filter scheme provides two assimilation modes:

In batch mode, the analysis domain is divided into small local partitions (indexed by `par_id`) and each `pid_mem` solves the analysis for its own list of `par_id`. The local observations are those falling inside the localization radius for each [`par_id`,`rec_id`]. The "local analysis" for each state variable is computed using the matrix-version ensemble filtering equations (such as [LETKF](https://doi.org/10.1016/j.physd.2006.11.008), [DEnKF](https://doi.org/10.1111/j.1600-0870.2007.00299.x)). The batch mode is favorable when the local observation volume is small and the matrix solution allows more flexible error covariance modeling (e.g., to include correlations in observation errors).

In serial mode, we go through the observation sequence and assimilation one observation at a time. Each `pid` stores a subset of state variables and observations with `par_id`, here locality doesn't matter in storage, the `pid` owning the observation being assimilated will first compute observation-space increments, then broadcast them to all the `pid` with state\_prior and/or lobs\_prior within the observation's localization radius and they will be updated. For the next observation, the updated observation priors will be used for computing increments. The whole process iteratively updates the state variables on each `pid`. The serial mode is more scalable especially for inhomogeneous network where load balancing is difficult, or when local observation volume is large. The scalar update equations allow more flexible use of nonlinear filtering approaches (such as particle filter, rank regression).

NEDAS allows flexible modifications in the interface between model/dataset modules and the core assimilation algorithms, to achieve more sophisticated functionality:

Multiple time steps can be added in the `time` dimension for the state and/or observations to achieve ensemble smoothing instead of filtering. Iterative smoothers can also be formulated by running the analysis cycle as an outer-loop iteration (although they can be very costly).

Miscellaneous transform functions can be added for state and/or observations, for example, Gaussian anamorphosis to deal with non-Gaussian variables; spatial bandpass filtering to run assimilation for "scale components" in multiscale DA; neural networks to provide a nonlinear mapping between the state space and observation space, etc.


Description of Key Variables and Functions

```{figure} ../imgs/workflow.png
:width: 100%
:align: center
```
<!--| **Figure 3**. Workflow for one assimilation cycle/iteration. For the sake of clarity, only the key variables and functions are shown. Black arrows show the flow of information through functions.| -->

Indices and lists:

* For each processor, its `pid` is the rank in the communicator `comm` with size `nproc`. The `comm` is split into `comm_mem` and `comm_rec`. Processors in `comm_mem` belongs to the same record group, with `pid_mem` in `[0:nproc_mem]`. Processors in `comm_rec` belongs to the same member group, with `pid_rec` in `[0:nproc_rec]`. Note that `nproc = nproc_mem * nproc_rec`, user should set `nproc` and `nproc_mem` in the config file.

* `mem_list`[`pid_mem`] is a list of members `mem_id` for processors with `pid_mem` to handle.

* `rec_list`[`pid_rec`] is a list of field records `rec_id` for processors with `pid_rec` to handle.

* `obs_rec_list`[`pid_rec`] is a list of observation records `obs_rec_id` for processors with `pid_rec` to handle.

* `partitions` is a list of tuples `(istart, iend, di, jstart, jend, dj)` defining the partitions of the 2D analysis domain, each partition holds a slice `[istart:iend:di, jstart:jend:dj]` of the field and is indexed by `par_id`.

* `par_list`[`pid_mem`] is a list of partition id `par_id` for processor with `pid_mem` to handle.

* `obs_inds`[`obs_rec_id`][`par_id`] is the indices in the entire observation record `obs_rec_id` that belong to the local observation sequence for partition `par_id`.


Data structures:

* `fields_prior`[`mem_id`, `rec_id`] points to the 2D fields `fld[...]` (np.array).

* `z_fields`[`mem_id`, `rec_id`] points to the z coordinate fields `z[...]` (np.array).

* `state_prior`[`mem_id`, `rec_id`][`par_id`] points to the field chunk `fld_chk` (np.array) in the partition.

* `obs_seq`[`obs_rec_id`] points to observation sequence `seq` that is a dictionary with keys ('obs', 't', 'z', 'y', 'x', 'err\_std') each pointing to a list containing the entire record.

* `lobs`[`obs_rec_id`][`par_id`] points to local observation sequence `lobs_seq` that is a dictionary with same keys as `seq` but the lists only contain a subset of the record.

* `obs_prior_seq`[`mem_id`, `obs_rec_id`] points to the observation prior sequence (np.array), same length with `seq['obs']`.

* `lobs_prior`[`mem_id`, `obs_rec_id`][`par_id`] points to the local observation prior sequence (np.array), same length with `lobs_seq`.


Functions:

* `prepare_state()`: For member `mem_id` in `mem_list` and field record `rec_id` in `rec_list`, load the model module and `read_var()` gets the variables in model native grid, convert to analysis grid, and apply miscellaneous user-defined transforms. Also, get z coordinates in the same prodcedure using `z_coords()` functions. Returns `fields_prior` and `z_fields`.

* `prepare_obs()`: For observation record `obs_rec_id` in `obs_rec_list`, load the dataset module and `read_obs()` get the observation sequence. Apply miscellaneous user-defined transforms if necessary. Returns `obs_seq`.

* `assign_obs()`: According to ('y', 'x') coordinates for `par_id` and ('variable', 'time', 'z') for `rec_id`, sort the full observation sequence `obs_rec_id` to find the indices that belongs to the local observation subset. Returns `obs_inds`.

* `prepare_obs_from_state()`: For member `mem_id` in `mem_list` and observation record `obs_rec_id` in `obs_rec_list`, compute the observation priors from model state. There are three ways to get the observation priors: 1) if the observed variable is one of the state variables, just get the variable with `read_field()`, or 2) if the observed variable can be provided by model module, then get it through `read_var()` and convert to analysis grid. These two options obtains observed variables defined on the analysis grid, then we convert them to the observing network and interpolate to the observed z location. Option 3) if the observation is a complex function of the state, the user can provide `obs_operator()` in the dataset module to compute the observation priors. Finally, the same miscellaneous user-defined transforms can be applied. Returns `obs_prior_seq`.

* `transpose_field_to_state()`: Transposes field-complete `field_prior` to ensemble-complete `state_prior` (illustrated in Fig.1). After assimilation, the reverse `transpose_state_to_field()` transposes `state_post` back to field-complete `fields_post`.

* `transpose_obs_to_lobs()`: Transpose the `obs_seq` and `obs_prior_seq` to their ensemble-complete counterparts `lobs` and `lobs_prior` (illustrated in Fig. 2).

* `batch_assim()`: Loop through the local state variables in `state_prior`, for each state variable, the local observation sequence is sorted based on the localization and impact factors. If the local observation sequence is not empty, compute the `local_analysis()` to update state variables, save to `state_post` and return.

* `serial_assim()`: Loop through the observation sequence, for each observation, the processor storing this observation will compute `obs_increment()` and broadcast. For all processors, if some of its local state/observations are within the localization radius of this observation, compute `update_local_ens()` to update these state/observations. Do this iteratively for all observations until end of sequence. Returns the updated local `state_post`.

* `update()`: Take `state_prior` and `state_post`, apply miscellaneous user-defined inverse transforms, compute `analysis_incr()`, convert the increment back to model native grid and add the increments to the model variables int he restart files. Apart from simply adding the increments, some other post-processing steps can be implemented, for example using the increments to compute optical flows and align the model variables instead.


"""

import os
import sys
import tempfile
import importlib
import subprocess
import numpy as np
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.progress import timer, progress_bar
from NEDAS.utils.shell_utils import makedir, run_command, run_job
from NEDAS.utils.parallel import Scheduler, bcast_by_root, distribute_tasks
from NEDAS.utils.random_perturb import random_perturb
from NEDAS.schemes.base import AnalysisScheme

class OfflineFilterAnalysisScheme(AnalysisScheme):
    """
    Subclass for cycling scheme with filter and forecast steps
    """
    def run_step(self, c, step, mpi=False):
        """
        Run the python script from external call
        This is useful when several steps are using different strategy in parallelism
        For offline filter analysis, the filter step is run with mpi4py
        while the ensemble forecast uses a custom scheduler, other steps can be run in serial
        """
        script_file = os.path.abspath(__file__)

        # create a temporary config yaml file to hold c, and pass into program through runtime arg
        with tempfile.NamedTemporaryFile(dir=c.work_dir,
                                         prefix='config',
                                         suffix='.yml') as tmp_config_file:
            c.dump_yaml(tmp_config_file.name)

            print(f"\n\033[1;33mRUNNING\033[0m {step} step")
            if c.debug:
                print(f"config file: {tmp_config_file.name}")

            ##build run commands for the ensemble forecast script
            commands = ""
            if c.python_env:
                commands = f". {c.python_env}; "
            if mpi:
                if importlib.util.find_spec("mpi4py") is not None:
                    commands += f"JOB_EXECUTE {sys.executable} -m mpi4py {script_file} -c {tmp_config_file.name}"
                else:
                    print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
                    commands += f"{sys.executable} {script_file} -c {tmp_config_file.name} --nproc=1"
            else:
                commands += f"{sys.executable} {script_file} -c {tmp_config_file.name}"
            commands += f" --step {step}"

            if c.debug:
                print(commands)

            if mpi:
                job_opts = {
                    'job_name': step,
                    'run_dir': c.cycle_dir(c.time),
                    'nproc': c.nproc,
                    **(c.job_submit or {}),
                    }
                run_job(commands, **job_opts)
            else:
                subprocess.run(commands, stdout=sys.stdout, stderr=sys.stderr, shell=True)

    def get_task_opts(self, c, **other_opts):
        opts = {
            'time': c.time,
            'forecast_period': c.cycle_period,
            'time_start': c.time_start,
            'time_end': c.time_end,
            'debug': c.debug,
            **(c.job_submit or {}),
            **other_opts,
            }
        return opts

    def get_restart_dir(self, c, model_name):
        model = c.model_config[model_name]
        if c.time == c.time_start:
            restart_dir = model.ens_init_dir
        else:
            restart_dir = c.forecast_dir(c.prev_time, model_name)
        print(f"using restart files in {restart_dir}", flush=True)
        return restart_dir

    def run_ensemble_tasks_in_scheduler(self, c, name, func, opts, nproc_per_run, walltime=None):

        ##get number of workers to initialize the scheduler
        if c.job_submit and c.job_submit.get('run_separate_jobs', False):
            ##all jobs will be submitted to external scheduler's queue
            ##just assign a worker to each ensemble member
            nworker = np.min(c.nens, c.nproc)
        else:
            ##Scheduler will use nworkers to spawn ensemble member runs to
            ##the available nproc processors
            nworker = c.nproc // nproc_per_run
        scheduler = Scheduler(nworker, walltime, debug=c.debug)

        for mem_id in range(c.nens):
            scheduler.submit_job(name+f"_mem{mem_id+1:03}", func, member=mem_id, **opts)

        scheduler.start_queue() ##start the job queue
        scheduler.shutdown()
        print(' done.', flush=True)

    def ensemble_forecast(self, c):
        for model_name, model in c.model_config.items():
            path = c.forecast_dir(c.time, model_name)
            makedir(path)
            print(f"Running {model_name} ensemble forecast:", flush=True)

            if model.ens_run_type == 'batch':
                opts = self.get_task_opts(c, path=path, nens=c.nens)
                model.run_batch(**opts)

            elif model.ens_run_type == 'scheduler':
                opts = self.get_task_opts(c, path=path)
                walltime = getattr(model, 'walltime', None)
                self.run_ensemble_tasks_in_scheduler(c, f'forecast_{model_name}', model.run, opts, model.nproc_per_run, walltime)

            else:
                raise ValueError(f"Unknown ensemble run type {model.ens_run_type} for {model_name}")

    def preprocess(self, c):
        for model_name, model in c.model_config.items():
            path = c.forecast_dir(c.time, model_name)
            makedir(path)
            print(f"Preprocessing {model_name} state:", flush=True)
            restart_dir = self.get_restart_dir(c, model_name)
            opts = self.get_task_opts(c, path=path, restart_dir=restart_dir)
            self.run_ensemble_tasks_in_scheduler(c, f'preproc_{model_name}', model.preprocess, opts, model.nproc_per_util)

    def postprocess(self, c):
        for model_name, model in c.model_config.items():
            path = c.forecast_dir(c.time, model_name)
            makedir(path)
            print(f"Postprocessing {model_name} state:", flush=True)
            restart_dir = self.get_restart_dir(c, model_name)
            opts = self.get_task_opts(c, path=path, restart_dir=restart_dir)
            self.run_ensemble_tasks_in_scheduler(c, f'postproc_{model_name}', model.postprocess, opts, model.nproc_per_util)

    def perturb(self, c):
        if c.perturb is None:
            c.print_1p(f"No perturbation defined in config, exiting.\n")
            return
        c.print_1p(f"Perturbing state:")

        ##clean perturb files in current cycle dir
        for rec in c.perturb:
            perturb_dir = os.path.join(c.forecast_dir(c.time, rec['model_src']), 'perturb')
            if c.pid==0:
                run_command(f"rm -rf {perturb_dir}; mkdir -p {perturb_dir}")
        c.comm.Barrier()

        ##distribute perturbation items among MPI ranks
        task_list = bcast_by_root(c.comm)(self.distribute_perturb_tasks)(c)

        c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

        ##first go through the fields to count how many (for showing progress)
        nfld = 0
        for rec in task_list[c.pid]:
            model_name = rec['model_src']
            model = c.model_config[model_name]
            vname = ensure_list(rec['variable'])[0]
            dt = model.variables[vname]['dt']
            nstep = c.cycle_period // dt + 1
            for n in range(nstep):
                for k in model.variables[vname]['levels']:
                    nfld += 1

        ##actually go through the fields to perturb now
        fld_id = 0
        for rec in task_list[c.pid]:
            model_name = rec['model_src']
            model = c.model_config[model_name]  ##model class object
            mem_id = rec['member']
            mstr = f'_mem{mem_id+1:03d}'
            path = c.forecast_dir(c.time, model_name)
            variable_list = ensure_list(rec['variable'])

            ##check if previous perturb is available from past cycles
            perturb = {}
            for vname in variable_list:
                psfile = os.path.join(c.forecast_dir(c.prev_time, model_name), 'perturb', vname+mstr+'.npy')
                if os.path.exists(psfile):
                    perturb[vname] = np.load(psfile)
                else:
                    perturb[vname] = None

            ##perturb all sub time steps for variables within this cycle
            dt = model.variables[vname]['dt']   ##time interval for variable vname in this cycle
            nstep = c.cycle_period // dt + 1
            for n in range(nstep):
                t = c.time + n * dt * dt1h

                #TODO: only works for surface layer variables now with k=0 (forcing variables)
                ##      but can be extended to 3D variables with additional vertical corr parameter
                for k in model.variables[vname]['levels']:
                    fld_id += 1
                    if c.debug:
                        print(f"PID {c.pid:4}: perturbing mem{mem_id+1:03} {variable_list} at {t} level {k}", flush=True)
                    else:
                        c.print_1p(progress_bar(fld_id, nfld+1))

                    vname =variable_list[0]  ##note: all variables in the list shall have same dt and k levels
                    model.read_grid(path=path, name=vname, time=t, member=mem_id, k=k)
                    model.grid.set_destination_grid(c.grid)
                    c.grid.set_destination_grid(model.grid)

                    fields = {}
                    for vname in variable_list:
                        ##read variable from model state
                        fld = model.read_var(path=path, name=vname, time=t, member=mem_id, k=k)
                        ##convert to analysis grid
                        fields[vname] = model.grid.convert(fld, is_vector=model.variables[vname]['is_vector'])

                    ##generate perturbation on analysis grid
                    fields_pert, perturb = random_perturb(c.grid, fields, prev_perturb=perturb, dt=dt, n=n, **rec)

                    if rec['type'].split(',')[0]=='displace' and hasattr(model, 'displace'):
                        ##use model internal method to apply displacement perturbations directly
                        model.displace(perturb, path=path, time=t, member=mem_id, k=k)
                    else:
                        ##convert from analysis grid to model grid, and
                        ##write the perturbed variables back to model state files
                        for vname in variable_list:
                            fld = c.grid.convert(fields_pert[vname], is_vector=model.variables[vname]['is_vector'])
                            model.write_var(fld, path=path, name=vname, time=t, member=mem_id, k=k)

            ##save a copy of perturbation at next_t, for use by next cycle
            for vname in variable_list:
                psfile = os.path.join(path, 'perturb', vname+mstr+'.npy')
                run_command(f"mkdir -p {os.path.dirname(psfile)}")
                np.save(psfile, perturb[vname])

        c.comm.Barrier()
        c.print_1p(' done.\n')

    def distribute_perturb_tasks(self, c):
        task_list_full = []
        for perturb_rec in ensure_list(c.perturb):
            for mem_id in range(c.nens):
                task_list_full.append({**perturb_rec, 'member':mem_id})
        task_list = distribute_tasks(c.comm, task_list_full)
        return task_list

    def diagnose(self, c):
        c.print_1p(f"Running diagnostics:")

        ##get task list for each rank
        task_list = bcast_by_root(c.comm)(self.distribute_diag_tasks)(c)

        ##the processor with most work load will show progress messages
        c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

        ##init file locks for collective i/o
        self.init_file_locks(c)

        ntask = len(task_list[c.pid])
        for task_id, rec in enumerate(task_list[c.pid]):
            if c.debug:
                print(f"PID {c.pid:4} running diagnostics '{rec['method']}'", flush=True)
            else:
                c.print_1p(progress_bar(task_id, ntask))

            method_name = f"NEDAS.diag.{rec['method']}"
            mod = importlib.import_module(method_name)

            ##perform the diag task
            mod.run(c, **rec)

        c.comm.Barrier()
        c.print_1p(' done.\n')
        c.comm.cleanup_file_locks()

    def distribute_diag_tasks(self, c):
        """Build the full task list and distribute among mpi ranks"""
        task_list_full = []
        for rec in ensure_list(c.diag):
            ##load the module for the given method
            method_name = f"NEDAS.diag.{rec['method']}"
            module = importlib.import_module(method_name)
            ##module returns a list of tasks to be done by each processor
            if not hasattr(module, 'get_task_list'):
                task_list_full.append(rec)
                continue
            task_list_rec = module.get_task_list(c, **rec)
            for task in task_list_rec:
                task_list_full.append(task)
        ##collected full list of tasks is evenly distributed across the mpi communicator
        task_list = distribute_tasks(c.comm, task_list_full)
        return task_list

    def init_file_locks(self, c):
        """Build the full task list for the diagnostics part of the config"""
        for rec in ensure_list(c.diag):
            ##load the module for the given method
            method_name = f"NEDAS.diag.{rec['method']}"
            module = importlib.import_module(method_name)
            ##module get_file_list returns a list of files for collective i/o
            if not hasattr(module, 'get_file_list'):
                continue
            files = module.get_file_list(c, **rec)
            for file in files:
                ##create the file lock across mpi ranks for this file
                c.comm.init_file_lock(file)

    def filter(self, c):
        """
        Main method for performing the analysis step

        Args:
            c (Config): runtime configuration object
        """
        self.validate_mpi_environment(c)

        ##multiscale approach: loop over scale components and perform assimilation on each scale
        ##more complex outer loops can be implemented here
        analysis_grid = c.grid
        for c.scale_id in range(c.nscale):
            c.print_1p(f"Running analysis for scale {c.scale_id}:")

            self.init_analysis_dir(c)
            c.grid = analysis_grid.change_resolution_level(c.resolution_level[c.scale_id])
            c.misc_transform = self.get_misc_transform(c)
            c.localization_funcs = self.get_localization_funcs(c)
            c.inflation_func = self.get_inflation_func(c)

            state = self.get_state(c)
            timer(c)(state.prepare_state)(c)

            obs = self.get_obs(c, state)
            timer(c)(obs.prepare_obs)(c, state)
            timer(c)(obs.prepare_obs_from_state)(c, state, 'prior')

            assimilator = self.get_assimilator(c)
            timer(c)(assimilator.assimilate)(c, state, obs)

            updator = self.get_updator(c)
            timer(c)(updator.update)(c, state)

    def __call__(self, c):
        c.show_summary()

        print("Cycling start...", flush=True)
        while c.time < c.time_end:
            print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

            os.system("mkdir -p "+c.cycle_dir(c.time))

            if c.run_preproc:
                self.run_step(c, 'preprocess')
                self.run_step(c, 'perturb', mpi=True)

            ##assimilation step
            if c.run_analysis and c.time >= c.time_analysis_start and c.time <= c.time_analysis_end:
                self.run_step(c, 'filter', mpi=True)
                self.run_step(c, 'postprocess')

            ##advance model state to next analysis cycle
            if c.run_forecast:
                self.run_step(c, 'ensemble_forecast')

            ##compute diagnostics
            if c.run_diagnose:
                self.run_step(c, 'diagnose', mpi=True)

            ##advance to next cycle
            c.time = c.next_time

        print("Cycling complete.", flush=True)

if __name__ == '__main__':
    # get config from runtime args, including the step to run (from --step)
    from NEDAS.config import Config
    c = Config(parse_args=True)

    scheme = OfflineFilterAnalysisScheme()
    step = getattr(scheme, c.step)
    timer(c)(step)(c)
