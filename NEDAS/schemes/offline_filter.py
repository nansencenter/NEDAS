import os
import sys
import tempfile
import importlib.util
import subprocess
import numpy as np
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.progress import timer, progress_bar
from NEDAS.utils.shell_utils import makedir, run_command, run_job
from NEDAS.utils.parallel import Scheduler, bcast_by_root, distribute_tasks
from NEDAS.utils.random_perturb import random_perturb
from NEDAS import assim_tools
from .filter import FilterAnalysisScheme

class OfflineFilterAnalysisScheme(FilterAnalysisScheme):
    """
    Offline filtering analysis scheme class.

    Using restart files I/O to pass information between the steps in a filtering analysis.
    """
    def __call__(self, c) -> None:
        c.show_summary()

        if c.step:
            ##if --step=STEP is specified at runtime, just run STEP and quit
            if c.step in ['perturb', 'filter', 'diagnose']:
                mpi = True
            else:
                mpi = False
            self.run_step(c, c.step, mpi)
            return

        print("Cycling start...", flush=True)
        while c.time < c.time_end:
            print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

            os.system("mkdir -p "+c.cycle_dir(c.time))

            if c.run_preproc:
                self.run_step(c, 'preprocess', mpi=False)
                if c.perturb:
                    self.run_step(c, 'perturb', mpi=True)

            ##assimilation step
            if c.run_analysis and c.time >= c.time_analysis_start and c.time <= c.time_analysis_end:
                self.run_step(c, 'filter', mpi=True)
                if c.run_postproc:
                    self.run_step(c, 'postprocess', mpi=False)

            ##advance model state to next analysis cycle
            if c.run_forecast:
                self.run_step(c, 'ensemble_forecast', mpi=False)

            ##compute diagnostics
            if c.run_diagnose:
                if c.diag:
                    self.run_step(c, 'diagnose', mpi=True)

            ##advance to next cycle
            c.time = c.next_time

        print("Cycling complete.", flush=True)

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
        for c.iter in range(c.niter):
            c.print_1p(f"Running analysis for outer iteration step {c.iter}:")

            self.init_analysis_dir(c)
            c.grid = analysis_grid.change_resolution_level(c.resolution_level[c.iter])
            c.transform_funcs = assim_tools.transforms.get_transform_funcs(c)
            c.localization_funcs = assim_tools.localization.get_localization_funcs(c)
            c.inflation_func = assim_tools.inflation.get_inflation_func(c)

            state = assim_tools.state.State(c)
            timer(c)(state.prepare_state)(c)

            obs = assim_tools.obs.Obs(c, state)
            timer(c)(obs.prepare_obs)(c, state)
            timer(c)(obs.prepare_obs_from_state)(c, state, 'prior')

            assimilator = assim_tools.assimilators.get_assimilator(c)
            timer(c)(assimilator.assimilate)(c, state, obs)

            updator = assim_tools.updators.get_updator(c)
            timer(c)(updator.update)(c, state)

    def ensemble_forecast(self, c):
        """
        Ensemble forecast step.

        """
        for model_name, model in c.models.items():
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
        """
        Pre-processing step before the forecast.

        This step prepares the necessary files (static data, boundary condition, namelist parameters etc.).
        Restart files from the previous step (during cycling) or from ``ens_init_dir`` (at first cycle)
        will be used as initial conditions.

        The ``preprocess`` method implemented in each model class details this step.
        """
        for model_name, model in c.models.items():
            path = c.forecast_dir(c.time, model_name)
            makedir(path)
            print(f"Preprocessing {model_name} state:", flush=True)
            restart_dir = self.get_restart_dir(c, model_name)
            opts = self.get_task_opts(c, path=path, restart_dir=restart_dir)
            self.run_ensemble_tasks_in_scheduler(c, f'preproc_{model_name}', model.preprocess, opts, model.nproc_per_util)

    def postprocess(self, c):
        """
        Post-processing step after the assimilation and before the next forecast.

        This step takes posterior model states after data assimilation. Run post-processing algorithm to remove any
        non-physical values and ensure the next model forecasts will run correctly.

        The ``postprocess`` method implemented in each model class details this step.
        """
        for model_name, model in c.models.items():
            path = c.forecast_dir(c.time, model_name)
            makedir(path)
            print(f"Postprocessing {model_name} state:", flush=True)
            restart_dir = self.get_restart_dir(c, model_name)
            opts = self.get_task_opts(c, path=path, restart_dir=restart_dir)
            self.run_ensemble_tasks_in_scheduler(c, f'postproc_{model_name}', model.postprocess, opts, model.nproc_per_util)

    def perturb(self, c):
        """
        Perturbation step.

        This step adds random perturbations to the model initial and/or boundary conditions,
        at the first or all the analysis cycles.

        The ``perturb`` section in configuration file defines the scheme and parameters for the perturbation.
        The `utils.random_perturb`` module implements the random field generator functions.
        """
        pass

    def diagnose(self, c):
        """
        Diagnostics step.

        This step runs diagnostics for the current analysis cycle.

        The ``diag`` section in configuration file defines the methods and parameters of the diagnostics,
        corresponding to the ``diag.method`` module that implements the particular diagnostic method.
        """
        pass

    def run_step(self, c, step, mpi):
        """
        Helper function to run this script (``offline_filter.py``) from an external call.

        Args:
            c (Config): Configuration.
            step (str): Step to run.
            mpi (bool): Whether to run the step within mpi environment.

        This is useful when each step uses different strategies in parallelization.
        For the ``filter`` , ``diagnose`` and ``perturb`` steps, an mpirun call starts the program with mpi4py support,
        while the ``ensemble_forecast``, ``preprocess`` and ``postprocess`` steps start as a serial call to python,
        and utilize a custom scheduler implemented in ``utils.parallel.Scheduler`` to spawn the tasks to multiple processors.
        """
        script_file = os.path.abspath(__file__)

        # create a temporary config yaml file to hold c, and pass into program through runtime arg
        with tempfile.NamedTemporaryFile(dir=c.work_dir,
                                         prefix='config-',
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
                    commands += f"JOB_EXECUTE {sys.executable} {script_file} -c {tmp_config_file.name}"
                else:
                    print("Warning: mpi4py is not found, will try to run with nproc=1.", flush=True)
                    commands += f"{sys.executable} {script_file} -c {tmp_config_file.name} --nproc=1"
            else:
                commands += f"{sys.executable} {script_file} -c {tmp_config_file.name}"
            commands += f" --step {step}"

            if mpi:
                job_opts = {
                    'job_name': step,
                    'run_dir': c.cycle_dir(c.time),
                    'nproc': c.nproc,
                    'debug': c.debug,
                    **(c.job_submit or {}),
                    }
                run_job(commands, **job_opts)
            else:
                p = subprocess.Popen(commands, shell=True, text=True)
                p.wait()
                if p.returncode != 0:
                    print(f"OfflineFilter: run_step '{step}' exited with error")
                    sys.exit(1)

    def get_task_opts(self, c, **other_opts):
        """
        
        """
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
        model = c.models[model_name]
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
            nworker = min(c.nens, c.nproc_util)
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


    def init_analysis_dir(self, c):
        """
        Initialize the analysis directory.
        """
        self._analysis_dir = c.analysis_dir(c.time, c.iter)
        if c.pid == 0:
            makedir(self._analysis_dir)
            print(f"\nRunning assimilation step in {self._analysis_dir}\n", flush=True)

if __name__ == '__main__':
    # get config from runtime args, including the step to run (from --step)
    from NEDAS.config import Config
    c = Config(parse_args=True)

    if not c.step:
        raise RuntimeError("no step specified for offline filter scheme")

    scheme = OfflineFilterAnalysisScheme()
    step = getattr(scheme, c.step)
    timer(c)(step)(c)

    c.comm.finalize()
