import importlib
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.progress import timer
from NEDAS.utils.parallel import Scheduler, bcast_by_root, distribute_tasks
from NEDAS.core import Scheme

class ForecastScheme(Scheme):
    """
    Forecast scheme class.

    This scheme runs only the ensemble forecasts for the start times defined by time_start time_end every cycle_period.
    The length of each forecast between cycles is forecast_period, which can be different from cycle_period.
    """
    def __call__(self) -> None:
        c = self.c
        c.show_summary()

        if c.config.step:
            ##if --step=STEP is specified at runtime, just run STEP and quit
            if c.config.step in ['perturb', 'diagnose']:
                mpi = True
            else:
                mpi = False
            self.run_step(c.config.step, mpi)
            return

        print("Cycling start...", flush=True)
        while c.time < c.config.time_end:
            print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

            # os.system("mkdir -p "+c.io.cycle_dir(c.time))

            if c.config.run_preproc:
                self.run_step('preprocess', mpi=False)
                if c.config.perturb:
                    self.run_step('perturb', mpi=True)

            ##advance model state to next analysis cycle
            if c.config.run_forecast:
                self.run_step('ensemble_forecast', mpi=False)

            ##compute diagnostics
            if c.config.run_diagnose:
                if c.config.diag:
                    self.run_step('diagnose', mpi=True)

            ##advance to next cycle
            c.time = c.next_time

        print("Cycling complete.", flush=True)

    def ensemble_forecast(self, c):
        """
        Ensemble forecast step.

        """
        for model_name, model in c.models.items():
            path = c.forecast_dir(c.time, model_name)
            c.io.makedir(path)
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
            c.io.makedir(path)
            print(f"Preprocessing {model_name} state:", flush=True)
            restart_dir = self.get_restart_dir(c, model_name)
            opts = self.get_task_opts(c, path=path, restart_dir=restart_dir)
            self.run_ensemble_tasks_in_scheduler(c, f'preproc_{model_name}', model.preprocess, opts, model.nproc_per_util)

    def perturb(self, c):
        """
        Perturbation step.

        This step adds random perturbations to the model initial and/or boundary conditions,
        at the first or all the analysis cycles.

        The ``perturb`` section in configuration file defines the scheme and parameters for the perturbation.
        The `utils.random_perturb`` module implements the random field generator functions.
        """

    def diagnose(self, c):
        """
        Diagnostics step.

        This step runs diagnostics for the current analysis cycle.

        The ``diag`` section in configuration file defines the methods and parameters of the diagnostics,
        corresponding to the ``diag.method`` module that implements the particular diagnostic method.
        """


    def get_task_opts(self, c, **other_opts):
        """
        
        """
        opts = {
            'time': c.time,
            'forecast_period': c.forecast_period,
            'time_start': c.time_start,
            'time_end': c.time_end,
            'debug': c.debug,
            **(c.job_submit or {}),
            **other_opts,
            }
        return opts

    def get_restart_dir(self, c, model_name):
        """
        TODO: here each forecast start time has restart files already prepared beforehand.
        Not using restart files from previous cycle
        """
        model = c.models[model_name]
        restart_dir = model.ens_init_dir.format(time=c.time)
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

    def distribute_perturb_tasks(self, c):
        task_list_full = []
        for perturb_rec in ensure_list(c.perturb):
            for mem_id in range(c.nens):
                task_list_full.append({**perturb_rec, 'member':mem_id})
        task_list = distribute_tasks(c.comm, task_list_full)
        return task_list

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


def main():
    # get config from runtime args, including the step to run (from --step) if specified
    from NEDAS.config import Config
    config = Config(parse_args=True)

    # initialize scheme
    scheme = ForecastScheme(config)

    # decide how to run based on runtime 
    if config.step:
        stepfunc = getattr(scheme, config.step)
        timer(scheme.c)(stepfunc)(scheme.c)

    else:
        if config.io_mode == 'offline':
            raise RuntimeError("no step specified for offline scheme")
        else:
            scheme()

    scheme.c.comm.finalize()

if __name__ == '__main__':
    main()
