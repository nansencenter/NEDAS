import os
import sys
import tempfile
import importlib.util
import subprocess
from datetime import datetime
from NEDAS.utils.progress import timer
from NEDAS.utils.parallel import Scheduler, bcast_by_root, distribute_tasks
from NEDAS.core import Scheme, Context, State, Obs, PerturbationScheme

class FilterAnalysisScheme(Scheme):
    """
    Scheme subclass for performing filter analysis.

    This scheme runs the 4D analysis by cycling through time steps. Running the ensemble forecast first,
    pause the model at a certain time step (`analysis cycle`), then perform data assimilation at the
    analysis cycle with observations within a time window, finally using the updated model states (the `posterior`)
    as new initial conditions, the ensemble forecast is run again to reach the next analysis cycle, until
    the end of the period of interest. The length of forecasts between cycles is called the `cycling period`.
    """

    def __call__(self) -> None:
        c = self.c
        c.show_summary()

        if c.config.step:
            ##if --step=STEP is specified at runtime, just run STEP and quit
            stepfunc = getattr(self, c.config.step)
            stepfunc(c)
            return

        print("Cycling start.", flush=True)
        while c.time < c.config.time_end:
            print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

            if self.check_cycle_complete(c, c.time):
                continue

            if c.config.run_preproc:
                self.preprocess(c)
                if c.config.perturb:
                    self.perturb(c)

            ##assimilation step
            if c.config.run_analysis and c.time >= c.config.time_analysis_start and c.time <= c.config.time_analysis_end:
                self.filter(c)
                if c.config.run_postproc:
                    self.postprocess(c)

            ##advance model state to next analysis cycle
            if c.config.run_forecast:
                self.ensemble_forecast(c)

            ##compute diagnostics
            if c.config.run_diagnose:
                if c.config.diag:
                    self.diagnose(c)

            ##advance to next cycle
            c.time = c.next_time

        print("Cycling complete.", flush=True)


    def check_cycle_complete(self, c: Context, time: datetime) -> bool:
        """
        Method checks on a given cycle to see if it's complete or not

        Args:
            c (Context): the runtime context
            time (datetime): datetime object for time of the cycle

        Returns:
            bool: True of this cycle has completed.
        """
        ...

    def preprocess(self, c: Context) -> None:
        """
        Pre-processing step before the forecast.
        """
        ...

    def postprocess(self, c: Context) -> None:
        """
        Post-processing step after the assimilation and before the next forecast.
        """
        ...

    def ensemble_forecast(self, c: Context) -> None:
        """
        Ensemble forecast step.
        """
        ...

    def filter(self, c: Context) -> None:
        """
        Main method for performing the analysis step
        """

        # outer loop (iter = 0, ..., niter-1)
        ##multiscale approach: loop over scale components and perform assimilation on each scale
        ##more complex outer loops can be implemented here
        for c.iter in range(c.config.niter):
            c.print_1p(f"Running analysis for outer iteration step {c.iter}:")

            # initialize
            c.init_scheme()

            # prepare the state variables
            c.state = State(c)
            timer(c)(c.state.prepare_state)(c)

            # prepare the observations
            c.obs = Obs(c)
            timer(c)(c.obs.prepare_obs)(c)
            timer(c)(c.obs.prepare_obs_from_state)(c, 'prior')

            # run assimilate algorithm
            timer(c)(c.assimilator.assimilate)(c)

            # update the state to get posteriors
            timer(c)(c.updator.update)(c)

    def perturb(self, c: Context):
        """
        Perturbation step.

        This step adds random perturbations to the model initial and/or boundary conditions,
        at the first or all the analysis cycles.

        The ``perturb`` section in configuration file defines the scheme and parameters for the perturbation.
        The `utils.random_perturb`` module implements the random field generator functions.
        """
        if c.config.perturb is None:
            c.print_1p(f"No perturbation defined in config, exiting.\n")
            return
        c.print_1p(f"Perturbing state:")

        perturb_scheme = PerturbationScheme(c)

    def diagnose(self, c: Context):
        """
        Diagnostics step.

        This step runs diagnostics for the current analysis cycle.

        The ``diag`` section in configuration file defines the methods and parameters of the diagnostics,
        corresponding to the ``diag.method`` module that implements the particular diagnostic method.
        """
        c.print_1p(f"Running diagnostics:")

        # ##get task list for each rank
        # task_list = bcast_by_root(c.comm)(self.distribute_diag_tasks)(c)

        # ##the processor with most work load will show progress messages
        # c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

        # ##init file locks for collective i/o
        # self.init_file_locks(c)

        # ntask = len(task_list[c.pid])
        # for task_id, rec in enumerate(task_list[c.pid]):
        #     if c.debug:
        #         print(f"PID {c.pid:4} running diagnostics '{rec['method']}'", flush=True)
        #     else:
        #         c.print_1p(progress_bar(task_id, ntask))

        #     method_name = f"NEDAS.diag.{rec['method']}"
        #     mod = importlib.import_module(method_name)

        #     ##perform the diag task
        #     mod.run(c, **rec)

        # c.comm.Barrier()
        # c.print_1p(' done.\n')
        # c.comm.cleanup_file_locks()


if __name__ == '__main__':
    # get config from runtime args, including the step to run (from --step)
    from NEDAS.config import Config
    config = Config(parse_args=True)
    if not config.step:
        raise RuntimeError("no step specified for offline filter scheme")

    scheme = FilterAnalysisScheme(config)
    step = getattr(scheme, config.step)
    timer(scheme.c)(step)(scheme.c)

    scheme.c.comm.finalize()
