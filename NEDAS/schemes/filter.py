
from typing import Any
from datetime import datetime
from NEDAS.utils.progress import timer
from NEDAS.core import Scheme, Context, State, Obs, Perturbation, Diagnostics

class FilterAnalysisScheme(Scheme):
    """
    Scheme subclass for performing filter analysis.

    This scheme runs the 4D analysis by cycling through time steps. Running the ensemble forecast first,
    pause the model at a certain time step (`analysis cycle`), then perform data assimilation at the
    analysis cycle with observations within a time window, finally using the updated model states (the `posterior`)
    as new initial conditions, the ensemble forecast is run again to reach the next analysis cycle, until
    the end of the period of interest. The length of forecasts between cycles is called the `cycling period`.
    """
    steps = {
        'generate_truth': {'ensemble':False, 'mpi':False},
        'generate_init_ensemble': {'ensemble':True, 'mpi':False},
        'preprocess': {}
    }

    def run_all(self) -> None:
        self.c.print_1p("Initializing...\n")
        self.c.show_summary()

        self.run_step('generate_truth')

        self.run_step('generate_init_ensemble')

        print("\nCycling start.\n")
        while self.c.time < self.config.time_end:
            self.c.print_1p(f"\n\033[1;33mCURRENT CYCLE\033[0m: {self.c.time} => {self.c.next_time}\n")

            if self.check_cycle_complete():
                continue

            if self.config.run_preproc:
                self.run_step('preprocess')
                if self.config.perturb:
                    self.run_step('perturb')

            ##assimilation step
            if self.config.run_analysis and c.time >= self.config.time_analysis_start and c.time <= self.config.time_analysis_end:
                self.run_step('filter')
                if self.config.run_postproc:
                    self.run_step('postprocess')

            ##advance model state to next analysis cycle
            if self.config.run_forecast:
                self.run_step('ensemble_forecast')

            ##compute diagnostics
            if self.config.run_diagnose:
                if self.config.diag:
                    self.run_step('diagnose')

            ##advance to next cycle
            c.time = c.next_time

        c.print_1p("Cycling complete.\n")

    def generate_truth(self, c: Context) -> None:
        # skip if not using synthetic obs (no need for the truth)
        if not c.use_synthetic_obs:
            return
        for model_name, model in c.models.items():
            c.print_1p(f"Generating truth for '{model_name}' model...\n")
            opts = self.get_task_opts(c)
            opts['model_src'] = model_name
            opts['member'] = None
            c.io.call_method(c, 'truth', model.generate_truth, **opts)

    def generate_init_ensemble(self, c: Context):
        for model_name, model in c.models.items():
            c.print_1p(f"Generating initial ensemble for '{model_name}' model...\n")
            opts = self.get_task_opts(c)
            for mem_id in c.mem_list[c.pid_mem]:
                opts['model_src'] = model_name
                opts['member'] = mem_id
                c.io.call_method(c, 'prior', model.generate_init_ensemble, **opts)

    def check_cycle_complete(self) -> bool:
        """
        Method checks on a given cycle to see if it's complete or not
        """
        return False

    def preprocess(self, c: Context) -> None:
        """
        Pre-processing step before the forecast.
        """


    def postprocess(self, c: Context) -> None:
        """
        Post-processing step after the assimilation and before the next forecast.
        """


    def ensemble_forecast(self, c: Context) -> None:
        """
        Ensemble forecast step.
        """
        for model_name, model in c.models.items():
            #path = c.forecast_dir(c.time, model_name)
            #makedir(path)
            c.print_1p(f"Running {model_name} ensemble forecast:\n")

            opts = self.get_task_opts(c)
            if model.ens_run_type == 'mpi':
                for mem_id in c.mem_list[c.pid_mem]:
                    opts['member'] = mem_id
                    opts['model_src'] = model_name
                    c.io.call_method(c, 'prior', model.run, **opts)

            elif model.ens_run_type == 'batch':
                opts['nens'] = c.config.nens
                c.io.call_method(c, 'prior', model.run_batch, **opts)

            elif model.ens_run_type == 'scheduler':
                walltime = getattr(model, 'walltime', None)
                self.run_ensemble_tasks_in_scheduler(c, f'forecast_{model_name}', model.run, opts, model.nproc_per_run, walltime)

            else:
                raise ValueError(f"Unknown ensemble run type {model.ens_run_type} for {model_name}")

    def filter(self, c: Context) -> None:
        """
        Main method for performing the analysis step
        """
        # outer loop (iter = 0, ..., niter-1)
        ##multiscale approach: loop over scale components and perform assimilation on each scale
        ##more complex outer loops can be implemented here
        for c.iter in range(c.config.niter):
            c.print_1p(f"Running analysis for outer iteration step {c.iter}:")

            c.update_assim_tools()

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
        """
        c.print_1p(f"Perturbing state:")

        pert = Perturbation(c)
        timer(c)(pert)(c)

    def diagnose(self, c: Context):
        """
        Diagnostics step.

        This step runs diagnostics for the current analysis cycle.

        The ``diag`` section in configuration file defines the methods and parameters of the diagnostics,
        corresponding to the ``diag.method`` module that implements the particular diagnostic method.
        """
        c.print_1p(f"Running diagnostics:")

        diag = Diagnostics(c)
        timer(c)(diag)(c)

    def get_task_opts(self, c: Context, **other_opts) -> dict[str, Any]:
        """
        Get common kwargs from configuration and return as task options dict
        """
        opts = {
            'time': c.time,
            'forecast_period': c.config.cycle_period,
            'time_start': c.config.time_start,
            'time_end': c.config.time_end,
            'debug': c.config.debug,
            **(c.config.job_submit or {}),
            **other_opts,
            }
        return opts


def main():
    # initialize scheme
    scheme = FilterAnalysisScheme(parse_args=True)

    step = scheme.config.step
    if step:
        scheme.run_step(step)
        return

    scheme()

if __name__ == '__main__':
    main()
