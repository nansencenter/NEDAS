
from typing import Any
from NEDAS.core import Scheme, State, Obs, Perturbation, Diagnostics

class FilterAnalysisScheme(Scheme):
    """
    Scheme subclass for performing filter analysis.

    This scheme runs the 4D analysis by cycling through time steps. Running the ensemble forecast first,
    pause the model at a certain time step (`analysis cycle`), then perform data assimilation at the
    analysis cycle with observations within a time window, finally using the updated model states (the `posterior`)
    as new initial conditions, the ensemble forecast is run again to reach the next analysis cycle, until
    the end of the period of interest. The length of forecasts between cycles is called the `cycling period`.
    """
    steps_need_mpi = {
        'run_all': True,
        'generate_truth': False,
        'generate_init_ensemble': False,
        'preprocess': False,
        'perturb': True,
        'filter': True,
        'postprocess': False,
        'ensemble_forecast': False,
        'diagnose': True,
    }

    def run_all(self) -> None:
        self.c.print_1p("INITIALIZING...\n")
        self.c.show_summary()

        if self.c.time == self.config.time_start:
            msg = f"PREPARATION"
            self.c.print_1p(f"{msg}\n{'═'*len(msg)}\n")

            self.run_step('generate_truth')

            self.run_step('generate_init_ensemble')

        self.c.print_1p("CYCLING START...\n")
        while self.c.time < self.config.time_end:
            msg = f"CURRENT CYCLE: {self.c.time} ➜ {self.c.next_time}"
            self.c.print_1p(f"{msg}\n{'═'*len(msg)}\n")

            if self.check_cycle_complete():
                continue

            if self.config.run_preproc:
                self.run_step('preprocess')
                if self.config.perturb:
                    self.run_step('perturb')

            ##assimilation step
            if self.config.run_analysis and self.c.time >= self.config.time_analysis_start and self.c.time <= self.config.time_analysis_end:
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
            self.c.time = self.c.next_time

        self.c.print_1p("CYCLING COMPLETE.\n")

    def generate_truth(self) -> None:
        """
        Generate the verifying truth
        """
        # skip if not using synthetic obs (no need for the truth)
        if not self.use_synthetic_obs:
            return

        for model_name, model in self.c.models.items():
            opts = self.get_task_opts(model_name, member=None)

            self.c.logger(f'Generate truth: {model_name}')(self.c.io.call_method)(self.c, 'truth', model.generate_truth, **opts)

    def generate_init_ensemble(self) -> None:
        """
        Generate initial ensemble.
        """
        #
        for model_name, model in self.c.models.items():
            opts = self.get_task_opts(model_name,
                                      total_nproc=self.config.nproc,
                                      nproc=model.nproc_per_run)

            self.c.logger(f'Generate init ensemble: {model_name}')(self.run_ensemble_tasks)(model.ens_run_strategy, 'prior', f'init_ens_{model_name}', model.generate_init_ensemble, **opts)

    def check_cycle_complete(self) -> bool:
        """
        Method checks on a given cycle to see if it's complete or not
        """
        return False

    def preprocess(self) -> None:
        """
        Pre-processing step before the assimilation.
        """
        for model_name, model in self.c.models.items():
            self.c.fs.make_dir(self.c.fs.forecast_dir(self.c.time, model_name))

            opts = self.get_task_opts(model_name,
                                      restart_dir=self.get_restart_dir(model_name),
                                      total_nproc=self.config.nproc_util,
                                      nproc=model.nproc_per_util)

            self.c.logger(f'Preprocess {model_name}')(self.run_ensemble_tasks)('scheduler', 'prior', f'preproc_{model_name}', model.preprocess, **opts)

    def postprocess(self) -> None:
        """
        Post-processing step after the assimilation and before the next forecast.
        """
        for model_name, model in self.c.models.items():
            opts = self.get_task_opts(model_name,
                                      restart_dir=self.get_restart_dir(model_name),
                                      total_nproc=self.config.nproc_util,
                                      nproc=model.nproc_per_util)

            self.c.logger(f'Postprocess {model_name}')(self.run_ensemble_tasks)('scheduler', 'prior', f'postproc_{model_name}', model.postprocess, **opts)

    def ensemble_forecast(self) -> None:
        """
        Ensemble forecast step.
        """
        for model_name, model in self.c.models.items():
            opts = self.get_task_opts(model_name,
                                      restart_dir=self.get_restart_dir(model_name),
                                      total_nproc=self.config.nproc,
                                      nproc=model.nproc_per_run,
                                      walltime=model.walltime)

            self.c.logger(f'Ensemble forecast: {model_name}')(self.run_ensemble_tasks)(model.ens_run_strategy, 'prior', f'forecast_{model_name}', model.run, **opts)

    def filter(self) -> None:
        """
        Main method for performing the analysis step
        """
        # outer loop (iter = 0, ..., niter-1)
        ##multiscale approach: loop over scale components and perform assimilation on each scale
        ##more complex outer loops can be implemented here
        for self.c.iter in range(self.config.niter):
            if self.config.niter > 1:
                self.c.logger(f"Outer-loop iteration #{self.c.iter+1}")(self.filter_iter)()
            else:
                self.filter_iter()

    def filter_iter(self) -> None:
        self.c.update_assim_tools()
        self.c.fs.make_dir(self.c.fs.analysis_dir(self.c.time, self.c.iter))

        self.c.state = State(self.c)
        self.c.logger('Prepare prior state')(self.c.state.prepare_state)(self.c)

        # prepare the observations
        self.c.obs = Obs(self.c)
        self.c.logger('Prepare obs')(self.c.obs.prepare_obs)(self.c)
        self.c.logger('Prepare obs from prior state')(self.c.obs.prepare_obs_from_state)(self.c, 'prior')

        # run assimilate algorithm
        self.c.logger('Assimilator')(self.c.assimilator.assimilate)(self.c)

        # update the state to get posteriors
        self.c.logger('Updator')(self.c.updator.update)(self.c)

    def perturb(self) -> None:
        """
        Perturbation step.

        This step adds random perturbations to the model initial and/or boundary conditions,
        at the first or all the analysis cycles.
        """
        if self.c.config.perturb is None:
            self.c.print_1p("Config: 'perturb' is not defined. Exiting...\n")
            return

        pert = Perturbation(self.c)
        pert(self.c)

    def diagnose(self) -> None:
        """
        Diagnostics step.

        This step runs diagnostics for the current analysis cycle.

        The ``diag`` section in configuration file defines the methods and parameters of the diagnostics,
        corresponding to the ``diag.method`` module that implements the particular diagnostic method.
        """
        if self.c.config.diag is None:
            self.c.print_1p("Config: 'diag' is not defined. Exiting...\n")
            return

        diag = Diagnostics(self.c)
        diag(self.c)

    def get_task_opts(self, model_name:str, **other_opts) -> dict[str, Any]:
        """
        Get common kwargs from configuration and return as task options dict
        """
        opts = {
            'model_src': model_name,
            'time': self.c.time,
            'forecast_period': self.config.cycle_period,
            **(self.config.job_submit or {}),
            **other_opts,
        }
        return opts
    
    def get_restart_dir(self, model_name) -> str:
        model = self.c.models[model_name]
        if self.c.time == self.config.time_start and model.ens_init_dir is not None:
            restart_dir = model.ens_init_dir.format(time=self.c.time)
        else:
            restart_dir = self.c.fs.forecast_dir(self.c.prev_time, model_name)
        if self.config.debug:
            print(f"using restart files in {restart_dir}", flush=True)
        return restart_dir

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
