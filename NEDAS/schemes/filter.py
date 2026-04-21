
from typing import Any
from datetime import datetime
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
        'prepare_truth': False,
        'prepare_init_ensemble': False,
        'preprocess': False,
        'perturb': True,
        'filter': True,
        'postprocess': False,
        'ensemble_forecast': False,
        'diagnose': True,
    }

    def run_all(self) -> None:
        self.c.log_event("PREPARING...")
        self.run_step('prepare_truth')
        if self.c.time == self.config.time_start:
            self.run_step('prepare_init_ensemble')
        else:
            self.load_model_state(self.c.prev_time)

        self.c.log_event("CYCLING START...")
        self.c.time = self.c.config.time
        while self.c.time < self.config.time_end:
            self.c.log_event(f"CURRENT CYCLE: {self.c.time} -> {self.c.next_time}")

            if self.check_cycle_complete():
                continue

            if self.config.run_preproc:
                self.run_step('preprocess')
                if self.config.perturb:
                    self.run_step('perturb')

            # assimilation step
            if self.config.run_analysis and self.c.time >= self.config.time_analysis_start and self.c.time <= self.config.time_analysis_end:
                self.run_step('filter')
                if self.config.run_postproc:
                    self.run_step('postprocess')

            # advance model state to next analysis cycle
            if self.config.run_forecast:
                self.run_step('ensemble_forecast')

            # compute diagnostics
            if self.config.run_diagnose:
                if self.config.diag:
                    self.run_step('diagnose')

            # dump data in memory to checkpoint files
            if self.c.time > self.c.config.time_start:
                self.save_model_state(self.c.prev_time)
                self.save_obs(self.c.prev_time)

            # advance to next cycle
            self.c.time = self.c.next_time

        self.c.log_event("CYCLING COMPLETE.", flag='finish')

    def load_model_state(self, time: datetime):
        for _, model in self.c.models.items():
            for tag in ['current', 'z']:
                model.load_memory(tag, time)

    def save_model_state(self, time: datetime):
        for _, model in self.c.models.items():
            for tag in ['current', 'prior', 'prior_mean', 'post', 'post_mean', 'z', 'z_mean']:
                model.save_memory(tag, time)

    def load_obs(self, time:datetime):
        for _, dataset in self.c.datasets.items():
            for tag in ['raw']:
                dataset.load_memory(tag, time)

    def save_obs(self, time:datetime):
        for _, dataset in self.c.datasets.items():
            for tag in ['raw', 'prior', 'post']:
                dataset.save_memory(tag, time)

    def prepare_truth(self) -> None:
        """
        Generate the verifying truth
        """
        # skip if not using synthetic obs (no need for the truth)
        if not self.use_synthetic_obs:
            self.c.message = "not using synthetic obs, skipping..."
            return

        for model_name, model in self.c.models.items():
            # if truth data is ready (either loadeable from memory save files, or truth files exists)
            if self.validate_truth_data(model_name):
                self.c.message = "all truth data ready"
                continue
            # generate the truth data
            opts = self.get_task_opts('prepare_truth', model_name, member=None)
            self.c.logger(f'Generate {model_name} truth')(self.c.io.call_method)(self.c, 'truth', model.generate_truth, **opts)
            model.save_memory('truth')

    def validate_truth_data(self, model_name: str) -> bool:
        model = self.c.models[model_name]
        model.load_memory('truth')
        name = list(model.variables.keys())[0]
        self.c.time = self.c.config.time_start
        while self.c.time < self.c.config.time_end:
            try:
                self.c.io.call_method(self.c, 'truth', model.read_var, name=name, member=None, time=self.c.time, model_src=model_name)
            except Exception:
                return False
            self.c.time = self.c.next_time
        self.c.time = self.c.config.time_start
        return True

    def prepare_init_ensemble(self) -> None:
        """
        Generate initial ensemble.
        """
        for model_name, model in self.c.models.items():
            if self.validate_init_ensemble_data(model_name):
                self.c.message = "all initial ensemble states ready"
                continue
            opts = self.get_task_opts('prepare_init_ensemble', model_name, nproc=model.nproc_per_run)
            self.c.logger(f'Generate {model_name} init ensemble')(self.run_ensemble_tasks)(model.ens_run_strategy, 'current', f'init_ens_{model_name}', model.generate_init_ensemble, **opts)
            model.save_memory('current', self.c.config.time_start)

    def validate_init_ensemble_data(self, model_name):
        model = self.c.models[model_name]
        model.load_memory('current', self.c.config.time_start)
        name = list(model.variables.keys())[0]
        for member in self.c.mem_list[self.c.pid_mem]:
            try:
                self.c.io.call_method(self.c, 'current', model.read_var, name=name, member=member, time=self.c.config.time_start, model_src=model_name)
            except Exception:
                return False
        return True

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
            opts = self.get_task_opts('preprocess', model_name, restart_dir=self.get_restart_dir(model_name), nproc=model.nproc_per_util)
            self.c.logger(f'Preprocess {model_name}')(self.run_ensemble_tasks)('scheduler', 'current', f'preproc_{model_name}', model.preprocess, **opts)

    def postprocess(self) -> None:
        """
        Post-processing step after the assimilation and before the next forecast.
        """
        for model_name, model in self.c.models.items():
            opts = self.get_task_opts('postprocess', model_name, restart_dir=self.get_restart_dir(model_name), nproc=model.nproc_per_util)
            self.c.logger(f'Postprocess {model_name}')(self.run_ensemble_tasks)('scheduler', 'current', f'postproc_{model_name}', model.postprocess, **opts)

    def ensemble_forecast(self) -> None:
        """
        Ensemble forecast step.
        """
        for model_name, model in self.c.models.items():
            opts = self.get_task_opts('ensemble_forecast', model_name, restart_dir=self.get_restart_dir(model_name), nproc=model.nproc_per_run, walltime=model.walltime)
            self.c.logger(f'Run {model_name} forecast')(self.run_ensemble_tasks)(model.ens_run_strategy, 'current', f'forecast_{model_name}', model.run, **opts)

    def filter(self) -> None:
        """
        Main method for performing the analysis step
        """
        # outer loop (iter = 0, ..., niter-1)
        # multiscale approach: loop over scale components and perform assimilation on each scale
        # more complex outer loops can be implemented here
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

    def get_task_opts(self, step: str, model_name:str, **other_opts) -> dict[str, Any]:
        """
        Get common kwargs from configuration and return as task options dict
        """
        opts = {
            'model_src': model_name,
            'time': self.c.time,
            'forecast_period': self.config.cycle_period,
            'total_nproc': self.config.nproc if self.steps_need_mpi[step] else self.config.nproc_util,
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
        self.c.debug_message = f"using restart files in {restart_dir}"
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
