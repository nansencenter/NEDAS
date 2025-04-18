import os
import sys
import tempfile
from NEDAS.utils.progress import timer
from NEDAS.utils.shell_utils import makedir
from NEDAS.schemes.base import AnalysisScheme

class OfflineFilterAnalysisScheme(AnalysisScheme):
    """
    Subclass for cycling scheme with filter and forecast steps
    """
    def run(self, c, step):
        """
        Run the python script from external call
        This is useful when several steps are using different strategy in parallelism
        For offline filter analysis, the filter step is run with mpi4py
        while the ensemble forecast uses a custom scheduler, other steps can be run in serial
        """

        # create a temporary config yaml file to hold c, and pass into program through runtime arg
        with tempfile.NamedTemporaryFile() as tmp_config_file:
            c.dump_yaml(tmp_config_file.name)

        print('running step'+step)

    def ensemble_forecast(self, c):
        for model_name, model in c.model_config.items():
            if model.ens_run_type == 'batch':
                timer(c)(self._ensemble_forecast_batch)(c, model_name)
            elif model.ens_run_type == 'scheduler':
                timer(c)(self._ensemble_forecast_scheduler)(c, model_name)
            else:
                raise NotImplementedError(f"Unknown ensemble forecast type: '{model.ens_run_type}' for '{model_name}'")


    def perturb(self, c):
        pass

    def preprocess(self, c):
        print("workflow of preprocess described here")

    def postprocess(self, c):
        pass

    def diagnose(self, c):
        pass    

    def filter(self, c):
        """
        Main method for performing the analysis step
        Input:
        - c: config object obtained at runtime
        """
        self.validate_mpi_environment(c)

        ##multiscale approach: loop over scale components and perform assimilation on each scale
        ##more complex outer loops can be implemented here
        analysis_grid = c.grid
        for c.scale_id in range(c.nscale):
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
        """
        Top-level workflow for filtering analysis (cycling through the analysis time steps)
        """
        c.show_summary()

        print("Cycling start...", flush=True)
        while c.time < c.time_end:
            print(f"\n\033[1;33mCURRENT CYCLE\033[0m: {c.time} => {c.next_time}", flush=True)

            os.system("mkdir -p "+c.cycle_dir(c.time))

            if c.run_prepare:
                timer(c)(self.run)(c, 'preprocess')
                timer(c)(self.perturb)(c)

            ##assimilation step
            if c.run_analysis and c.time >= c.time_analysis_start and c.time <= c.time_analysis_end:
                timer(c)(self.filter)(c)
                timer(c)(self.postprocess)(c)

            ##advance model state to next analysis cycle
            if c.run_forecast:
                timer(c)(self.ensemble_forecast)(c)

            ##compute diagnostics
            if c.run_diagnose:
                timer(c)(self.diagnose)(c)

            ##advance to next cycle
            c.time = c.next_time

        print("Cycling complete.", flush=True)


if __name__ == '__main__':
    from NEDAS.config import Config
    c = Config(parse_args=True)  # get config from runtime args

    scheme = OfflineFilterAnalysisScheme()

    step = getattr(scheme, c.step)
    step(c)
