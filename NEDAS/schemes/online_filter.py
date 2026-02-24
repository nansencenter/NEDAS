from datetime import datetime

from NEDAS.config import Config
import numpy as np
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.progress import timer, progress_bar
from NEDAS.utils.shell_utils import makedir, run_command, run_job
from NEDAS.utils.parallel import Scheduler, bcast_by_root, distribute_tasks
from NEDAS.utils.random_perturb import random_perturb
from NEDAS import assim_tools
from .filter import FilterAnalysisScheme

class OnlineFilterAnalysisScheme(FilterAnalysisScheme):
    """
    Online filtering analysis scheme class.

    Keeping the ensemble model states in memory and the entire process run in a single program.
    """
    io_mode = 'online'

    def check_cycle_complete(self, c: Config, time: datetime) -> bool:
        """
        For online scheme, always restart from the beginning, so this always return False for now.
        A restart mechanism can be added if snapshot is saved to disk and reload from snapshot,
        then we can skip the past cycles here.
        """
        return True

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

    def ensemble_forecast(self, c: Config) -> None:
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

if __name__ == '__main__':
    # get config from runtime args, including the step to run (from --step)
    from NEDAS.config import Config
    c = Config(parse_args=True)
    scheme = OnlineFilterAnalysisScheme()
    scheme(c)
    c.comm.finalize()
