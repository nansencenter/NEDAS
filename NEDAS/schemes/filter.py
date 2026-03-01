import os
import sys
import tempfile
import importlib.util
import subprocess
import numpy as np
from datetime import datetime
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils.progress import timer, progress_bar
from NEDAS.utils.shell_utils import makedir, run_command, run_job
from NEDAS.utils.parallel import Scheduler, bcast_by_root, distribute_tasks
from NEDAS.utils.random_perturb import random_perturb
from NEDAS import assim_tools
from NEDAS.core import Scheme, Context, State, Obs

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
            self.run_step(c, c.config.step)
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
        pass
        # if c.perturb is None:
        #     c.print_1p(f"No perturbation defined in config, exiting.\n")
        #     return
        # c.print_1p(f"Perturbing state:")

        # ##clean perturb files in current cycle dir
        # for rec in c.perturb:
        #     perturb_dir = os.path.join(c.forecast_dir(c.time, rec['model_src']), 'perturb')
        #     if c.pid==0:
        #         run_command(f"rm -rf {perturb_dir}; mkdir -p {perturb_dir}")
        # c.comm.Barrier()

        # ##distribute perturbation items among MPI ranks
        # task_list = bcast_by_root(c.comm)(self.distribute_perturb_tasks)(c)

        # c.pid_show = [p for p,lst in task_list.items() if len(lst)>0][0]

        # ##first go through the fields to count how many (for showing progress)
        # nfld = 0
        # for rec in task_list[c.pid]:
        #     model_name = rec['model_src']
        #     model = c.models[model_name]
        #     vname = ensure_list(rec['variable'])[0]
        #     dt = model.variables[vname]['dt']
        #     niter = c.cycle_period // dt + 1
        #     for n in range(niter):
        #         for k in model.variables[vname]['levels']:
        #             nfld += 1

        # ##actually go through the fields to perturb now
        # fld_id = 0
        # for rec in task_list[c.pid]:
        #     model_name = rec['model_src']
        #     model = c.models[model_name]  ##model class object
        #     mem_id = rec['member']
        #     mstr = f'_mem{mem_id+1:03d}'
        #     path = c.forecast_dir(c.time, model_name)
        #     variable_list = ensure_list(rec['variable'])

        #     ##check if previous perturb is available from past cycles
        #     perturb = {}
        #     for vname in variable_list:
        #         psfile = os.path.join(c.forecast_dir(c.prev_time, model_name), 'perturb', vname+mstr+'.npy')
        #         if os.path.exists(psfile):
        #             perturb[vname] = np.load(psfile)
        #         else:
        #             perturb[vname] = None

        #     # get number of time steps for this set of variables
        #     # perturbation will be generated for all time steps if variable is available
        #     dt = max([model.variables[v]['dt'] for v in variable_list])
        #     nstep = c.cycle_period // dt + 1
        #     for n in range(nstep):
        #         t = c.time + n * dt * dt1h

        #         # TODO: perturbation for each k level is drawn independently, can be improved
        #         # by introducing a vertical correlation length scale, or using EOF modes.
        #         # Note: assuming all variables in the list have the same k levels
        #         for k in model.variables[variable_list[0]]['levels']:
        #             fld_id += 1
        #             if c.debug:
        #                 print(f"PID {c.pid:4}: perturbing mem{mem_id+1:03} {variable_list} at {t} level {k}", flush=True)
        #             else:
        #                 c.print_1p(progress_bar(fld_id, nfld+1))

        #             vname =variable_list[0]  ##note: all variables in the list shall have same dt and k levels
        #             model.read_grid(path=path, name=vname, time=t, member=mem_id, k=k)
        #             model.grid.set_destination_grid(c.grid)
        #             c.grid.set_destination_grid(model.grid)

        #             # collect variable fields
        #             fields = {}
        #             for vname in variable_list:
        #                 ##read variable from model state
        #                 fld = model.read_var(path=path, name=vname, time=t, member=mem_id, k=k)
        #                 ##convert to analysis grid
        #                 fields[vname] = model.grid.convert(fld, is_vector=model.variables[vname]['is_vector'])

        #             ##generate perturbation on analysis grid
        #             fields_pert, perturb = random_perturb(c.grid, fields, prev_perturb=perturb, dt=dt, n=n, **rec)

        #             if rec['type'].split(',')[0]=='displace' and hasattr(model, 'displace'):
        #                 ##use model internal method to apply displacement perturbations directly
        #                 model.displace(perturb, path=path, time=t, member=mem_id, k=k)
        #             else:
        #                 ##convert from analysis grid to model grid, and
        #                 ##write the perturbed variables back to model state files
        #                 for vname in variable_list:
        #                     fld = c.grid.convert(fields_pert[vname], is_vector=model.variables[vname]['is_vector'])
        #                     model.write_var(fld, path=path, name=vname, time=t, member=mem_id, k=k)

        #     ##save a copy of perturbation at next_t, for use by next cycle
        #     for vname in variable_list:
        #         psfile = os.path.join(path, 'perturb', vname+mstr+'.npy')
        #         run_command(f"mkdir -p {os.path.dirname(psfile)}")
        #         np.save(psfile, perturb[vname])

        # c.comm.Barrier()
        # c.print_1p(' done.\n')

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
