import os
import numpy as np
from datetime import datetime
from typing import Callable, Any
from NEDAS.utils.conversion import ensure_list, dt1h
from NEDAS.utils import random_perturb, spatial_operation, parallel
from NEDAS.grid import GridType, RegularGrid
from .context import Context

class PerturbField:
    """
    Generates and applies perturbation on given 2D field(s)
    """
    grid: GridType
    mask: np.ndarray
    perturb_methods: dict[str, Callable]
    perturb_type: str
    other_opts: list[str] = []
    params: dict[str, dict[str, Any]]= {}

    def __init__(self, **kwargs) -> None:
        # get seed, if not specified get a random seed from system entropy
        seed = kwargs.get('seed', int.from_bytes(os.urandom(4), 'little'))
        assert isinstance(seed, int), f"seed {seed} invalid"
        # set the random seed
        np.random.seed(seed)

        self.grid = kwargs['grid']

        self.perturb_methods = {
            'gaussian': self.perturb_random_gaussian,
            'powerlaw': self.perturb_random_powerlaw,
            'displace': self.perturb_random_displace,
        }

        # parse kwargs and init the perturbation parameters
        self.parse_perturb_opts(**kwargs)

    def parse_perturb_opts(self, **kwargs) -> None:
        # perturb['type'] string format:
        #main option (gaussian/powerlaw/displace) followed by , then additional options separated by ,
        opts = kwargs['type'].split(',')
        self.perturb_type = opts[0]
        if self.perturb_type not in self.perturb_methods:
            raise NotImplementedError(f"Perturbation type: '{self.perturb_type}' is not implemented")

        self.other_opts = []
        for opt in opts[1:]:
            self.other_opts.append(opt)

        key_list = []
        for key in ['amp', 'hcorr', 'tcorr', 'powerlaw']:
            if key in kwargs:
                key_list.append(key)

        # a list of variables can be specified if running a multivariate perturbation scheme
        # rectify variable and parameter to be lists for further processing
        if not isinstance(kwargs['variable'], list):
            kwargs['variable'] = [kwargs['variable']]
            for key in key_list:
                kwargs[key] = [kwargs[key]]
        variable_list = kwargs['variable']
        nv = len(variable_list)  # number of variables

        # ensure again that parameters are rectified to lists
        for key in key_list:
            if not isinstance(kwargs[key], list):
                kwargs[key] = [kwargs[key]]

            # check for mismatch in list length
            if len(kwargs[key]) != nv:
                raise ValueError(f"perturb option: {key} has {len(kwargs[key])} entries, but {nv} variables are specified")

        # get perturbation parameters for each variable from kwargs
        self.params = {}
        for v in range(nv):
            vname = variable_list[v]
            self.params[vname] = {}
            # in multiscale approach, a list of parameters can be specified for a variable;
            # one separate perturbation will be generated for each, then they will be added together
            if isinstance(kwargs[key_list[0]][v], list):
                nscale = len(kwargs[key_list[0]][v])
            else:
                nscale = 1
                for key in key_list:  # make a list even if only one value for the key
                    kwargs[key][v] = [kwargs[key][v]]
            self.params[vname]['nscale'] = nscale
            # check if all keys are lists with same len
            for key in key_list[1:]:
                if len(kwargs[key][v]) != nscale:
                    raise ValueError(f"perturb option: {key} has different number of entries from {key_list[0]}, check config")
            # assign the parameters
            for key in key_list:
                self.params[vname][key] = kwargs[key][v]

    def generate_perturb(self, grid: GridType,
                        fields: dict[str, np.ndarray],
                        prev_perturb: dict[str, Any],
                        dt: float=1,
                        n: int=0,) -> dict[str, np.ndarray]:
        """
        Add random perturbation to the given 2D fields

        Args:
            grid (GridType): Grid object describing the 2d domain
            fields (dict[str, np.ndarray]): the input fields
            prev_perturb (dict[str, Any]): previous perturbation data, dict[str, None] if unavailable
            dt (float): interval (hours) between time steps
            n (int), current time step index

        Returns:
            dict[str, np.ndarray]: the generated perturbations
        """
        perturb = {}
        for vname,rec in self.params.items():
            fld = fields[vname]
            assert grid.x.shape == fld.shape[-2:], f"input fields[{vname}] dimension mismatch with grid"

            ns = rec['nscale']
            if self.perturb_type == 'displace':
                perturb[vname] = np.zeros((ns,2)+fld.shape[-2:])
            else:
                perturb[vname] = np.zeros((ns,)+fld.shape)

            if prev_perturb[vname] is not None and n==0:
                perturb[vname] = prev_perturb[vname]
                continue

            # loop over scale s and generate perturbation
            for s in range(ns):
                # draw a random field for each 2d field component in fields
                for ind in np.ndindex(fld.shape[:-2]):
                    perturb[vname][(s,)+ind] = self.perturb_methods[self.perturb_type](rec, s)

                # make perturb temporally correlated by blending with prev_perturb
                pp = prev_perturb[vname]
                if pp is not None:
                    perturb[vname][s] = self.make_correlated_perturb(pp[s], perturb[vname][s], rec['tcorr'][s] / dt)

        if 'press_wind_relate' in self.other_opts:
            perturb = self.make_wind_perturb_from_press(perturb)

        return perturb

    def add_perturb(self, fields: dict[str, np.ndarray], perturb: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        """ Add perturbations to each field """
        # mask_zero: skip perturbation at points where the *prior* field is already
        # at/below zero (e.g. open-ocean points for seaice_conc/seaice_thick).
        # Clipping negative draws back up to 0 at those points while leaving positive
        # draws untouched biases the ensemble mean upward; masking avoids that.
        variable_list = ensure_list(kwargs['variable'])
        mask_zero = kwargs.get('mask_zero', False)
        if not isinstance(mask_zero, list):
            mask_zero = [mask_zero] * len(variable_list)
        orig = {vname: fields[vname].copy()
                for vname, mz in zip(variable_list, mask_zero) if mz}

        for vname,rec in self.params.items():
            for s in range(rec['nscale']):
                if self.perturb_type == 'displace':
                    fields[vname] = spatial_operation.warp(self.grid, fields[vname], perturb[vname][s,0,...], perturb[vname][s,1,...])
                else:
                    if 'exp' in self.other_opts:
                        # add lognormal perturbations
                        fields[vname] *= np.exp(perturb[vname][s,...] - 0.5*rec['amp'][s]**4)
                    else:
                        # just add the gaussian perturbations
                        fields[vname] += perturb[vname][s,...]

            # respect value bounds after perturbing
            if 'bounds' in kwargs:
                vmin, vmax = kwargs['bounds']
                fields[vname] = np.minimum(np.maximum(fields[vname], vmin), vmax)

            # restore prior value where it was already at/below zero
            if vname in orig:
                fields[vname] = np.where(orig[vname] <= 0, orig[vname], fields[vname])
        return fields

    def perturb_random_gaussian(self, rec: dict[str, Any], s: int) -> np.ndarray:
        """ Generate a random perturbation using the Gaussian random field method """
        grid = self.grid
        assert isinstance(grid, RegularGrid), f"perturbation by random_field_gaussian only support RegularGrid, {grid}"
        p = random_perturb.random_field_gaussian(grid.nx, grid.ny, rec['amp'][s], rec['hcorr'][s]/grid.dx)
        return p

    def perturb_random_powerlaw(self, rec: dict[str, Any], s: int) -> np.ndarray:
        """ Generate a random perturbation using the powerlaw method """
        grid = self.grid
        assert isinstance(grid, RegularGrid), "perturbation by random_field_powerlaw only support RegularGrid"
        p = random_perturb.random_field_powerlaw(grid.nx, grid.ny, rec['amp'][s], rec['powerlaw'][s])
        return p

    def perturb_random_displace(self, rec: dict[str, Any], s: int) -> np.ndarray:
        """ Generate a random perturbation using the displacement method (returns a vector field) """
        du, dv = random_perturb.random_displacement(self.grid, self.grid.mask, rec['amp'][s], rec['hcorr'][s]/self.grid.dx)
        return np.array([du, dv])

    def make_correlated_perturb(self, prev_perturb: np.ndarray, perturb: np.ndarray, corr: float) -> np.ndarray:
        """ Create perturbations that are correlated in time """
        autocorr = 0.75
        alpha = autocorr**(1.0 / corr)
        return np.sqrt(1-alpha**2) * perturb + alpha * prev_perturb

    def make_wind_perturb_from_press(self, perturb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Legacy option in TOPAZ prsflg==1,2 options in force_perturb program, reproduced here.
        Expecting the vnames 'atmos_surf_press' for pressure field and 'atmos_surf_velocity' for the wind field.
        Derive the wind perturbation from pressure perturbations, so that they are in wind-pressure relation (prsflg==1)
        Additionally, derived wind perturbations are rescaled to match the specified amp (prsflg==2)
        """
        wind_to_press = random_perturb.get_velocity_from_press
        for vname in ['atmos_surf_velocity', 'atmos_surf_press']:
            assert vname in self.params.keys(), f'{vname} not in variable list, cannot run press_wind_relate option'

        for s in range(self.params['atmos_surf_press']['nscale']):
            pres_pert = perturb['atmos_surf_press'][s]
            scale_wind = ('scale_wind' in self.other_opts)
            pres_amp = self.params['atmos_surf_press']['amp'][s]
            pres_hcorr = self.params['atmos_surf_press']['hcorr'][s]
            wind_amp = self.params['atmos_surf_velocity']['amp'][s]
            wind_pert = wind_to_press(self.grid, pres_pert, scale_wind, pres_amp, pres_hcorr, wind_amp)
            perturb['atmos_surf_velocity'][s] = wind_pert
        return perturb

class Perturbation:
    """
    Perturbation top-level manager
    """
    nfld: int = 0
    task_list: dict[int, list[dict]] = {}
    perturb: dict[str, Any] = {}

    def __init__(self, c: Context):
        # distribute perturbation items among MPI ranks
        self.task_list = parallel.bcast_by_root(c.comm)(self.distribute_perturb_tasks)(c)

        # go through the opts to count how many fields will be perturbed (for showing progress)
        self.count_num_fields(c)

    def distribute_perturb_tasks(self, c: Context) -> dict[int, list[dict]]:
        task_list_full = []
        for perturb_rec in ensure_list(c.config.perturb):
            # init_only: True restricts this perturb record to the very first cycle (c.time ==
            # config.time_start), e.g. a one-time initial-condition kick for spinning up ensemble
            # spread from a near-identical start. Without this, the scheme's "perturb" step (run
            # every cycle in run_all()'s while loop, for ongoing BC/forcing-type perturbations)
            # would re-generate this record each cycle too -- and for an IC-type variable (dt >=
            # restart_dt), generate_perturb's reuse-from-prev_perturb logic means it would then
            # silently re-add the SAME frozen perturbation on top of the already-evolved state
            # every cycle, rather than applying it once at t=0.
            if perturb_rec.get('init_only', False) and c.time != c.config.time_start:
                continue
            for mem_id in range(c.nens):
                task_list_full.append({**perturb_rec, 'member':mem_id})
        task_list = parallel.distribute_tasks(c.comm, task_list_full)
        return task_list

    def count_num_fields(self, c: Context):
        for rec in self.task_list[c.pid]:
            model = c.models[rec['model_src']]
            variable_list = ensure_list(rec['variable'])
            dt = self.rec_dt(model, variable_list)
            nstep = self.nstep(c, model, dt)
            for _ in range(nstep):
                for _ in model.variables[variable_list[0]].levels:
                    self.nfld += 1

    def __call__(self, c: Context) -> None:
        # nothing to do (e.g. all perturb records are init_only and this isn't the initial cycle)
        active_pids = [p for p,lst in self.task_list.items() if len(lst)>0]
        if not active_pids:
            return

        if c.config.io_mode == 'offline':
            self.prepare_perturb_dir(c)

        c.pid_show = active_pids[0]

        self.init_file_locks(c)

        # go through the tasks
        c.total_tasks = self.nfld+1
        fld_id = 0
        error = None
        try:
            for rec in self.task_list[c.pid]:
                p = PerturbField(**rec, grid=c.grid)
                model = c.models[rec['model_src']]  # model class object
                member = rec['member']
                variable_list = ensure_list(rec['variable'])

                # check if previous perturb is available from past cycles
                self.load_perturb_data(c, **rec)

                # IC variables (dt >= forecast_period) perturbed once; BC/forcing variables loop
                dt = self.rec_dt(model, variable_list)
                nstep = self.nstep(c, model, dt)
                for n in range(nstep):
                    t = c.time + n * dt * dt1h

                    # TODO: perturbation for each k level is drawn independently, can be improved
                    # by introducing a vertical correlation length scale, or using EOF modes.
                    # Note: assuming all variables in the list have the same k levels
                    for k in model.variables[variable_list[0]].levels:
                        fld_id += 1
                        c.debug_message = f"perturbing mem{member+1:03} {variable_list} at {t} level {k}"
                        c.current_task = fld_id

                        fields = self.collect_fields(c, t, k, **rec)
                        self.perturb = p.generate_perturb(c.grid, fields, prev_perturb=self.perturb, dt=dt, n=n)
                        fields = p.add_perturb(fields, self.perturb, **rec)

                        self.output_perturbed_fields(c, fields, t, k, **rec)

                self.save_perturb_data(c, **rec)

        except Exception as e:
            error = e

        # all ranks must reach the barrier; re-raise after so a failed rank exits cleanly
        c.comm.Barrier()
        c.comm.cleanup_file_locks()

        if error is not None:
            raise error

    def init_file_locks(self, c: Context) -> None:
        """Initialize file locks for nc-backed perturb variables (e.g. iced, iceh).

        Mirrors the updator pattern: collect all output file paths across every rank,
        gather them globally so every rank knows about every file, then init a lock
        for each unique path before any parallel write begins.

        The file-collection loop is wrapped in try/except so that a rank that fails
        mid-loop still participates in the collective allgather and Barrier, preventing
        an MPI deadlock.  Any error is re-raised after the barrier so the outer
        try/except in __call__ can handle it uniformly.
        """
        files = []
        error = None
        try:
            for rec in self.task_list[c.pid]:
                model = c.models[rec['model_src']]
                variable_list = ensure_list(rec['variable'])
                dt = self.rec_dt(model, variable_list)
                nstep = self.nstep(c, model, dt)
                for n in range(nstep):
                    t = c.time + n * dt * dt1h
                    for vname in variable_list:
                        for k in model.variables[vname].levels:
                            get_filename = getattr(model, 'filename', None)
                            if get_filename is None:
                                continue
                            file = c.io.call_method(c, 'current', get_filename,
                                                    name=vname, time=t, k=k, **rec)
                            if file:
                                files.append(file)
        except Exception as e:
            error = e
        all_files = c.comm.allgather(files)
        unique_files = {f for sublist in all_files for f in sublist if f}
        for file in unique_files:
            c.comm.init_file_lock(file)
        c.comm.Barrier()
        if error is not None:
            raise error

    def rec_dt(self, model, variable_list: list) -> float:
        """Return the shared dt for all variables in a perturb record.

        Raises ValueError if variables have different dt, since one time loop
        cannot serve variables with different temporal resolutions.
        """
        dts = {model.variables[v].dt for v in variable_list}
        if len(dts) > 1:
            raise ValueError(
                f"Variables in a perturb record must all have the same dt, "
                f"got {dict(zip(variable_list, [model.variables[v].dt for v in variable_list]))}"
            )
        return dts.pop()

    def nstep(self, c: Context, model, dt: float) -> int:
        """Number of time steps to perturb for a variable with the given dt.

        A variable is BC/forcing if its dt is strictly less than the model restart_dt
        (sub-restart temporal resolution). It is perturbed at every dt step over the
        forecast period. IC/restart variables are perturbed once at t=0.

        Uses model.restart_dt as the threshold when available; falls back to
        cycle_period so models without restart_dt get the conservative nstep=1.
        """
        period = getattr(c.config, 'forecast_period', None) or c.config.cycle_period
        restart_dt = getattr(model, 'restart_dt', None) or period
        if dt < restart_dt:
            return int(period / dt) + 1
        return 1

    def prepare_perturb_dir(self, c):
        """ Prepare and clear the directory where perturbation data will be stored (offline mode) """
        assert c.config.io_mode == 'offline', f"prepare_perturb_dir only needed in offline io mode"
        # clean up perturb files in current cycle dir
        for rec in c.config.perturb:
            path = c.fs.forecast_dir(c.time, rec['model_src'])
            perturb_dir = os.path.join(path, 'perturb')
            if c.pid==0:
                c.run_job(f"rm -rf {perturb_dir}; mkdir -p {perturb_dir}")
        c.comm.Barrier()

    def save_perturb_data(self, c: Context, **rec):
        """ Save a copy of perturbation data, for use by the next analysis cycle """
        path = None
        if c.config.io_mode == 'offline':
            path = os.path.join(c.fs.forecast_dir(c.time, rec['model_src']), 'perturb')

        for vname in ensure_list(rec['variable']):
            data = self.perturb[vname]
            assert data is not None
            c.io.save_ndarray(c, f"{vname}_mem{rec['member']+1:03d}", data, path)

    def load_perturb_data(self, c: Context, **rec):
        """ Load the perturbation data """
        path = None
        if c.config.io_mode == 'offline':
            path = os.path.join(c.fs.forecast_dir(c.time, rec['model_src']), 'perturb')

        for vname in ensure_list(rec['variable']):
            data = c.io.load_ndarray(c, f"{vname}_mem{rec['member']+1:03d}", path)
            self.perturb[vname] = data

    def collect_fields(self, c: Context, t: datetime, k: int, **rec) -> dict[str, np.ndarray]:
        """ Collect all model fields to be perturbed """
        variable_list = ensure_list(rec['variable'])
        model = c.models[rec['model_src']]

        # set up grids
        vname =variable_list[0]  # note: all variables in the list shall have same dt and k levels
        c.io.call_method(c, 'current', model.read_grid, name=vname, time=t, k=k, **rec)
        model.grid.set_destination_grid(c.grid)
        c.grid.set_destination_grid(model.grid)

        # collect model variable fields
        fields = {}
        for vname in variable_list:
            # read variable from model state
            fld = c.io.call_method(c, 'current', model.read_var, name=vname, time=t, k=k, **rec)
            # convert to analysis grid
            fields[vname] = model.grid.convert(fld, is_vector=model.variables[vname].is_vector)
        return fields

    def output_perturbed_fields(self, c: Context, fields: dict[str, np.ndarray], t: datetime, k:int, **rec) -> None:
        variable_list = ensure_list(rec['variable'])
        model = c.models[rec['model_src']]

        if rec['type'].split(',')[0]=='displace' and hasattr(model, 'displace'):
            # use model internal method to apply displacement perturbations directly
            displace_method = getattr(model, 'displace')
            c.io.call_method(c, 'current', displace_method, self.perturb, time=t, k=k, **rec)
        else:
            # convert from analysis grid to model grid, and
            # write the perturbed variables back to model state files
            for vname in variable_list:
                fld = c.grid.convert(fields[vname], is_vector=model.variables[vname].is_vector)
                c.io.call_method(c, 'current', model.write_var, fld, name=vname, time=t, k=k, **rec)
