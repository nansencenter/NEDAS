import os
from datetime import datetime, timedelta, timezone
import numpy as np
from pyproj import Proj
from NEDAS.grid import RegularGrid
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_open, nc_close, nc_read_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc
from .util import grid_coords, vel_to_temp, temp_to_vel, write_restart_file, input_nml

class BgridSoloModel(Model[RegularGrid]):
    """
    Dry dynamical core of the GFDL B-grid atmospheric model, with Held-Suarez forcing.

    This is the ``bgrid_solo`` model of DART: the FMS B-grid core, with the physics reduced to
    the Held and Suarez (1994) benchmark (Newtonian relaxation to a zonally symmetric
    equilibrium temperature and Rayleigh friction near the surface), on a global latitude-
    longitude grid. The default 60x30 grid with 5 levels is close to the smallest that has
    realistic baroclinic instability, with midlatitude storm tracks.

    The model is run through ``nedas_bgrid_advance`` (see ``build_bgrid_solo.sh``), which links
    the DART model source: it reads a state from a DART netCDF restart file, advances it by the
    requested time and writes it back. Only offline io mode is supported.

    The prognostic variables are surface pressure ``ps`` (Pa), temperature ``t`` (K) and
    horizontal wind ``u``, ``v`` (m/s) on 5 sigma levels (level 0 is the top). NEDAS carries them
    all on the temperature grid, the wind is on a staggered grid in the model (at the north-east
    corners of the temperature cells). Reading the wind averages the four surrounding wind
    points, writing it adds the changes from the (analysis of the) averaged field back to the
    wind points, so that a field left unchanged is written back exactly.

    Args:
        nlon, nlat, nlev (int): number of longitudes, latitudes and levels
        dt_atmos (int): model time step in seconds
        model_exe (str): path to the nedas_bgrid_advance executable
        spinup_hours (int): spin up period for the initial conditions
    """
    nlon: int
    nlat: int
    nlev: int
    dt_atmos: int
    time_ref: str
    model_exe: str
    spinup_hours: float
    init_perturb_sd: float
    init_ens_mode: str
    seed: int
    restart_dt: float
    memory: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.io_mode == 'offline', f"{self.__class__.__name__} only supports offline io mode"
        assert self.nproc_per_run == 1, f"{self.__class__.__name__} only supports serial runs"
        assert self.init_ens_mode in ('spinup', 'truth'), f"unknown init_ens_mode '{self.init_ens_mode}'"
        self.memory = {}  # not the class-level dict shared by all instances

        self._time_ref = datetime.fromisoformat(str(self.time_ref))
        if not self.model_exe:
            self.model_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nedas_bgrid_advance')

        # global lon-lat grid, periodic in longitude. The poles themselves are not grid points.
        # NEDAS wants the longitudes in -180 ~ 180, increasing along the row, whereas the model
        # (and its files) start at 0, so the columns of a field are reordered, see _to_grid
        lon, lat, _, _ = grid_coords(self.nlon, self.nlat)
        lon = (lon + 180.) % 360. - 180.
        self._lon_order = np.argsort(lon)
        x, y = np.meshgrid(lon[self._lon_order], lat)
        self.grid = RegularGrid(Proj('+proj=longlat'), x, y, cyclic_dim='x', distance_type='spherical')
        self.grid.mask = np.full(self.grid.x.shape, False)

        levels = np.arange(self.nlev, dtype=float)
        self.variables = {
            'wind': VarDesc(name=('u', 'v'), dtype='float', is_vector=True, dt=self.restart_dt,
                            levels=levels, units='m/s', z_units='hPa'),
            'temperature': VarDesc(name='t', dtype='float', is_vector=False, dt=self.restart_dt,
                                   levels=levels, units='K', z_units='hPa'),
            'ps': VarDesc(name='ps', dtype='float', is_vector=False, dt=self.restart_dt,
                          levels=np.array([0.]), units='Pa', z_units='hPa'),
        }

    # --- file layout and time -------------------------------------------------------------

    def filename(self, **kwargs) -> str:
        kwargs = super().parse_kwargs(kwargs)
        return os.path.join(kwargs['path'], self.get_tstr(kwargs['time']) + self.get_mstr(kwargs['member']) + '.nc')

    def model_day(self, time: datetime) -> float:
        """DART model time (days) of a date"""
        if time.tzinfo is not None:  # NEDAS dates are UTC, and aware if the config says so
            time = time.astimezone(timezone.utc).replace(tzinfo=None)
        day = (time - self._time_ref) / timedelta(days=1)
        assert day >= 0, f"{time} is before time_ref {self._time_ref}"
        return day

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def z_coords(self, **kwargs) -> np.ndarray:
        """nominal pressure (hPa) of the sigma levels, equally spaced from the top; ps is at the surface"""
        kwargs = super().parse_kwargs(kwargs)
        if kwargs['name'] == 'ps':
            p = 1000.
        else:
            p = (kwargs['k'] + 0.5) / self.nlev * 1000.
        return np.full(self.grid.x.shape, p)

    # --- reading and writing the state ------------------------------------------------------

    def _to_grid(self, a: np.ndarray) -> np.ndarray:
        """a field in the file's column order (longitude from 0) to the order of self.grid"""
        return a[..., self._lon_order]

    def _from_grid(self, a: np.ndarray) -> np.ndarray:
        """the inverse of _to_grid"""
        b = np.empty_like(a)
        b[..., self._lon_order] = a
        return b

    def _read(self, fname: str, varname: str, k: int|None) -> np.ndarray:
        index = (0, 0, slice(None), slice(None)) if k is None else (0, 0, k, slice(None), slice(None))
        return nc_read_var(fname, varname, index=index)

    def read_var_from_file(self, **kwargs) -> np.ndarray:
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        k = int(kwargs['k'])
        if name == 'ps':
            return self._to_grid(self._read(fname, 'ps', None))
        if name == 'temperature':
            return self._to_grid(self._read(fname, 't', k))
        if name == 'wind':
            return self._to_grid(np.array([vel_to_temp(self._read(fname, v, k)) for v in ('u', 'v')]))
        raise ValueError(f"unknown variable name '{name}'")

    def write_var_to_file(self, var, **kwargs) -> None:
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        k = int(kwargs['k'])
        var = self._from_grid(np.asarray(var))
        f = nc_open(fname, 'a', self.c.comm)
        try:
            if name == 'ps':
                f['ps'][0, 0, :, :] = var
            elif name == 'temperature':
                f['t'][0, 0, k, :, :] = var
            elif name == 'wind':
                for i, v in enumerate(('u', 'v')):
                    old = f[v][0, 0, k, :, :]
                    f[v][0, 0, k, :, :] = old + temp_to_vel(var[i] - vel_to_temp(old))
            else:
                raise ValueError(f"unknown variable name '{name}'")
        finally:
            nc_close(fname, f, self.c.comm)

    def _set_time(self, fname: str, time: datetime) -> None:
        f = nc_open(fname, 'a', self.c.comm)
        try:
            f['time'][0] = self.model_day(time)
        finally:
            nc_close(fname, f, self.c.comm)

    # --- running the model --------------------------------------------------------------------

    def preprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.c.fs.make_dir(kwargs['path'])
        file1 = self.filename(**{**kwargs, 'path': kwargs['restart_dir']})
        file2 = self.filename(**kwargs)
        self.c.run_job(f"cp -fL {file1} {file2}")

    def postprocess(self, *args, **kwargs):
        pass  # the files are the posterior states already

    def _advance(self, run_dir: str, ic_file: str, ud_file: str, job_name: str, task_id: int=0,
                 advance_seconds: int=0, cold_start: bool=False, init_days: int=0) -> None:
        """Run nedas_bgrid_advance in run_dir: ic_file (template, if cold_start) -> ud_file"""
        assert os.path.exists(self.model_exe), \
            f"{self.model_exe} not found, build it with NEDAS/models/bgrid_solo/build_bgrid_solo.sh"
        assert advance_seconds % self.dt_atmos == 0, \
            f"{advance_seconds} s is not a multiple of the model time step dt_atmos={self.dt_atmos} s"
        self.c.fs.make_dir(run_dir)
        for fname in (ud_file, os.path.join(run_dir, 'time_stamp.out')):
            if os.path.exists(fname):
                os.remove(fname)
        with open(os.path.join(run_dir, 'input.nml'), 'w') as f:
            # the model runs in run_dir, where the files are found by their names alone
            ic_name, ud_name = os.path.basename(ic_file), os.path.basename(ud_file)
            f.write(input_nml(self, ic_name, ic_name, ud_name, advance_seconds, cold_start, init_days))
        log_file = os.path.join(run_dir, 'run.log')
        self.c.run_job(f"cd {run_dir}; {self.model_exe} > run.log 2>&1", job_name=job_name,
                       offset=task_id * self.nproc_per_run)
        with open(log_file, 'rt') as f:
            finished = 'Finished ...' in f.read()
        if not finished or not os.path.exists(ud_file):
            raise RuntimeError(f"{self.__class__.__name__}: model run failed, see {log_file}")

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        task_id = kwargs.get('worker_id', 0)
        mstr = self.get_mstr(kwargs['member'])
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        self.run_status = 'running'

        input_file = self.filename(**kwargs)
        output_file = self.filename(**{**kwargs, 'time': next_time})
        run_dir = os.path.join(kwargs['path'], 'run' + mstr)
        ic_file = os.path.join(run_dir, 'temp_ic.nc')
        ud_file = os.path.join(run_dir, 'temp_ud.nc')

        # the model starts from the time stored in the file
        self.c.fs.make_dir(run_dir)
        self.c.fs.copy_file(input_file, ic_file)
        self._set_time(ic_file, time)

        self._advance(run_dir, ic_file, ud_file, f"bgrid_solo_run{mstr}", task_id,
                      advance_seconds=int(round(forecast_period * 3600)))

        self._set_time(ud_file, next_time)
        self.c.fs.move_file(ud_file, output_file)
        self.c.fs.remove_files(ic_file)
        self.run_status = 'complete'

    # --- initial conditions ---------------------------------------------------------------------

    def cold_start(self, fname: str, time: datetime, seed: int, **kwargs) -> None:
        """
        Write the initial state at `time` to `fname`: the model's cold start (at rest, uniform
        temperature) plus a random temperature perturbation to trigger baroclinic instability.
        """
        run_dir = os.path.join(os.path.dirname(fname), 'cold_start_' + os.path.basename(fname)[:-3])
        self.c.fs.make_dir(run_dir)
        template = os.path.join(run_dir, 'template.nc')
        ud_file = os.path.join(run_dir, 'temp_ud.nc')
        zero = lambda *shape: np.zeros(shape)
        write_restart_file(template, self.nlon, self.nlat, self.nlev, 0.,
                           zero(self.nlat, self.nlon), zero(self.nlev, self.nlat, self.nlon),
                           zero(self.nlev, self.nlat-1, self.nlon), zero(self.nlev, self.nlat-1, self.nlon))
        self._advance(run_dir, template, ud_file, 'bgrid_solo_cold_start', kwargs.get('worker_id', 0),
                      cold_start=True, init_days=int(self.model_day(time)))
        self.c.fs.move_file(ud_file, fname)
        self._set_time(fname, time)
        self.perturb_state(fname, seed, self.init_perturb_sd)
        self.c.fs.remove_dir(run_dir)

    def perturb_state(self, fname: str, seed: int, sd: float) -> None:
        """add random noise of standard deviation sd (K) to the temperature in a state file"""
        rng = np.random.default_rng(seed)
        f = nc_open(fname, 'a', self.c.comm)
        try:
            f['t'][0, 0, :, :, :] = f['t'][0, 0, :, :, :] + rng.normal(0, sd, (self.nlev, self.nlat, self.nlon))
        finally:
            nc_close(fname, f, self.c.comm)

    def generate_truth(self, *args, **kwargs) -> None:
        assert self.truth_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        self.c.fs.make_dir(self.truth_dir)
        opts = {**kwargs, 'path': self.truth_dir, 'member': None}
        time_start = self.c.config.time_start

        # spin up from the cold start to a statistically steady state at time_start
        if self.spinup_hours > 0:
            time = time_start - self.spinup_hours * dt1h
            self.c.debug_message = f"spinning up the truth run from {time} for {self.spinup_hours} hours"
            self.cold_start(self.filename(**{**opts, 'time': time}), time, self.seed)
            self.run(**{**opts, 'time': time, 'forecast_period': self.spinup_hours})
            os.remove(self.filename(**{**opts, 'time': time}))
        else:
            self.cold_start(self.filename(**{**opts, 'time': time_start}), time_start, self.seed)

        time = time_start
        while time < self.c.config.time_end:
            next_time = time + kwargs['forecast_period'] * dt1h
            self.c.debug_message = f"running model, saving output at {next_time}"
            self.run(**{**opts, 'time': time})
            time = next_time
        self.c.fs.remove_dir(os.path.join(self.truth_dir, 'run'))

    def generate_init_ensemble(self, *args, **kwargs) -> None:
        assert self.ens_init_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        member = kwargs['member'] if kwargs['member'] is not None else 0
        time = kwargs['time']
        opts = {**kwargs, 'path': self.ens_init_dir}
        fname = self.filename(**opts)
        if os.path.exists(fname):
            return
        self.c.fs.make_dir(self.ens_init_dir)

        if self.init_ens_mode == 'truth':
            assert self.truth_dir is not None, "init_ens_mode 'truth' needs a truth run (truth_dir)"
            self.c.fs.copy_file(self.filename(**{**kwargs, 'path': self.truth_dir, 'member': None}), fname)
            self.perturb_state(fname, self.seed + 1 + member, self.init_perturb_sd)
            return

        # independent cold starts, each spun up
        seed = self.seed + 1 + member
        if self.spinup_hours > 0:
            t0 = time - self.spinup_hours * dt1h
            self.cold_start(self.filename(**{**opts, 'time': t0}), t0, seed, worker_id=kwargs.get('worker_id', 0))
            self.run(**{**opts, 'time': t0, 'forecast_period': self.spinup_hours})
            os.remove(self.filename(**{**opts, 'time': t0}))
            self.c.fs.remove_dir(os.path.join(self.ens_init_dir, 'run' + self.get_mstr(kwargs['member'])))
        else:
            self.cold_start(fname, time, seed, worker_id=kwargs.get('worker_id', 0))
