import os
import numpy as np
from NEDAS.grid import RegularGrid
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.core import Model
from NEDAS.core.types import VarDesc
from .util import initial_condition, advance_time

class Vort2DModel(Model[RegularGrid]):
    nx: int
    ny: int
    dx: float
    dt: float
    restart_dt: float
    Vmax: float
    Rmw: float
    Vbg: float
    Vslope: float
    loc_sprd: int
    gen: float
    diss: float
    memory: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # define the model grid
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x = ii*self.dx
        y = jj*self.dx
        self.grid = RegularGrid(None, x, y, cyclic_dim='xy')
        self.grid.mask = np.full(self.grid.x.shape, False)  # no mask

        levels = np.array([0])  # there is no vertical levels
        self.variables = {
            'velocity': VarDesc(name=('u', 'v'), dtype='float', is_vector=True, dt=self.restart_dt, levels=levels, units='m/s', z_units='m'),
        }

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        mstr = self.get_mstr(kwargs['member'])
        tstr = self.get_tstr(kwargs['time'])
        return os.path.join(kwargs['path'], tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var_from_file(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)

        rec = self.variables[kwargs['name']].asdict()
        comm = None # reading files doesn't require collective io (file locks).
        if rec['is_vector']:
            u = nc_read_var(fname, rec['name'][0], comm=comm)[0, ...]
            v = nc_read_var(fname, rec['name'][1], comm=comm)[0, ...]
            var = np.array([u, v])
        else:
            var = nc_read_var(fname, rec['name'], comm=comm)[0, ...]
        return var

    def write_var_to_file(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)

        rec = self.variables[kwargs['name']].asdict()
        comm = self.c.comm   # for async file io (netcdf without parallel support)
        if rec['is_vector']:
            for i in range(2):
                nc_write_var(fname, {'t':None, 'y':self.ny, 'x':self.nx}, rec['name'][i], var[i,...], recno={'t':0}, comm=comm)
        else:
            nc_write_var(fname, {'t':None, 'y':self.ny, 'x':self.nx}, rec['name'], var, recno={'t':0}, comm=comm)

    def z_coords(self, **kwargs):
        return np.zeros(self.grid.x.shape)

    def generate_initial_condition(self):
        state = initial_condition(self.grid, self.Vmax, self.Rmw, self.Vbg, self.Vslope, self.loc_sprd)
        return state

    def preprocess(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if self.io_mode == 'offline':
            self.c.fs.make_dir(kwargs['path'])
            file1 = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
            file2 = self.filename(**kwargs)
            self.c.run_job(f"cp -fL {file1} {file2}")
        elif self.io_mode == 'online':
            # save a copy of the current state (forecast) as prior
            var = self.read_var_from_memory(**kwargs)
            self.write_var_to_memory(var.copy(), **{**kwargs, 'tag':'prior'})
 
    def postprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        # if offline mode, the current files are just posterior states
        # do nothing
        if self.io_mode == 'online':
            # save a copy of the current state (analysis) as posterior
            # don't need to copy since current state is no longer updated
            var = self.read_var_from_memory(**kwargs)
            self.write_var_to_memory(var, **{**kwargs, 'tag':'post'})

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.run_status = 'running'

        state = self.read_var(**kwargs)
        forecast_period = kwargs['forecast_period']
        time = kwargs['time']
        next_time = kwargs['time'] + forecast_period * dt1h
        while time < next_time:
            time += self.restart_dt * dt1h
            new_state = advance_time(state, self.dx, self.restart_dt, self.dt, self.gen, self.diss)
            self.write_var(new_state, **{**kwargs, 'time':time})
        self.run_status = 'complete'

    def generate_truth(self, *args, **kwargs) -> None:
        assert self.truth_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        debug = kwargs.get('debug', False)
        self.c.fs.make_dir(self.truth_dir)

        self.c.total_tasks = int((self.c.config.time_end - self.c.config.time_start) / (dt1h*self.c.config.cycle_period))

        t = self.c.config.time_start
        self.c.current_task = 0
        while t < self.c.config.time_end:
            opts = {
                **kwargs,
                'path': self.truth_dir,
                'name': 'velocity',
                'is_vector': True,
                'time': t,
                }

            if t == self.c.config.time_start:
                state = self.generate_initial_condition()
                if debug:
                    print(f"generating initial condition {self.filename(**opts)}")
                self.write_var(state, **opts)

            next_t = t + kwargs['forecast_period'] * dt1h
            self.c.debug_message = f"running model, saving output {self.filename(**{**opts, 'time':next_t})}"
            self.run(**{**kwargs, 'path':self.truth_dir, 'time':t})
            t = next_t
            self.c.current_task += 1

    def generate_init_ensemble(self, *args, **kwargs) -> None:
        assert self.ens_init_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        debug = kwargs.get('debug', False)
        self.c.fs.make_dir(self.ens_init_dir)

        opts = {
            **kwargs,
            'path': self.ens_init_dir,
            'name': 'velocity',
            'is_vector': True,
            }
        if debug:
            print(f"generating initial condition for member {kwargs['member']+1}, output to {self.filename(**opts)}")

        state = self.generate_initial_condition()
        self.write_var(state, **opts)
