import os
import numpy as np
from NEDAS.grid import Grid
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.shell_utils import run_command, makedir
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.models import Model
from .util import initial_condition, advance_time

class Vort2DModel(Model):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        ##define the model grid
        ii, jj = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        x = ii*self.dx
        y = jj*self.dx
        self.grid = Grid(None, x, y, cyclic_dim='xy')
        self.grid.mask = np.full(self.grid.x.shape, False)  ##no mask

        levels = np.array([0])  ##there is no vertical levels

        self.variables = {'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':self.restart_dt, 'levels':levels, 'units':'m/s'}, }

        self.z_units = '*'

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert kwargs['time'] is not None, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(kwargs['path'], tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)

        rec = self.variables[kwargs['name']]
        comm = kwargs['comm']
        if rec['is_vector']:
            u = nc_read_var(fname, rec['name'][0], comm=comm)[0, ...]
            v = nc_read_var(fname, rec['name'][1], comm=comm)[0, ...]
            var = np.array([u, v])
        else:
            var = nc_read_var(fname, rec['name'], comm=comm)[0, ...]
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)

        rec = self.variables[kwargs['name']]
        comm = kwargs['comm']
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

    def preprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        makedir(kwargs['path'])

        file1 = self.filename(**{**kwargs, 'path':kwargs['restart_dir']})
        file2 = self.filename(**kwargs)
        run_command(f"cp -fL {file1} {file2}")

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        makedir(kwargs['path'])

        state = self.read_var(**kwargs)
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        next_state = advance_time(state, self.dx, forecast_period, self.dt, self.gen, self.diss)
        self.write_var(next_state, **{**kwargs, 'time':next_time})
        self.run_status = 'complete'
