import numpy as np
import os
import inspect
import signal
from functools import lru_cache

from utils.conversion import units_convert, t2s, s2t, dt1h
from config import parse_config

from ..abfile import ABFileRestart, ABFileArchv, ABFileBathy, ABFileGrid, ABFileForcing
from ..model_grid import get_topaz_grid, stagger, destagger

##some constants
ONEM = 9806.  ##pressure (Pa) in 1 meter water

class Model(object):
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.restart_dt = 168

        levels = np.arange(1, 51, 1)
        level_sfc = np.array([0])
        self.variables = {
                'ocean_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'levels':levels, 'units':'m/s'},
                'ocean_layer_thick': {'name':'dp', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'Pa'},
                'ocean_temp': {'name':'temp', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'K'},
                'ocean_saln': {'name':'saln', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'psu'},
                'ocean_surf_height': {'name':'ssh', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m'},
                'ocean_b_velocity':  {'name':('ubavg', 'vbavg'), 'dtype':'float', 'is_vector':True, 'levels':level_sfc, 'units':'m/s'},
                'ocean_b_press': {'name':'pbavg', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'Pa'},
                'ocean_mixl_depth': {'name':'dpmixl', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'Pa'},
                'seaice_velocity': {'name':('uice', 'vice'), 'dtype':'float', 'is_vector':True, 'levels':level_sfc, 'units':'m/s'},
                'seaice_conc': {'name':'ficem', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'%'},
                'seaice_thick': {'name':'hicem', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'m'},
                }

        self.z_units = 'm'

        self.run_process = None
        self.run_status = 'pending'


    def filename(self, **kwargs):
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'
        if 'time' in kwargs and kwargs['time'] is not None:
            assert isinstance(kwargs['time'], datetime), 'time shall be a datetime object'
            tstr = kwargs['time'].strftime('%Y_%j_%H')
        else:
            tstr = '????_???_??'
        if 'member' in kwargs and kwargs['member'] is not None:
            assert kwargs['member'] >= 0, 'member index shall be >= 0'
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        return path+'/'+'TP4restart'+tstr+mstr+'.a'


    def read_grid(self, **kwargs):
        rundir = os.path.dirname(self.filename(**kwargs))
        grid_info_file = os.path.join(rundir, 'topo', 'grid.info')
        self.grid = get_topaz_grid(grid_info_file)


    def write_grid(self, **kwargs):
        pass


    def read_mask(self):
        depthfile = path+'/topo/depth.a'
        f = ABFileBathy(depthfile, 'r', idm=grid.nx, jdm=grid.ny)
        mask = f.read_field('depth').mask
        f.close()
        return mask


    def read_var(self, **kwargs):
        ##check name in kwargs and read the variables from file
        assert 'name' in kwargs, 'please specify which variable to get, name=?'
        name = kwargs['name']
        assert name in variables, 'variable name '+name+' not listed in variables'
        fname = filename(path, **kwargs)

        if 'k' in kwargs:
            ##Note: ocean level indices are negative in assim_tools.state
            ##      but in abfiles, they are defined as positive indices
            k = -kwargs['k']
        else:
            k = -variables[name]['levels'][-1]  ##get the first level if not specified
        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            mask = None

        if 'is_vector' in kwargs:
            is_vector = kwargs['is_vector']
        else:
            is_vector = variables[name]['is_vector']

        if 'units' in kwargs:
            units = kwargs['units']
        else:
            units = variables[name]['units']

        f = ABFileRestart(fname, 'r', idm=grid.nx, jdm=grid.ny)
        if is_vector:
            var1 = f.read_field(variables[name]['name'][0], level=k, tlevel=1, mask=mask)
            var2 = f.read_field(variables[name]['name'][1], level=k, tlevel=1, mask=mask)
            var = np.array([var1, var2])
        else:
            var = f.read_field(variables[name]['name'], level=k, tlevel=1, mask=mask)
        f.close()

        var = units_convert(units, variables[name]['units'], var)
        return var


    def write_var(path, grid, var, **kwargs):
        ##check name in kwargs
        assert 'name' in kwargs, 'please specify which variable to write, name=?'
        name = kwargs['name']
        assert name in variables, 'variable name '+name+' not listed in variables'
        fname = filename(path, **kwargs)

        ##same logic for setting level indices as in read_var()
        if 'k' in kwargs:
            k = -kwargs['k']
        else:
            k = -variables[name]['levels'][-1]
        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            mask = None

        ##open the restart file for over-writing
        ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
        f = ABFileRestart(fname, 'r+', idm=grid.nx, jdm=grid.ny, mask=True)

        ##convert units back if necessary
        var = units_convert(kwargs['units'], variables[name]['units'], var, inverse=True)

        if kwargs['is_vector']:
            for i in range(2):
                f.overwrite_field(var[i,...], mask, variables[name]['name'][i], level=k, tlevel=1)
        else:
            f.overwrite_field(var, mask, variables[name]['name'], level=k, tlevel=1)
        f.close()


    @lru_cache(maxsize=3)
    def z_coords(self, **kwargs):
        """calculate vertical coordinates given the 3D model state
        """
        ##check if level is provided
        assert 'k' in kwargs, 'missing level index in kwargs for z_coords calc, level=?'

        z = np.zeros(grid.x.shape)
        if kwargs['k'] == 0:
            ##if level index is 0, this is the surface, so just return zeros
            return z
        else:
            ##get layer thickness above level k, convert to z_units, and integrate to total depth
            for k in [k for k in levels if k>=kwargs['k']]:
                rec = kwargs.copy()
                rec['name'] = 'ocean_layer_thick'
                rec['units'] = variables['ocean_layer_thick']['units']
                rec['k'] = k
                d = read_var(path, grid, **rec)
                if kwargs['units'] == 'm':
                    onem = 9806.
                    z -= d/onem  ##accumulate depth in meters, negative relative to surface
                else:
                    raise ValueError('do not know how to calculate z_coords for z_units = '+kwargs['units'])
            return z


    def run(self, task_id=0, task_nproc=256, **kwargs):
        self.run_status = 'running'

        host = kwargs['host']
        nedas_dir = kwargs['nedas_dir']

        fname = self.filename(**kwargs)
        run_dir = os.path.dirname(fname)

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        time_start = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time_start + forecast_period * dt1h
        input_file = self.filename(**kwargs)
        kwargs_out = {**kwargs, 'time':next_time}
        output_file = self.filename(**kwargs_out)

        ##build the shell command line
        shell_cmd =  "source "+topaz4_src+"; "   ##enter topaz v4 env
        shell_cmd += "cd "+run_dir+"; "          ##enter run directory
        shell_cmd += ""

        log_file = os.path.join(run_dir, "run.log")

        namelist(self, time_start, forecast_period, run_dir)

        self.run_process = subprocess.Popen(shell_cmd, shell=True, preexec_fn=os.setsid)
        self.run_process.wait()

        with open(log_file, 'rt') as f:
            if '' not in f.read():
                raise RuntimeError('errors in '+log_file)
        if not os.path.exists(output_file):
            raise RuntimeError('output file not found: '+output_file)


    def kill(self):
        os.killpg(os.getpgid(self.run_process.pid), signal.SIGKILL)

