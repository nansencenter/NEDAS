import numpy as np
import os
import subprocess
from functools import lru_cache

from NEDAS.utils.conversion import units_convert, dt1h
from NEDAS.utils.shell_utils import run_job
from NEDAS.models import Model
from ..abfile import ABFileRestart, ABFileBathy
from ..model_grid import get_topaz_grid
from .namelist import namelist

class Topaz4Model(Model):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        levels = np.arange(1, 51, 1)
        level_sfc = np.array([0])
        self.variables = {
            'ocean_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'levels':levels, 'units':'m/s'},
            'ocean_layer_thick': {'name':'dp', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'Pa'},
            'ocean_temp': {'name':'temp', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'K'},
            'ocean_saln': {'name':'saln', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'psu'},
            'ocean_surf_height': {'name':'msshb', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m'},
            'ocean_surf_temp': {'name':'sstb', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'K'},
            'ocean_b_velocity':  {'name':('ubavg', 'vbavg'), 'dtype':'float', 'is_vector':True, 'levels':level_sfc, 'units':'m/s'},
            'ocean_b_press': {'name':'pbavg', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'Pa'},
            'ocean_mixl_depth': {'name':'dpmixl', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'Pa'},
            }

        self.z_units = 'm'
        
        ##model grid
        grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
        self.grid = get_topaz_grid(grid_info_file)

        self.depthfile = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}.a')
        f = ABFileBathy(self.depthfile, 'r', idm=self.grid.nx, jdm=self.grid.ny)
        depth = f.read_field('depth')
        f.close()
        self.depth = -depth.data
        self.grid.mask = depth.mask
        
    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        tstr = kwargs['time'].strftime('%Y_%j_%H_0000')
        return os.path.join(kwargs['path'], mstr[1:], 'TP4restart'+tstr+mstr+'.a')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        f = ABFileRestart(fname, 'r', idm=self.grid.nx, jdm=self.grid.ny)
        if rec['is_vector']:
            var1 = f.read_field(rec[name]['name'][0], level=kwargs['k'], tlevel=1, mask=None)
            var2 = f.read_field(rec[name]['name'][1], level=kwargs['k'], tlevel=1, mask=None)
            var = np.array([var1, var2])
        else:
            var = f.read_field(rec[name]['name'], level=kwargs['k'], tlevel=1, mask=None)
        f.close()

        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##open the restart file for over-writing
        ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
        f = ABFileRestart(fname, 'r+', idm=self.grid.nx, jdm=self.grid.ny)

        ##convert units back if necessary
        var = units_convert(kwargs['units'], rec['units'], var)

        if rec['is_vector']:
            for i in range(2):
                f.overwrite_field(var[i,...], None, rec['name'][i], level=kwargs['k'], tlevel=1)
        else:
            f.overwrite_field(var, None, rec['name'], level=kwargs['k'], tlevel=1)
        f.close()

    @lru_cache(maxsize=3)
    def z_coords(self, **kwargs):
        """calculate vertical coordinates given the 3D model state
        """
        """
        Calculate vertical coordinates given the 3D model state
        Return:
        - z: np.array
        The corresponding z field
        """
        ##some defaults if not set in kwargs
        if 'k' not in kwargs:
            kwargs['k'] = 0

        z = np.zeros(self.grid.x.shape)

        if kwargs['k'] == 0:
            ##if level index is 0, this is the surface, so just return zeros
            return z

        else:
            ##get layer thickness and convert to units
            rec = kwargs.copy()
            rec['name'] = 'ocean_layer_thick'
            rec['units'] = self.variables['ocean_layer_thick']['units'] ##should be Pa
            if self.z_units == 'm':
                dz = - self.read_var(**rec) / self.onem ##in meters, negative relative to surface
            elif self.z_units == 'Pa':
                dz = self.read_var(**rec)
            else:
                raise ValueError('do not know how to calculate z_coords for z_units = '+self.z_units)

            ##use recursive func, get previous layer z and add dz
            kwargs['k'] -= 1
            z_prev = self.z_coords(**kwargs)
            return z_prev + dz

    def preprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        init_file = self.self.filename(**{**kwargs, 'path':self.ens_init_dir})
        input_file = self.self.filename(**kwargs)
        os.system("mkdir -p "+os.path.dirname(input_file))
        os.system("cp "+init_file+" "+input_file)
        os.system("cp "+init_file.replace('.a', '.b')+" "+input_file.replace('.a', '.b'))

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        input_file = self.self.filename(**kwargs)
        run_dir = os.path.dirname(input_file)
        os.system("mkdir -p "+run_dir)
        os.chdir(run_dir)
        log_file = os.path.join(run_dir, "run.log")
        os.system('touch '+log_file)

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        kwargs_out = {**kwargs, 'time':next_time}
        output_file = self.self.filename(**kwargs_out)

        ##create namelist config files
        namelist(self, time, forecast_period, run_dir)

        ##link files
        partit_file = os.path.join(self.basedir, 'topo', 'partit', f'depth_{self.R}_{self.T}.{self.nproc:04d}')
        os.system("cp "+partit_file+" patch.input")

        for ext in ['.a', '.b']:
            os.system("ln -fs "+os.path.join(self.basedir, 'topo', 'regional.grid'+ext)+" regional.grid"+ext)
            os.system("ln -fs "+os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}'+ext)+" regional.depth"+ext)
            os.system("ln -fs "+os.path.join(self.basedir, 'topo', 'tbaric'+ext)+" tbaric"+ext)
        os.system("ln -fs "+os.path.join(self.basedir, 'topo', 'grid.info')+" grid.info")

        ##TODO: switches for other forcing options
        if self.forcing_frc == 'era5':
            forcing_path = self.era5_path
        if self.forcing_frc == 'era40':
            pass
        os.system("ln -fs "+forcing_path+" .")
        os.system("ln -fs "+os.path.join(self.basedir, 'force', 'other', 'iwh_tabulated.dat')+" .")
        for ext in ['.a', '.b']:
            if self.priver == 1:
                os.system("ln -fs "+os.path.join(self.basedir, 'force', 'rivers', self.E, 'rivers'+ext)+" forcing.rivers"+ext)
            if self.jerlv0 == 0:
                os.system("ln -fs "+os.path.join(self.basedir, 'force', 'seawifs', 'kpar'+ext)+" forcing.kpar"+ext)
            if self.relax == 1:
                for comp in ['saln', 'temp', 'intf', 'rmu']:
                    os.system("ln -fs "+os.path.join(self.basedir, 'relax', self.E, 'relax_'+comp[:3]+ext)+" relax."+comp+ext)
            os.system("ln -fs "+os.path.join(self.basedir, 'relax', self.E, 'thkdf4'+ext)+" thkdf4"+ext)
        os.system("ln -fs "+os.path.join(self.basedir, 'relax', self.E, 'clim_tran.txt')+" .")
        # if self.gpflag: TODO
        # if self.nestoflag:
        # if self.nestiflag:
        # if self.tideflag:

        model_src = os.path.join(self.basedir, 'setup.src')
        model_exe = os.path.join(self.basedir, f'Build_V{self.V}_X{self.X}', 'hycom')

        ##build the shell command line
        shell_cmd =  ". "+model_src+"; "   ##enter topaz v4 env
        shell_cmd += "cd "+run_dir+"; "          ##enter run directory
        shell_cmd += f"JOB_EXECUTE {model_exe} {kwargs['member']+1} >& run.log"

        for tr in range(2):  ##number of tries
            with open(log_file, 'rt') as f:
                if '(normal)' in f.read():
                    break
            self.run_process = subprocess.Popen(shell_cmd, shell=True)
            self.run_process.wait()
            run_job(shell_cmd, job_name='topaz4_run', run_dir=run_dir,
                    nproc=self.nproc, offset=task_id*self.nproc_per_run,
                    walltime=self.walltime, **kwargs)

        with open(log_file, 'rt') as f:
            if '(normal)' not in f.read():
                raise RuntimeError('errors in '+log_file)
        if not os.path.exists(output_file):
            raise RuntimeError('output file not found: '+output_file)
