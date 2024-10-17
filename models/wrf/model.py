import numpy as np
import os
import inspect
import signal
import subprocess
from pyproj import Proj

from config import parse_config
from grid import Grid
from utils.conversion import t2s, s2t, dt1h

# from .namelist import namelist
# from .bin_io import read_

class Model(object):

    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        ##derived default values
        self.ref_x = self.e_we[0] / 2
        self.ref_y = self.e_sn[0] / 2

        levels = np.arange(1, self.e_vert[0]+1, 1)  ##use domain 1 setting for z levels
        level_sfc = np.array([0])
        self.variables = {
                'atmos_velocity': {'name':('u_1', 'v_1'), 'dtype':'float', 'is_vector':True, 'levels':levels, 'units':'m/s'},
                'atmos_surf_velocity': {'name':('U10', 'V10'), 'dtype':'float', 'is_vector':True, 'levels':level_sfc, 'units':'m/s'},
                'atmos_temp': {'name':'T', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'K'},
                'atmos_pres': {'name':'P', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'Pa'},
                'atmos_q_vapor': {'name':'QVAPOR', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'kg/kg'},
                }

        self.run_process = None
        self.run_status = 'pending'


    def filename(self, **kwargs):
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'

        tstr = kwargs['time'].strftime('%Y-%m-%d_%H:%M:%S')
        return os.path.join(path, 'wrfout_'+tstr+'.nc')


    def read_grid(self, **kwargs):

        if self.map_proj == 'polar':
            proj = Proj(f'+proj=stere +lat_0={self.ref_lat} +lon_0={self.ref_lon} +lat_ts={self.truelat1}')

        elif self.map_proj == 'lambert':
            proj = Proj(f'+proj=lcc +lat_0={self.ref_lat} +lon_0={self.ref_lon} +lat_1={self.truelat1} +lat_2={self.truelat2}')

        elif self.map_proj == 'mercator':
            proj = Proj(f'+proj=merc +lat_0={self.ref_lat} +lon_0={self.ref_lon}')

        elif self.map_proj == 'lat-lon':
            proj = Proj(f'+proj=longlat')

        else:
            raise ValueError(f'unknown map_proj type {self.map_proj}')

        ##staggering here?
        ##again, use domain 1 settings for grid
        x_coords = (np.arange(self.e_we[0]) - self.ref_x) * self.dx
        y_coords = (np.arange(self.e_sn[0]) - self.ref_y) * self.dy
        x, y = np.meshgrid(x_coords, y_coords)

        self.grid = Grid(proj, x, y)


    def write_grid(self, grid, **kwargs):
        pass


    def read_var(self, **kwargs):
        pass


    def write_var(self, **kwargs):
        pass


    def z_coords(self, **kwargs):
        pass


    def genrate_initial_condition(self, task_id=0, task_nproc=1, **kwargs):
        pass


    def run(self, task_id=0, task_nproc=1, **kwargs):

        self.run_status = 'running'

        nedas_dir = kwargs['nedas_dir']
        job_submit_cmd = kwargs['job_submit_cmd']
        model_code_dir = kwargs['model_code_dir']
        model_data_dir = kwargs['model_data_dir']

        fname = self.filename(**kwargs)
        run_dir = os.path.dirname(fname)
        print('running wrf model in '+run_dir, flush=True)

        wrf_src = os.path.join(model_code_dir, 'setup.src')
        wrf_exe = os.path.join(model_code_dir, 'main', 'wrf.exe')

        log_file = 'rsl.error.0000'

        ##collect restart variables from bin and write to wrfrst

        ##build the run command
        shell_cmd = "source "+wrf_src+"; "   ##enter wrf env
        shell_cmd += job_submit_cmd+f" {task_nproc} {task_id*task_nproc} "+wrf_exe+" >& run.log"

        self.run_process = subprocess.Popen(shell_cmd, shell=True)
        self.run_process.wait()

        returncode = self.run_process.returncode
        if returncode is not None and returncode < 0:
            ##kill signal received, exit the run
            return

        # "SUCCESS COMPLETE" in log_file

        ##wrfrst at nexttime collect to bin file

