import os
import numpy as np
from pyproj import Proj
from NEDAS.grid import Grid
from NEDAS.utils.conversion import dt1h
from NEDAS.utils.shell_utils import run_command, run_job, makedir
# from .namelist import namelist
# from .bin_io import read_
from NEDAS.models import Model

class WRFModel(Model):

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

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
        kwargs = super().parse_kwargs(**kwargs)
        tstr = kwargs['time'].strftime('%Y-%m-%d_%H:%M:%S')
        return os.path.join(kwargs['path'], 'wrfout_'+tstr+'.nc')

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

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        pass

    def write_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        pass

    def z_coords(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        pass

    def preprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        pass

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        fname = self.filename(**kwargs)
        run_dir = os.path.dirname(fname)
        print('running wrf model in '+run_dir, flush=True)

        wrf_src = os.path.join(self.model_code_dir, 'setup.src')
        wrf_exe = os.path.join(self.model_code_dir, 'main', 'wrf.exe')

        log_file = 'rsl.error.0000'

        ##collect restart variables from bin and write to wrfrst

        ##build the run command
        shell_cmd = ". "+wrf_src+"; "   ##enter wrf env
        shell_cmd += f"JOB_EXECUTE {wrf_exe} >& run.log"

        run_job(shell_cmd, job_name='wrf.run', run_dir=run_dir,
                nproc=self.nproc_per_run, offset=task_id*self.nproc_per_run,
                walltime=self.walltime, **kwargs)

        # "SUCCESS COMPLETE" in log_file

        ##wrfrst at nexttime collect to bin file

