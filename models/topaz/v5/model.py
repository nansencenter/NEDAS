import numpy as np
import os
import subprocess
import inspect
import signal
from functools import lru_cache
from datetime import datetime, timedelta

from config import parse_config
from utils.conversion import units_convert, t2s, s2t, dt1h
from utils.netcdf_lib import nc_read_var, nc_write_var
from utils.shell_utils import run_command
from utils.progress import watch_log, find_keyword_in_file, watch_files

from .namelist import namelist
from ..time_format import dayfor, jultodate
from ..abfile import ABFileRestart, ABFileArchv, ABFileBathy, ABFileGrid, ABFileForcing
from ..model_grid import get_topaz_grid, stagger, destagger
from ...model_config import ModelConfig

class Topaz5Model(ModelConfig):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        levels = np.arange(self.kdm) + 1  ##ocean levels, from top to bottom, k=1..kdm
        level_sfc = np.array([0])    ##some variables are only defined on surface level k=0
        level_ncat = np.array([5])   ##some ice variables have 5 categories, treating them as levels also indexed by k

        self.hycom_variables = {
            'ocean_velocity':    {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':levels, 'units':'m/s'},
            'ocean_layer_thick': {'name':'dp', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':levels, 'units':'Pa'},
            'ocean_temp':        {'name':'temp', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':levels, 'units':'K'},
            'ocean_saln':        {'name':'saln', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':levels, 'units':'psu'},
            'ocean_b_velocity':  {'name':('ubavg', 'vbavg'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m/s'},
            'ocean_b_press':     {'name':'pbavg', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_mixl_depth':  {'name':'dpmixl', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_bot_press':   {'name':'pbot', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'Pa'},
            'ocean_bot_dense':   {'name':'thkk', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'?'},
            'ocean_bot_montg_pot': {'name':'psik', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'?'},
            }

        self.cice_variables = {
            'seaice_velocity': {'name':('uvel', 'vvel'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':level_sfc, 'units':'m/s'},
            'seaice_conc_n':   {'name':'aicen', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'%'},
            'seaice_thick_n':  {'name':'vicen', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':level_ncat, 'units':'m'},
            }

        self.diag_variables = {
            'seaice_conc':  {},
            }

        self.atmos_forcing_variables = {
            'atmos_surf_velocity': {'name':('wndewd', 'wndnwd'), 'dtype':'float', 'is_vector':True, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'m/s'},
            'atmos_surf_temp':     {'name':'airtmp', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'C'},
            'atmos_surf_dewpoint': {'name':'dewpt', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'K'},
            'atmos_surf_press':    {'name':'mslprs', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'Pa'},
            'atmos_precip':        {'name':'precip', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'m/s'},
            'atmos_down_longwave': {'name':'radflx', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'W/m2'},
            'atmos_down_shortwave': {'name':'shwflx', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'W/m2'},
            'atmos_vapor_mix_ratio': {'name':'vapmix', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'kg/kg'},
            }
        self.force_synoptic_names = [name for r in self.atmos_forcing_variables.values() for name in (r['name'] if isinstance(r['name'], tuple) else [r['name']])]

        self.variables = {**self.hycom_variables, **self.cice_variables, **self.atmos_forcing_variables}

        ##model grid
        grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
        self.grid = get_topaz_grid(grid_info_file)

        self.depthfile = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}.a')
        f = ABFileBathy(self.depthfile, 'r', idm=self.grid.nx, jdm=self.grid.ny)
        depth = f.read_field('depth')
        f.close()
        self.depth = -depth.data
        self.mask = depth.mask

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        ##filename for each model component
        if kwargs['name'] in self.hycom_variables:
            tstr = kwargs['time'].strftime('%Y_%j_%H_0000')
            return os.path.join(kwargs['path'], 'restart.'+tstr+mstr+'.a')

        elif kwargs['name'] in self.cice_variables:
            tstr = kwargs['time'].strftime('%Y-%m-%d-00000')
            return os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')

        elif kwargs['name'] in self.atmos_forcing_variables:
            return os.path.join(kwargs['path'], mstr[1:], 'SCRATCH', 'forcing')

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)

        ##get the variable from restart files
        name = kwargs['name']
        if name in self.hycom_variables:
            rec = self.hycom_variables[name]
            f = ABFileRestart(fname, 'r', idm=self.grid.nx, jdm=self.grid.ny)
            if rec['is_vector']:
                var1 = f.read_field(rec['name'][0], level=kwargs['k'], tlevel=1, mask=None)
                var2 = f.read_field(rec['name'][1], level=kwargs['k'], tlevel=1, mask=None)
                var = np.array([var1, var2])
            else:
                var = f.read_field(rec['name'], level=kwargs['k'], tlevel=1, mask=None)
            f.close()

        elif name in self.cice_variables:
            rec = self.cice_variables[name]
            if rec['is_vector']:
                var1 = nc_read_var(fname, rec['name'][0])
                var2 = nc_read_var(fname, rec['name'][1])
                var = np.array([var1, var2])
            else:
                var = nc_read_var(fname, rec['name'])

        elif name in self.atmos_forcing_variables:
            rec = self.atmos_forcing_variables[name]
            dtime = (kwarts['time'] - datetime(1900,12,31)) / (24*dt1h)
            if rec['is_vector']:
                f1 = ABFileForcing(fname+'.'+rec['name'][0], 'r')
                var1 = f1.read_field(rec['name'][0], dtime)
                f2 = ABFileForcing(fname+'.'+rec['name'][1], 'r')
                var2 = f2.read_field(rec['name'][1], dtime)
                var = np.array([var1, var2])
                f1.close()
                f2.close()
            else:
                f = ABFileForcing(fname+'.'+rec['name'], 'r')
                var = f.read_field(rec['name'], dtime)
                f.close()

        ##convert units if necessary
        var = units_convert(kwargs['units'], self.variables[name]['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)

        ##convert back to old units
        var = units_convert(kwargs['units'], self.variables[self.name]['units'], var, inverse=True)

        name = kwargs['name']
        if name in self.hycom_variables:
            rec = self.hycom_variables[name]
            ##open the restart file for over-writing
            ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
            f = ABFileRestart(fname, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
            if rec['is_vector']:
                for i in range(2):
                    f.overwrite_field(var[i,...], None, rec['name'][i], level=kwargs['k'], tlevel=1)
            else:
                f.overwrite_field(var, None, rec['name'], level=kwargs['k'], tlevel=1)
            f.close()

        elif name in self.cice_variables:
            rec = self.cice_variables[name]
            if rec['is_vector']:
                for i in range(2):
                    if rec['name'][i][-2:] == '_n':  ##ncat variable
                        dims = {'ncat':None, 'nj':self.grid.ny, 'ni':self.grid.nx}
                        recno = {'ncat':kwargs['k']}
                    else:
                        dims = {'nj':self.grid.ny, 'ni':self.grid.nx}
                        recno = None
                    nc_write_var(fname, dims, rec['name'][i], var[i,...], recno=recno, comm=kwargs['comm'])
            else:
                if rec['name'][-2:] == '_n':  ##ncat variable
                    dims = {'ncat':None, 'nj':self.grid.ny, 'ni':self.grid.nx}
                    recno = {'ncat':kwargs['k']}
                else:
                    dims = {'nj':self.grid.ny, 'ni':self.grid.nx}
                    recno = None
                nc_write_var(fname, dims, rec['name'], var, recno=recno, comm=kwargs['comm'])

        elif name in self.atmos_forcing_variables:
            rec = self.atmos_forcing_variables[name]
            dtime = (kwargs['time'] - datetime(1900,12,31)) / timedelta(days=1)
            if rec['is_vector']:
                for i in range(2):
                    f = ABFileForcing(fname+'.'+rec['name'][i], 'r+')
                    f.overwrite_field(var[i,...], None, rec['name'][i], dtime)
                    f.close()
            else:
                f = ABFileForcing(fname+'.'+rec['name'], 'r+')
                f.overwrite_field(var, None, rec['name'], dtime)
                f.close()

    @lru_cache(maxsize=3)
    def z_coords(self, **kwargs):
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

        offset = task_id * self.nproc_per_util
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr[1:], 'SCRATCH')
        ##make sure model run directory exists
        run_command(f"mkdir -p {run_dir}")

        ##generate namelists, blkdat, ice_in, etc.
        namelist(self, time, forecast_period, run_dir)

        ##copy synoptic forcing fields from a long record in basedir, will be perturbed later
        for varname in self.force_synoptic_names:
            forcing_file = os.path.join(self.basedir, 'force', 'synoptic', self.E, varname)
            forcing_file_out = os.path.join(run_dir, 'forcing.'+varname)
            f = ABFileForcing(forcing_file, 'r')
            fo = ABFileForcing(forcing_file_out, 'w', idm=f.idm, jdm=f.jdm, cline1=f._cline1, cline2=f._cline2)
            t = time
            dt = self.forcing_dt
            rdtime = dt / 24
            while t <= next_time:
                dtime1 = dayfor(self.yrflag, t.year, int(t.strftime('%j')), t.hour)
                fld = f.read_field(varname, dtime1)
                fo.write_field(fld, None, varname, dtime1, rdtime)
                t += dt * dt1h
            f.close()
            fo.close()

        ##link necessary files for model run
        shell_cmd = f"cd {run_dir}; "
        ##partition setting
        partit_file = os.path.join(self.basedir, 'topo', 'partit', f'depth_{self.R}_{self.T}.{self.nproc:04d}')
        shell_cmd += f"ln -fs {partit_file} patch.input; "
        ##topo files
        for ext in ['.a', '.b']:
            file = os.path.join(self.basedir, 'topo', 'regional.grid'+ext)
            shell_cmd += f"ln -fs {file} .; "
            file = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}'+ext)
            shell_cmd += f"ln -fs {file} regional.depth{ext}; "
        file = os.path.join(self.basedir, 'topo', f'kmt_{self.R}_{self.T}.nc')
        shell_cmd += f"ln -fs {file} cice_kmt.nc; "
        file = os.path.join(self.basedir, 'topo', 'cice_grid.nc')
        shell_cmd += f"ln -fs {file} .; "
        ##nest files
        nest_dir = os.path.join(self.basedir, 'nest', self.E)
        shell_cmd += f"ln -fs {nest_dir} nest; "
        ##TODO: there is extra logic in nhc_root/bin/expt_preprocess.sh to be added here
        ##relax files
        for ext in ['.a', '.b']:
            for varname in ['intf', 'saln', 'temp']:
                file = os.path.join(self.basedir, 'relax', self.E, 'relax_'+varname[:3]+ext)
                shell_cmd += f"ln -fs {file} {'relax.'+varname+ext}; "
            for varname in ['thkdf4', 'veldf4']:
                file = os.path.join(self.basedir, 'relax', self.E, varname+ext)
                shell_cmd += f"ln -fs {file} {varname+ext}; "
        ##other forcing files
        for ext in ['.a', '.b']:
            ##rivers
            file = os.path.join(self.basedir, 'force', 'rivers', self.E, 'rivers'+ext)
            shell_cmd += f"ln -fs {file} {'forcing.rivers'+ext}; "
            ##seawifs
            file = os.path.join(self.basedir, 'force', 'seawifs', 'kpar'+ext)
            shell_cmd += f"ln -fs {file} {'forcing.kpar'+ext}; "
        run_command(shell_cmd)

        ##copy restart files from restart_dir
        restart_dir = kwargs['restart_dir']
        job_submit_cmd = kwargs['job_submit_cmd']
        tstr = time.strftime('%Y_%j_%H_0000')
        for ext in ['.a', '.b']:
            file = os.path.join(restart_dir, 'restart.'+tstr+mstr+ext)
            file1 = os.path.join(path, 'restart.'+tstr+mstr+ext)
            run_command(f"{job_submit_cmd} 1 {offset} cp -fL {file} {file1}")
            run_command(f"ln -fs {file1} {os.path.join(run_dir, 'restart.'+tstr+ext)}")
        run_command(f"mkdir -p {os.path.join(run_dir, 'cice')}")
        tstr = time.strftime('%Y-%m-%d-00000')
        file = os.path.join(restart_dir, 'iced.'+tstr+mstr+'.nc')
        file1 = os.path.join(path, 'iced.'+tstr+mstr+'.nc')
        run_command(f"{job_submit_cmd} 1 {offset} cp -fL {file} {file1}")
        run_command(f"ln -fs {file1} {os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')}")
        run_command(f"echo {os.path.join('.', 'cice', 'iced.'+tstr+'.nc')} > {os.path.join(run_dir, 'cice', 'ice.restart_file')}")

    def postprocess(self, task_id=0, **kwargs):
        ##TODO: run fixhycom_cice here
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(kwargs['path'], mstr[1:], 'SCRATCH')
        run_command(f"mkdir -p {run_dir}")  ##make sure model run directory exists
        log_file = os.path.join(run_dir, "run.log")
        run_command("touch "+log_file)

        ##check if input file exists
        input_files = []
        tstr = time.strftime('%Y_%j_%H_0000')
        for ext in ['.a', '.b']:
            input_files.append(os.path.join(run_dir, 'restart.'+tstr+ext))
        tstr = time.strftime('%Y-%m-%d-00000')
        input_files.append(os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc'))
        for file in input_files:
            if not os.path.exists(file):
                raise RuntimeError(f"topaz.v5.model.run: input file missing: {file}")

        ##early exit if the run is already finished
        if find_keyword_in_file(log_file, 'Exiting hycom_cice'):
            return

        offset = task_id*self.nproc_per_run
        job_submit_cmd = kwargs['job_submit_cmd']

        ##build the shell command line
        model_exe = os.path.join(self.basedir, f'expt_{self.X}', 'build', f'src_{self.V}ZA-07Tsig0-i-sm-sse_relo_mpi', 'hycom_cice')
        shell_cmd =  "source "+self.model_env+"; "  ##enter topaz5 env
        shell_cmd += "cd "+run_dir+"; "             ##enter run directory
        shell_cmd += job_submit_cmd+f" {self.nproc} {offset} "+model_exe+" >& run.log"

        self.run_process = subprocess.Popen(shell_cmd, shell=True)
        self.run_process.wait()

        ##check output
        watch_log(log_file, 'Exiting hycom_cice')

        ##move the output restart files to forecast_dir
        tstr = next_time.strftime('%Y_%j_%H_0000')
        for ext in ['.a', '.b']:
            file1 = os.path.join(run_dir, 'restart.'+tstr+ext)
            file2 = os.path.join(kwargs['path'], 'restart.'+tstr+mstr+ext)
            run_command(f"mv {file1} {file2}")
        tstr = next_time.strftime('%Y-%m-%d-00000')
        file1 = os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')
        file2 = os.path.join(kwargs['path'], 'iced.'+tstr+mstr+'.nc')
        run_command(f"mv {file1} {file2}")

