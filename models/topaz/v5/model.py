import numpy as np
import os
import subprocess
import inspect
import signal
from datetime import datetime, timedelta
from functools import lru_cache

from utils.conversion import units_convert, t2s, s2t, dt1h
from utils.netcdf_lib import nc_read_var, nc_write_var
from utils.dir_def import forecast_dir
from config import parse_config

from .namelist import namelist
from ..abfile import ABFileRestart, ABFileArchv, ABFileBathy, ABFileGrid, ABFileForcing
from ..model_grid import get_topaz_grid, stagger, destagger

class Model(object):
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

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
        self.diag_variables = {}
        self.forcing_variables = {
            'atmos_surf_velocity': {'name':('wndewd', 'wndnwd'), 'dtype':'float', 'is_vector':True, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'m/s'},
            'atmos_surf_temp':     {'name':'airtmp', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'C'},
            'atmos_surf_dewpoint': {'name':'dewpt', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'K'},
            'atmos_surf_press':    {'name':'mslprs', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'Pa'},
            'atmos_precip':        {'name':'precip', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'m/s'},
            'atmos_down_longwave': {'name':'radflx', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'W/m2'},
            'atmos_down_shortwave': {'name':'shwflx', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'W/m2'},
            'atmos_vapor_mix_ratio': {'name':'vapmix', 'dtype':'float', 'is_vector':False, 'dt':self.forcing_dt, 'levels':level_sfc, 'units':'kg/kg'},
            }
        self.variables = {**self.hycom_variables, **self.cice_variables, **self.forcing_variables}

        self.z_units = 'm'
        self.read_grid()

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        """
        Parse kwargs and find matching filename
        for keys in kwargs that are not set, here we define the default values
        key values in kwargs will also be checked for erroneous values here
        """
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'

        if 'member' in kwargs and kwargs['member'] is not None:
            assert kwargs['member'] >= 0, 'member index shall be >= 0'
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        if 'name' not in kwargs:
            name = list(self.variables.keys())[0]  ##if not specified, use first variable listed
        else:
            name = kwargs['name']

        assert 'time' in kwargs, 'time is not defined in kwargs'
        time = kwargs['time']
        assert isinstance(kwargs['time'], datetime), 'time shall be a datetime object'

        ##filename for each model component
        if name in self.hycom_variables:
            tstr = kwargs['time'].strftime('%Y_%j_%H_0000')
            return os.path.join(path, 'restart.'+tstr+mstr+'.a')

        elif name in self.cice_variables:
            tstr = kwargs['time'].strftime('%Y-%m-%d-00000')
            return os.path.join(path, 'iced.'+tstr+mstr+'.nc')

        elif name in self.forcing_variables:
            return os.path.join(path, mstr[1:], 'SCRATCH', 'forcing')

        else:
            raise ValueError(f"variable name '{name}' is not defined for topaz.v5 model!")

    def read_grid(self, **kwargs):
        grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
        self.grid = get_topaz_grid(grid_info_file)

    def read_mask(self, **kwargs):
        depthfile = path+'/topo/depth.a'
        f = ABFileBathy(depthfile, 'r', idm=grid.nx, jdm=grid.ny)
        mask = f.read_field('depth').mask
        f.close()
        self.mask = mask

    def read_depth(self, **kwargs):
        depthfile = path+'/topo/depth.a'
        f = ABFileBathy(depthfile, 'r', idm=self.grid.nx, jdm=self.grid.ny)
        depth = f.read_field('depth').data
        f.close()
        return -depth

    def read_var(self, **kwargs):
        """
        Read the state variable from a model restart file

        Inputs:
        - **kwargs: time, level, and member to pinpoint where to get the variable

        Return:
        - var: np.array
        a 2D field defined on grid with the state variable
        """
        ##check name in kwargs and read the variables from file
        assert 'name' in kwargs, 'please specify which variable to get, name=?'
        name = kwargs['name']

        fname = self.filename(**kwargs)

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = self.variables[name]['levels'][0]  ##get the first level if not specified

        time = kwargs['time']

        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            mask = None

        is_vector = self.variables[name]['is_vector']

        if 'units' in kwargs:
            units = kwargs['units']
        else:
            units = self.variables[name]['units']

        ##get the variable from restart files
        if name in self.hycom_variables:
            rec = self.hycom_variables[name]
            f = ABFileRestart(fname, 'r', idm=self.grid.nx, jdm=self.grid.ny)
            if is_vector:
                var1 = f.read_field(rec['name'][0], level=k, tlevel=1, mask=mask)
                var2 = f.read_field(rec['name'][1], level=k, tlevel=1, mask=mask)
                var = np.array([var1, var2])
            else:
                var = f.read_field(rec['name'], level=k, tlevel=1, mask=mask)
            f.close()

        elif name in self.cice_variables:
            rec = self.cice_variables[name]
            if is_vector:
                var1 = nc_read_var(fname, rec['name'][0])
                var2 = nc_read_var(fname, rec['name'][1])
                var = np.array([var1, var2])
            else:
                var = nc_read_var(fname, rec['name'])

        elif name in self.forcing_variables:
            rec = self.forcing_variables[name]
            dtime = (time - datetime(1900,12,31)) / timedelta(days=1)
            if is_vector:
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
        var = units_convert(units, self.variables[name]['units'], var)
        return var

    def write_var(self, var, **kwargs):
        """
        Write a variable (overwrite) to the model restart file
        """
        ##check name in kwargs
        assert 'name' in kwargs, 'please specify which variable to write, name=?'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in model.variables'
        fname = self.filename(**kwargs)

        ##same logic for setting level indices as in read_var()
        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = self.variables[name]['levels'][0]

        time = kwargs['time']

        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            mask = None

        is_vector = self.variables[name]['is_vector']

        if 'units' in kwargs:
            units = kwargs['units']
        else:
            units = self.variables[name]['units']
        ##convert back to old units
        var = units_convert(units, self.variables[name]['units'], var, inverse=True)

        if name in self.hycom_variables:
            rec = self.hycom_variables[name]
            ##open the restart file for over-writing
            ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
            f = ABFileRestart(fname, 'r+', idm=self.grid.nx, jdm=self.grid.ny, mask=True)
            if is_vector:
                for i in range(2):
                    f.overwrite_field(var[i,...], mask, rec['name'][i], level=k, tlevel=1)
            else:
                f.overwrite_field(var, mask, rec['name'], level=k, tlevel=1)
            f.close()

        elif name in self.cice_variables:
            rec = self.cice_variables[name]
            ##TODO: multiprocessor write to the same file is still broken, convert to binary file then back to solve this?

        elif name in self.forcing_variables:
            rec = self.forcing_variables[name]
            dtime = (time - datetime(1900,12,31)) / timedelta(days=1)
            if is_vector:
                for i in range(2):
                    f = ABFileForcing(fname+'.'+rec['name'][i], 'r+')
                    f.overwrite_field(var[i,...], mask, rec['name'][i], dtime)
                    f.close()
            else:
                f = ABFileForcing(fname+'.'+rec['name'], 'r+')
                f.overwrite_field(var, mask, rec['name'], dtime)
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
        if 'units' not in kwargs:
            kwargs['units'] = z_units
        if 'k' not in kwargs:
            kwargs['k'] = 0

        z = np.zeros(grid.x.shape)

        if kwargs['k'] == 0:
            ##if level index is 0, this is the surface, so just return zeros
            return z

        else:
            ##get layer thickness and convert to units
            rec = kwargs.copy()
            rec['name'] = 'ocean_layer_thick'
            rec['units'] = variables['ocean_layer_thick']['units'] ##should be Pa
            if kwargs['units'] == 'm':
                dz = - read_var(path, grid, **rec) / ONEM ##in meters, negative relative to surface
            elif kwargs['units'] == 'Pa':
                dz = read_var(path, grid, **rec)
            else:
                raise ValueError('do not know how to calculate z_coords for z_units = '+kwargs['units'])

            ##use recursive func, get previous layer z and add dz
            kwargs['k'] -= 1
            z_prev = z_coords(path, grid, **kwargs)
            return z_prev + dz

    def preprocess(self, task_id=0, task_nproc=1, **kwargs):
        job_submit_cmd = kwargs['job_submit_cmd']
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        restart_dir = kwargs['restart_dir']

        ##make sure model run directory exists
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'
        if 'member' in kwargs and kwargs['member'] is not None:
            assert kwargs['member'] >= 0, 'member index shall be >= 0'
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        run_dir = os.path.join(path, mstr[1:], 'SCRATCH')
        os.system(f"mkdir -p {run_dir}")

        ##generate namelists, blkdat, ice_in, etc.
        namelist(self, time, forecast_period, run_dir)

        ##prepare partition file
        partit_file = os.path.join(self.basedir, 'topo', 'partit', f'depth_{self.R}_{self.T}.{self.nproc:04d}')
        os.system(f"ln -fs {partit_file} {os.path.join(run_dir, 'patch.input')}")

        ##link topo files
        for ext in ['.a', '.b']:
            file = os.path.join(self.basedir, 'topo', 'regional.grid'+ext)
            os.system(f"ln -fs {file} {os.path.join(run_dir, 'regional.grid'+ext)}")
            file = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}'+ext)
            os.system(f"ln -fs {file} {os.path.join(run_dir, 'regional.depth'+ext)}")
        file = os.path.join(self.basedir, 'topo', f'kmt_{self.R}_{self.T}.nc')
        os.system(f"ln -fs {file} {os.path.join(run_dir, 'cice_kmt.nc')}")
        file = os.path.join(self.basedir, 'topo', 'cice_grid.nc')
        os.system(f"ln -fs {file} {os.path.join(run_dir, 'cice_grid.nc')}")

        ##link nest files
        nest_dir = os.path.join(self.basedir, 'nest', self.E)
        os.system(f"ln -fs {nest_dir} {os.path.join(run_dir, 'nest')}")

        ##TODO: there is extra logic in nhc_root/bin/expt_preprocess.sh to be added here
        ##link relax files
        for ext in ['.a', '.b']:
            for varname in ['intf', 'saln', 'temp']:
                file = os.path.join(self.basedir, 'relax', self.E, 'relax_'+varname[:3]+ext)
                os.system(f"ln -fs {file} {os.path.join(run_dir, 'relax.'+varname+ext)}")
            for varname in ['thkdf4', 'veldf4']:
                file = os.path.join(self.basedir, 'relax', self.E, varname+ext)
                os.system(f"ln -fs {file} {os.path.join(run_dir, varname+ext)}")

        ##copy forcing files, (not linking since each member will have a perturbed version)
        copy_file_src = []  ##list of files to copy
        copy_file_dst = []
        for ext in ['.a', '.b']:
            ##synoptic
            for varname in ['airtmp', 'dewpt', 'mslprs', 'precip', 'radflx', 'shwflx', 'vapmix', 'wndewd', 'wndnwd']:
                copy_file_src.append(os.path.join(self.basedir, 'force', 'synoptic', self.E, varname+ext))
                copy_file_dst.append(os.path.join(run_dir, 'forcing.'+varname+ext))
            ##rivers
            copy_file_src.append(os.path.join(self.basedir, 'force', 'rivers', self.E, 'rivers'+ext))
            copy_file_dst.append(os.path.join(run_dir, 'forcing.rivers'+ext))
            ##seawifs
            copy_file_src.append(os.path.join(self.basedir, 'force', 'seawifs', 'kpar'+ext))
            copy_file_dst.append(os.path.join(run_dir, 'forcing.kpar'+ext))

        ##restart files
        for ext in ['.a', '.b']:
            tstr = time.strftime('%Y_%j_%H_0000')
            copy_file_src.append(os.path.join(restart_dir, 'restart.'+tstr+mstr+ext))
            copy_file_dst.append(os.path.join(path, 'restart.'+tstr+mstr+ext))
            os.system(f"ln -fs {os.path.join(path, 'restart.'+tstr+mstr+ext)} {os.path.join(run_dir, 'restart.'+tstr+ext)}")

        os.system(f"mkdir -p {os.path.join(run_dir, 'cice')}")
        tstr = time.strftime('%Y-%m-%d-00000')
        copy_file_src.append(os.path.join(restart_dir, 'iced.'+tstr+mstr+'.nc'))
        copy_file_dst.append(os.path.join(path, 'iced.'+tstr+mstr+'.nc'))
        os.system(f"ln -fs {os.path.join(path, 'iced.'+tstr+mstr+'.nc')} {os.path.join(run_dir, 'cice', 'iced.'+tstr+'.nc')}")

        ##parallel process the copying
        shell_cmd = ""
        n = 0 ##index in nproc commands
        for file1, file2 in zip(copy_file_src, copy_file_dst):
            offset = task_id * task_nproc + n
            shell_cmd += f"{job_submit_cmd} 1 {offset} cp -fL {file1} {file2} & "
            n += 1
            if n == task_nproc:  ##wait for all nproc commands to finish, before next batch
                n = 0
                shell_cmd += "wait; "
        shell_cmd += "wait; "  ##wait for remaining commands
        os.system(shell_cmd)

        log_file = os.path.join(run_dir, "run.log")
        os.system("touch "+log_file)

    def postprocess(self, task_id=0, **kwargs):
        ##TODO: run fixhycom_cice here
        pass

    def run(self, task_id=0, **kwargs):
        self.run_status = 'running'

        job_submit_cmd = kwargs['job_submit_cmd']
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']

        input_file = self.filename(**kwargs)
        run_dir = os.path.dirname(input_file)
        os.system("mkdir -p "+run_dir)
        os.chdir(run_dir)

        model_exe = os.path.join(self.model_code_dir, 'hycom_cice')
        offset = task_id*task_nproc

        ##build the shell command line
        shell_cmd =  "source "+self.model_env_src+"; "  ##enter topaz5 env
        shell_cmd += "cd "+run_dir+"; "                 ##enter run directory
        shell_cmd += job_submit_cmd+f" {self.nproc} {offset} "+model_exe+" >& run.log"

        self.run_process = subprocess.Popen(shell_cmd, shell=True)
        self.run_process.wait()

