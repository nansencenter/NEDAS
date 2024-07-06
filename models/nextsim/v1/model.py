import numpy as np
import os
import glob
import inspect
import signal
import subprocess
from datetime import datetime

from config import parse_config
from grid import Grid
from utils.conversion import t2s, s2t, dt1h, units_convert

from .gmshlib import read_mshfile, proj
from .bin_io import read_data, write_data
from .diag_var import get_diag_var
# from .forcing import
from .namelist import namelist

class Model(object):
    """
    Class for configuring and running the nextsim v1 model (lagrangian version)
    """

    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.restart_dt = 3

        ##Note: we only work with restart files, normal nextsim binfile have some variables names that
        ##are different from restart files, e.g. Concentration instead of M_conc
        self.variables = {'seaice_conc': {'name':'M_conc', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'%' },
            'seaice_thick': {'name':'M_thick', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m' },
            'seaice_damage': {'name':'M_damage', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'%' },
            'snow_thick': {'name':'M_snow_thick', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m' },
            'seaice_velocity': {'name':'M_VT', 'dtype':'float', 'is_vector':True, 'levels':[0], 'units':'m/s' },
            'seaice_drift': {'name':'', 'dtype':'float', 'is_vector':True, 'levels':[0], 'units':'km/day' },
            'seaice_shear': {'name':'', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'1/day' },
            }

        self.grid_bank = {}
        # self.grid
        # self.mask

        self.run_process = None
        self.run_status = 'pending'


    def filename(self, **kwargs):
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'
        if 'member' in kwargs and kwargs['member'] is not None:
            mstr = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        name = kwargs['name'] if 'name' in kwargs else list(self.variables.keys())[0]
        if 'time' in kwargs and kwargs['time'] is not None:
            assert isinstance(kwargs['time'], datetime), 'Error: time is not a datetime object'
            tstr = kwargs['time'].strftime('%Y%m%dT%H%M%SZ')
        else:
            tstr = '*'
        if 'dt' in kwargs:
            dt = kwargs['dt'] * timedelta(hours=1)
        else:
            dt = 0
        # search = os.path.join(path, mstr, self.restart_input_path, 'field_'+tstr+'.bin')
        # flist = glob.glob(search)
        # assert len(flist)>0, 'no matching files found: '+search
        # return flist[0]
        return os.path.join(path, mstr, self.restart_input_path, 'field_'+tstr+'.bin')


    def read_grid(self, **kwargs):
        meshfile = self.filename(**kwargs).replace('field', 'mesh')
        ###only need to read the uniq grid once, store known meshfile in memory bank
        if meshfile not in self.grid_bank:
            ##read the grid from mesh file
            x = read_data(meshfile, 'Nodes_x')
            y = read_data(meshfile, 'Nodes_y')
            elements = read_data(meshfile, 'Elements')
            ne = int(elements.size/3)
            triangles = elements.reshape((ne, 3)) - 1
            grid = Grid(proj, x, y, regular=False, triangles=triangles)
            ##add the grid to grid_bank
            self.grid_bank[meshfile] = grid

        self.grid = self.grid_bank[meshfile]


    def write_grid(self, grid, **kwargs):
        """
        write updated mesh back to mesh file

        Note: now we assume that number of mesh elements and their indices doesn't change!
        only updating the mesh node position x,y
        """
        meshfile = self.filename(**kwargs).replace('field', 'mesh')

        write_data(meshfile, 'Nodes_x', grid.x)
        write_data(meshfile, 'Nodes_y', grid.y)


    def read_grid_from_msh(self, mshfile):
        """
        get the grid object directly from .msh definition file
        this function is uniq to nextsim, not required by assim_tools
        """
        info = read_mshfile(mshfile)
        x = info['nodes_x']
        y = info['nodes_y']
        triangles = np.array([np.array(el.node_indices) for el in info['triangles']])
        self.grid = Grid(proj, x, y, regular=False, triangles=triangles)


    def read_var(self, **kwargs):
        """read native variable defined on native grid from model restart files"""
        ##check name in kwargs and read the variables from file
        vname = kwargs['name']
        assert vname in self.variables, 'variable name '+vname+' not listed in variables'
        fname = self.filename(**kwargs)

        ##get diagnostic variables from their own getters
        if vname in get_diag_var.keys():
            return get_diag_var[vname](**kwargs)

        var = read_data(fname, self.variables[vname]['name'])

        ##nextsim restart file concatenates u,v component, so reshape if is_vector
        if self.variables[vname]['is_vector']:
            var = var.reshape((2, -1))

        ##convert units if native unit is not the same as required by kwargs
        if 'units' in kwargs:
            units = kwargs['units']
        else:
            units = self.variables[vname]['units']
        var = units_convert(units, self.variables[vname]['units'], var)
        return var


    def write_var(self, var, **kwargs):
        """write native variable back to a model restart file"""
        fname = self.filename(**kwargs)

        ##check name in kwargs and read the variables from file
        assert 'name' in kwargs, 'please specify which variable to write, name=?'
        vname = kwargs['name']
        assert vname in self.variables, "variable name "+vname+" not listed in variables"

        ##ignore diagnostic variables
        if vname in ['seaice_drift', 'seaice_shear']:
            return

        ##nextsim restart file concatenates u,v component, so flatten if is_vector
        if kwargs['is_vector']:
            var = var.flatten()

        ##convert units back if necessary
        var = units_convert(kwargs['units'], self.variables[vname]['units'], var, inverse=True)

        ##check if original var is on mesh nodes or elements
        # var_orig = read_data(fname, variables[vname]['name']).flatten()
        # if var_orig.size != var.size:
        #     ##the grid.convert interpolate to nodes by default, if size mismatch, this means
        #     ##we need element values, take the average of the node values here
        #     var = np.nanmean(var[grid.tri.triangles], axis=1)

        ##output the var to restart file
        write_data(fname, self.variables[vname]['name'], var)


    def postproc(self, var, **kwargs):
        vname = kwargs['name']
        if vname == 'seaice_conc':
            ##set values outside physical range back
            var[np.where(var<0)] = 0.0
            var[np.where(var>1)] = 1.0

        if vname == 'seaice_thick':
            ##set values outside physical range back
            var[np.where(var<0)] = 0.0

        if vname == 'seaice_damage':
            ##set values outside physical range back
            var[np.where(var<0)] = 0.0
            var[np.where(var>1)] = 1.0
        return var


    def z_coords(self, **kwargs):
        ##for nextsim, just discard inputs and simply return zero as z_coords
        return np.zeros(self.grid.x.shape)


    def read_param(self, **kwargs):
        param = 0
        return param


    def write_param(self, param, **kwargs):
        pass


    def run(self, task_id=0, task_nproc=16, **kwargs):
        self.run_status = 'running'

        host = kwargs['host']
        nedas_dir = kwargs['nedas_dir']
        code_dir = kwargs['code_dir']
        data_dir = kwargs['data_dir']

        restart_file = self.filename(**kwargs)
        run_dir = os.path.dirname(os.path.dirname(restart_file))

        # print('running nextsim v1 model in '+run_dir, flush=True)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        input_file = self.filename(**kwargs)
        kwargs_out = {**kwargs, 'time':next_time}
        output_file = self.filename(**kwargs_out)

        ##check restart input files
        field_bin = input_file
        field_dat = field_bin.replace('.bin', '.dat')
        mesh_bin = input_file.replace('field', 'mesh')
        mesh_dat = mesh_bin.replace('.bin', '.dat')
        for file in [field_bin, field_dat, mesh_bin, mesh_dat]:
            if not os.path.exists(file):
                raise RuntimeError("input file is missing: "+file)

        ##prepare the input files
        ##restart files should be created by the cycling
        ##other data:
        shell_cmd = "cd "+run_dir+"; "
        shell_cmd += "rm -rf data; mkdir data; cd data; "
        shell_cmd += "ln -fs "+os.path.join(data_dir, 'BATHYMETRY', '*')+" .; "
        shell_cmd += "ln -fs "+os.path.join(data_dir, 'TOPAZ4', 'TP4DAILY_*')+" .; "
        shell_cmd += "ln -fs "+os.path.join(data_dir, 'GENERIC_PS_ATM')+" .; "
        subprocess.run(shell_cmd, shell=True)

        env_dir = os.path.join(nedas_dir, 'config', 'env', host)
        model_src = os.path.join(env_dir, 'nextsim.v1.src')
        model_exe = os.path.join(code_dir, 'nextsim', 'model', 'bin', 'nextsim.exec')

        offset = task_id*task_nproc
        submit_cmd = os.path.join(env_dir, 'job_submit.sh')+f" {task_nproc} {offset} "

        ##build the shell command line
        shell_cmd = "source "+model_src+"; "
        shell_cmd += "cd "+run_dir+"; "
        shell_cmd += "export NEXTSIM_DATA_DIR="+os.path.join(run_dir,'data')+"; "
        shell_cmd += submit_cmd
        shell_cmd += model_exe+" --config-files=nextsim.cfg "
        shell_cmd += ">& run.log"
        # print(shell_cmd, flush=True)
        log_file = os.path.join(run_dir, 'run.log')

        ##give it several tries, each time decreasing time step
        for dt_ratio in [1, 0.5]:

            self.timestep *= dt_ratio

            namelist(self, time, forecast_period, run_dir)

            self.run_process = subprocess.Popen(shell_cmd, shell=True)
            self.run_process.wait()

            ##check output
            with open(log_file, 'rt') as f:
                if 'Simulation done' in f.read():
                    break

            if self.run_process.returncode < 0:
                ##kill signal received, exit the run func
                return

        ##check output
        with open(log_file, 'rt') as f:
            if 'Simulation done' not in f.read():
                raise RuntimeError('errors in '+log_file)
        if not os.path.exists(output_file):
            raise RuntimeError(output_file+' not generated, run failed')

