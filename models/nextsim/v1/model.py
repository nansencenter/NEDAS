import numpy as np
import os
import glob
import inspect
import signal
import subprocess
from datetime import datetime

from config import parse_config
from utils.conversion import t2s, s2t, dt1h, units_convert
from grid import Grid

from .gmshlib import read_mshfile, proj
from .bin_io import read_data, write_data
from .namelist import namelist
from . import diag, atmos_forcing

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
        self.native_variables = {
            'seaice_conc': {'name':'M_conc', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'%' },
            'seaice_thick': {'name':'M_thick', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m' },
            'seaice_velocity': {'name':'M_VT', 'dtype':'float', 'is_vector':True, 'levels':[0], 'units':'m/s' },
            'seaice_damage': {'name':'M_damage', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'%' },
            'snow_thick': {'name':'M_snow_thick', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m' },
            }
        self.diag_variables = diag.variables
        self.atmos_forcing_variables = atmos_forcing.variables

        self.variables = {**self.native_variables, **self.diag_variables, **self.atmos_forcing_variables}

        ##default grid and mask (before getting set by read_grid)
        self.read_grid_from_mshfile(os.path.join(os.environ['NEXTSIM_MESH_DIR'], self.msh_filename))

        self.mask = np.full(self.grid.x.shape, False)  ##no grid points are masked

        self.grid_bank = {}

        self.perturb_history = {}

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        if 'name' not in kwargs:
            name = list(self.native_variables.keys())[0]   ##if not specified, return the first variable name
        else:
            name = kwargs['name']

        if name in {**self.native_variables, **self.diag_variables}:

            if 'path' in kwargs:
                path = kwargs['path']
            else:
                path = '.'

            if 'member' in kwargs and kwargs['member'] is not None:
                mstr = '{:03d}'.format(kwargs['member']+1)
            else:
                mstr = ''

            if 'time' in kwargs and kwargs['time'] is not None:
                assert isinstance(kwargs['time'], datetime), 'Error: time is not a datetime object'
                tstr = kwargs['time'].strftime('%Y%m%dT%H%M%SZ')
            else:
                tstr = '*'
            # search = os.path.join(path, mstr, self.restart_input_path, 'field_'+tstr+'.bin')
            # flist = glob.glob(search)
            # assert len(flist)>0, 'no matching files found: '+search
            # return flist[0]
            return os.path.join(path, mstr, self.restart_input_path, 'field_'+tstr+'.bin')

        elif name in self.atmos_forcing_variables:
            return atmos_forcing.filename(**kwargs)

        else:
            raise ValueError(f"variable name '{name}' is not defined for nextsim.v1 model!")

    def read_grid_from_mshfile(self, mshfile):
        grid_info = read_mshfile(mshfile)
        x, y = grid_info['nodes_x'], grid_info['nodes_y']
        triangles = np.array([np.array(el.node_indices) for el in grid_info['triangles']])
        self.grid = Grid(proj, x, y, regular=False, triangles=triangles)
        self.edges = np.array([np.array(el.node_indices) for el in grid_info['edges']])

    def read_grid(self, **kwargs):
        name = kwargs['name']
        if name in {**self.native_variables, **self.diag_variables}:
            meshfile = self.filename(**kwargs).replace('field', 'mesh')
            ###only need to read the uniq grid once, store known meshfile in memory bank
            if meshfile not in self.grid_bank:
                ##read the grid from mesh file and add to grid_bank
                x = read_data(meshfile, 'Nodes_x')
                y = read_data(meshfile, 'Nodes_y')
                elements = read_data(meshfile, 'Elements')
                n_elements = int(elements.size/3)
                triangles = elements.reshape((n_elements, 3)) - 1
                self.grid_bank[meshfile] = Grid(proj, x, y, regular=False, triangles=triangles)
            self.grid = self.grid_bank[meshfile]

        elif name in self.atmos_forcing_variables:
            self.grid = atmos_forcing.grid

    def write_grid(self, **kwargs):
        """
        write updated mesh back to mesh file

        Note: now we assume that number of mesh elements and their indices doesn't change!
        only updating the mesh node position x,y
        """
        name = kwargs['name']
        if name in self.native_variables:
            meshfile = self.filename(**kwargs).replace('field', 'mesh')

            write_data(meshfile, 'Nodes_x', self.grid.x)
            write_data(meshfile, 'Nodes_y', self.grid.y)

            elements = (self.grid.triangles + 1).flatten()
            write_data(meshfile, 'Elements', elements)

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        """read variable from a model restart file"""
        fname = self.filename(**kwargs)
        name = kwargs['name']

        if name in self.native_variables:
            var = read_data(fname, self.variables[name]['name'])

            ##nextsim restart file concatenates u,v component, so reshape if is_vector
            if self.native_variables[name]['is_vector']:
                var = var.reshape((2, -1))

        elif name in self.diag_variables:
            var = diag.variables[name]['getter'](**kwargs)

        elif name in self.atmos_forcing_variables:
            var = atmos_forcing.read_var(**kwargs)

        ##convert units if native unit is not the same as required by kwargs
        if 'units' in kwargs and 'units' in self.variables[name]:
            units = kwargs['units']
        else:
            units = self.variables[name]['units']
        var = units_convert(units, self.variables[name]['units'], var)
        return var

    def write_var(self, var, **kwargs):
        """write variable back to a model restart file"""
        fname = self.filename(**kwargs)
        name = kwargs['name']

        ##convert units back if necessary
        var = units_convert(kwargs['units'], self.variables[name]['units'], var, inverse=True)

        if name in self.native_variables:
            ##nextsim restart file concatenates u,v component, so flatten if is_vector
            if kwargs['is_vector']:
                var = var.flatten()

            ##check if original var is on mesh nodes or elements
            # var_orig = read_data(fname, variables[name]['name']).flatten()
            # if var_orig.size != var.size:
            #     ##the grid.convert interpolate to nodes by default, if size mismatch, this means
            #     ##we need element values, take the average of the node values here
            #     var = np.nanmean(var[grid.tri.triangles], axis=1)

            ##output the var to restart file
            write_data(fname, self.variables[name]['name'], var)

        elif name in self.atmos_forcing_variables:
            atmos_forcing.write_var(var, **kwargs)

    def postproc(self, var, **kwargs):
        name = kwargs['name']
        if name == 'seaice_conc':
            ##set values outside physical range back
            var[np.where(var<0)] = 0.0
            var[np.where(var>1)] = 1.0

        if name == 'seaice_thick':
            ##set values outside physical range back
            var[np.where(var<0)] = 0.0

        if name == 'seaice_damage':
            ##set values outside physical range back
            var[np.where(var<0)] = 0.0
            var[np.where(var>1)] = 1.0
        return var

    def z_coords(self, **kwargs):
        ##for nextsim, just discard inputs and simply return zero as z_coords
        return np.zeros(self.grid.x.shape)

    def read_param(self, **kwargs):
        return getattr(self, kwargs['name'])

    def write_param(self, param, **kwargs):
        setattr(self, kwargs['name'], param)

    def generate_initial_condition(self, task_id=0, task_nproc=1, **kwargs):
        ##put sequence of operation here to generate the initial condition files for nextsim
        ens_init_dir = kwargs['ens_init_dir']
        time_start = kwargs['time_start']
        time_end = kwargs['time_end']
        job_submit_cmd = kwargs['job_submit_cmd']

        ##prepare initial condition
        ##restart files are in ens_init_dir, prepared beforehand
        init_file = self.filename(**{**kwargs, 'path':ens_init_dir, 'time':time_start})

        ##where restart files are stored for the model run
        restart_file = self.filename(**{**kwargs, 'time':time_start})
        restart_dir = os.path.dirname(restart_file)
        os.system("mkdir -p "+restart_dir)

        ##copy the files over
        field_bin = init_file
        field_dat = field_bin.replace('.bin', '.dat')
        mesh_bin = init_file.replace('field', 'mesh')
        mesh_dat = mesh_bin.replace('.bin', '.dat')
        for file in [field_bin, field_dat, mesh_bin, mesh_dat]:
            os.system("cp "+file+" "+restart_dir)

        ##prepare boundary forcing
        ##where forcing files are stored in ens_init_dir
        forcing_dir = os.path.join(os.path.dirname(os.path.dirname(init_file)), 'data', 'GENERIC_PS_ATM')
        os.system("rm -rf "+forcing_dir+"; mkdir -p "+forcing_dir)

        ##original generic_ps_atm forcing in model_data_dir, prepared beforehand
        forcing_dir_orig = os.path.join(self.model_data_dir, 'GENERIC_PS_ATM')

        ##make copy of original files, do batch processing (task_nproc commands in background)
        # shell_cmd = ""
        # n = 0  ##index in nproc commands
        # t = time_start
        # while t < time_end:
        #     forcing_file_orig = atmos_forcing.filename(**{**kwargs, 'path':forcing_dir_orig, 'time':t})
        #     offset = task_id * task_nproc + n
        #     shell_cmd += f"{job_submit_cmd} 1 {offset} cp -L {forcing_file_orig} {forcing_dir}/. & "
        #     n += 1
        #     if n == task_nproc:  ##wait for all nproc commands to finish, before next batch
        #         n = 0
        #         shell_cmd += "wait; "
        #     t += 24*dt1h  ##forcing files are stored daily
        # shell_cmd += "wait; "  ##wait for remaining commands
        # os.system(shell_cmd)

    def run(self, task_id=0, task_nproc=16, **kwargs):
        self.run_status = 'running'

        job_submit_cmd = kwargs['job_submit_cmd']
        model_src = os.path.join(self.model_code_dir, 'setup.src')
        model_exe = os.path.join(self.model_code_dir, 'model', 'bin', 'nextsim.exec')
        offset = task_id*task_nproc

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        prev_time = time - forecast_period * dt1h
        input_file = self.filename(**kwargs)
        output_file = self.filename(**{**kwargs, 'time':next_time})

        ##check run_dir and input files
        restart_file = self.filename(**kwargs)
        run_dir = os.path.dirname(os.path.dirname(restart_file))
        os.system("mkdir -p "+run_dir)
        field_bin = input_file
        field_dat = field_bin.replace('.bin', '.dat')
        mesh_bin = input_file.replace('field', 'mesh')
        mesh_dat = mesh_bin.replace('.bin', '.dat')
        for file in [field_bin, field_dat, mesh_bin, mesh_dat]:
            if not os.path.exists(file):
                raise RuntimeError("input file is missing: "+file)

        ##add perturbations
        # ens_init_dir = kwargs['ens_init_dir']
        # init_file = self.filename(**{**kwargs, 'path':ens_init_dir})
        # forcing_dir = os.path.join(os.path.dirname(os.path.dirname(init_file)), 'data', 'GENERIC_PS_ATM')
        # if 'perturb' in kwargs:
            # forcing.perturb_var(**{**kwargs, 'path':forcing_dir, 'time':time})
        self.perturb(**kwargs)

        ##link input data files
        shell_cmd = "cd "+run_dir+"; "
        shell_cmd += "rm -rf data; mkdir data; cd data; "
        shell_cmd += "ln -fs "+os.path.join(self.model_data_dir, 'BATHYMETRY', '*')+" .; "
        shell_cmd += "ln -fs "+os.path.join(self.model_data_dir, 'TOPAZ4', 'TP4DAILY_*')+" .; "
        shell_cmd += "ln -fs "+forcing_dir+" .; "
        os.system(shell_cmd)

        ##build command to run the model
        shell_cmd = "source "+model_src+"; "
        shell_cmd += "cd "+run_dir+"; "
        shell_cmd += "export NEXTSIM_DATA_DIR="+os.path.join(run_dir,'data')+"; "
        shell_cmd += job_submit_cmd+f" {task_nproc} {offset} "
        shell_cmd += model_exe+" --config-files=nextsim.cfg "
        shell_cmd += ">& run.log"
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

        ##make a copy of output file to the output_dir
        if 'output_dir' in kwargs:
            output_dir = kwargs['output_dir']
            if output_dir != kwargs['path']:
                output_file_cp = self.filename(**{**kwargs, 'path':output_dir, 'time':next_time})
                os.system("mkdir -p "+os.path.dirname(output_file_cp))
                field_bin = output_file
                field_dat = field_bin.replace('.bin', '.dat')
                mesh_bin = output_file.replace('field', 'mesh')
                mesh_dat = mesh_bin.replace('.bin', '.dat')
                for file in [field_bin, field_dat, mesh_bin, mesh_dat]:
                    os.system("cp "+file+" "+os.path.dirname(output_file_cp)+"/.")

