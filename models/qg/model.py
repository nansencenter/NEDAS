import numpy as np
import os
import inspect
import signal
import subprocess
from pyproj import Proj

from config import parse_config
from grid import Grid
from utils.conversion import t2s, dt1h

from .namelist import namelist
from .util import read_data_bin, write_data_bin, grid2spec, spec2grid
from .util import psi2zeta, psi2u, psi2v, psi2temp, uv2zeta, zeta2psi, temp2psi

class QGModel(object):
    """
    Class for configuring and running the qg model
    """

    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.dx = kwargs['dx'] if 'dx' in kwargs else 1.0
        n = 2*(self.kmax+1)
        self.ny, self.nx = n, n
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        self.grid = Grid(Proj('+proj=stere'), x, y, cyclic_dim='xy')
        self.mask = np.full(self.grid.x.shape, False)  ##no grid points are masked

        self.dz = kwargs['dz'] if 'dz' in kwargs else 1.0
        levels = np.arange(0, self.nz, self.dz)

        ##the model is nondimensionalized, but it's convenient to introduce
        ##a scaling for time control in cycling experiments:
        ##0.05 time units ~ 12 hours
        ##dt = 0.00025 ~ 216 seconds

        self.variables = {
            'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'dt':12, 'levels':levels, 'units':'*'},
            'streamfunc': {'name':'psi', 'dtype':'float', 'is_vector':False, 'dt':12, 'levels':levels, 'units':'*'},
            'vorticity': {'name':'zeta', 'dtype':'float', 'is_vector':False, 'dt':12, 'levels':levels, 'units':'*'},
            'temperature': {'name':'temp', 'dtype':'float', 'is_vector':False, 'dt':12, 'levels':levels, 'units':'*'},
            }

        self.z_units = '*'

        ##
        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs):
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'

        if 'member' in kwargs and kwargs['member'] is not None:
            mstr = '{:04d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert 'time' in kwargs, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(path, mstr, 'output_'+tstr+'.bin')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        assert 'name' in kwargs, 'missing variable name in kwargs'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(**kwargs)

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 0  ##read the first layer by default
        assert k>=0 and k<self.nz, f'level index {k} is not within range 0-{self.nz}'

        k1 = int(k)
        if k1 < self.nz-1:
            k2 = k1+1
        else:
            k2 = k1

        psik1 = read_data_bin(fname, self.kmax, self.nz, k1)
        psik2 = read_data_bin(fname, self.kmax, self.nz, k2)

        if name == 'streamfunc':
            var1 = spec2grid(psik1).T
            var2 = spec2grid(psik2).T

        elif name == 'velocity':
            uk1 = psi2u(psik1)
            vk1 = psi2v(psik1)
            u1 = spec2grid(uk1).T
            v1 = spec2grid(vk1).T
            var1 = np.array([u1, v1])
            uk2 = psi2u(psik2)
            vk2 = psi2v(psik2)
            u2 = spec2grid(uk2).T
            v2 = spec2grid(vk2).T
            var2 = np.array([u2, v2])

        elif name == 'vorticity':
            zetak1 = psi2zeta(psik1)
            var1 = spec2grid(zetak1).T
            zetak2 = psi2zeta(psik2)
            var2 = spec2grid(zetak2).T

        elif name == 'temperature':
            tempk1 = psi2temp(psik1)
            var1 = spec2grid(tempk1).T
            tempk2 = psi2temp(psik2)
            var2 = spec2grid(tempk2).T

        ##vertical interp between var1 and var2
        if k1 < self.nz-1:
            return (var1*(k2-k) + var2*(k-k1)) / (k2-k1)
        else:
            return var1

    def write_var(self, var, **kwargs):
        ##check kwargs
        assert 'name' in kwargs, 'missing variable name in kwargs'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(**kwargs)

        if 'k' in kwargs:
            k = kwargs['k']
        else:
            k = 0  ##read the first layer by default
        assert k>=0 and k<self.nz, f'level index {k} is not within range 0-{self.nz}'

        if k==int(k):
            if name == 'streamfunc':
                psik = grid2spec(var.T)

            elif name == 'velocity':
                uk = grid2spec(var[0,...].T)
                vk = grid2spec(var[1,...].T)
                psik = zeta2psi(uv2zeta(uk, vk))

            elif name == 'vorticity':
                zetak = grid2spec(var.T)
                psik = zeta2psi(zetak)

            elif name == 'temperature':
                tempk = grid2spec(var.T)
                psik = temp2psi(tempk)

            write_data_bin(fname, psik, self.kmax, self.nz, int(k))

    def z_coords(self, **kwargs):
        assert 'k' in kwargs, 'qg.z_coords: missing k in kwargs'
        z = np.ones(self.grid.x.shape) * kwargs['k']
        return z

    def preprocess(self, task_id=0, task_nproc=1, **kwargs):
        time = kwargs['time']
        restart_dir = kwargs['restart_dir']

        ##restart files for each member in ens_init_dir, this is prepared beforehand
        restart_file = self.filename(**{**kwargs, 'path':restart_dir, 'time':time})

        ##restart file to be used in this experiment, in work_dir/cycle/...
        input_file = self.filename(**kwargs)
        input_dir = os.path.dirname(input_file)

        ##just cp the prepared files to the work_dir location
        subprocess.run("mkdir -p "+input_dir+"; cp "+restart_file+" "+input_file, shell=True)

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, task_nproc=1, **kwargs):
        assert task_nproc==1, f'qg model only support serial runs (got task_nproc={task_nproc})'
        self.run_status = 'running'

        job_submit_cmd = kwargs['job_submit_cmd']

        input_file = self.filename(**kwargs)
        run_dir = os.path.dirname(input_file)
        subprocess.run("mkdir -p "+run_dir, shell=True)

        path = kwargs['path']
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        kwargs_out = {**kwargs, 'time':next_time}
        output_file = self.filename(**kwargs_out)

        ##check status, skip if already run
        # if os.path.exists(output_file):
        #     return

        if self.psi_init_type == 'read':
            prep_input_cmd = 'ln -fs '+input_file+' input.bin; '
        else:
            prep_input_cmd = ''

        qg_exe = os.path.join(self.model_code_dir, 'src', 'qg.exe')

        offset = task_id*task_nproc
        submit_cmd = job_submit_cmd+f" {task_nproc} {offset} "

        ##build the shell command line
        shell_cmd = "source "+self.model_env+"; "   ##enter the qg model env
        shell_cmd += "cd "+run_dir+"; "         ##enter the run dir
        shell_cmd += "rm -f restart.nml; "      ##clean up before run
        shell_cmd += prep_input_cmd             ##prepare input file
        shell_cmd += submit_cmd                 ##job_submitter
        shell_cmd += qg_exe+" . "               ##the qg model exe
        shell_cmd += ">& run.log"               ##output to log
        # print(shell_cmd, flush=True)
        log_file = os.path.join(run_dir, 'run.log')

        ##give it several tries, each time decreasing time step
        for dt_ratio in [1, 0.6, 0.2]:
            namelist(self, forecast_period, dt_ratio, run_dir)

            self.run_process = subprocess.Popen(shell_cmd, shell=True)
            self.run_process.wait()

            ##check output
            with open(log_file, 'rt') as f:
                if 'Calculation done' in f.read():
                    break
            returncode = self.run_process.returncode
            if returncode is not None and returncode < 0:
                ##kill signal received, exit the run func
                return

        ##check output
        with open(log_file, 'rt') as f:
            if 'Calculation done' not in f.read():
                raise RuntimeError('errors in '+log_file)
        if not os.path.exists(os.path.join(run_dir, 'output.bin')):
            raise RuntimeError('output.bin file not found')

        shell_cmd = "cd "+run_dir+"; "
        shell_cmd += "mv output.bin "+output_file
        subprocess.run(shell_cmd, shell=True)

