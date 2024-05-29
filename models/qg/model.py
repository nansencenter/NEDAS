import numpy as np
import os
import inspect
import signal
import time
import subprocess
from pyproj import Proj

from config import parse_config
from grid import Grid
from utils.conversion import t2s

from .namelist import namelist
from .util import read_data_bin, write_data_bin, grid2spec, spec2grid
from .util import psi2zeta, psi2u, psi2v, psi2temp, uv2zeta, zeta2psi, temp2psi

class Model(object):
    """
    Class for configuring and running the qg model
    """

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        """
        """

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        ##the model is nondimensionalized, but it's convenient to introduce a scaling
        ##for time control in cycling experiments
        self.dx = kwargs['dx'] if 'dx' in kwargs else 1.0
        n = 2*(self.kmax+1)
        self.ny, self.nx = n, n
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        self.grid = Grid(Proj('+proj=stere'), x, y, cyclic_dim='xy')

        self.dz = kwargs['dz'] if 'dz' in kwargs else 1.0
        levels = np.arange(0, self.nz, self.dz)

        restart_dt = 6
        self.restart_dt = restart_dt

        self.variables = {
            'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            'streamfunc': {'name':'psi', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            'vorticity': {'name':'zeta', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            'temperature': {'name':'temp', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            }

        self.uniq_grid_key = ()
        self.uniq_z_key = ('k')
        self.z_units = '*'

        ##
        self.run_process = None
        self.run_status = 'pending'


    def filename(self, path, **kwargs):
        if 'member' in kwargs and kwargs['member'] is not None:
            mstr = '{:04d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert 'time' in kwargs, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(path, mstr, 'output_'+tstr+'.bin')


    def read_grid(self, path, **kwargs):
        return self.grid


    def write_grid(self, path, grid, **kwargs):
        pass


    def read_mask(self, path, grid):
        mask = np.full(grid.x.shape, False)  ##no grid points are masked
        return mask


    def read_var(self, path, grid, **kwargs):
        assert 'name' in kwargs, 'missing variable name in kwargs'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(path, **kwargs)

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
        if k1 < nz-1:
            return (var1*(k2-k) + var2*(k-k1)) / (k2-k1)
        else:
            return var1


    def write_var(self, path, grid, var, **kwargs):
        ##check kwargs
        assert 'name' in kwargs, 'missing variable name in kwargs'
        name = kwargs['name']
        assert name in self.variables, 'variable name '+name+' not listed in variables'
        fname = self.filename(path, **kwargs)

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


    def z_coords(self, path, grid, **kwargs):
        assert 'k' in kwargs, 'qg.z_coords: missing k in kwargs'
        z = np.ones(grid.x.shape) * kwargs['k']
        return z


    def run(self, task_id, c, path, **kwargs):
        self.run_status = 'running'

        fname = self.filename(path, **kwargs)
        run_dir = os.path.dirname(fname)

        print('running qg model in '+run_dir, flush=True)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        ##check status, skip if already run

        namelist(self, run_dir)

        offset = task_id

        env_dir = os.path.join(c.nedas_dir, 'config', 'env', c.host)
        # submit_cmd = os.path.join(env_dir, 'job_submit.sh')+f" 1 1 {offset} "
        submit_cmd = ''
        qg_src = os.path.join(env_dir, 'qg.src')
        qg_exe = os.path.join(c.nedas_dir, 'models', 'qg', 'src', 'qg.exe')

        ##build the shell command line
        shell_cmd = "source "+qg_src+"; "   ##enter the qg model env
        shell_cmd += "cd "+run_dir+"; "         ##enter the run dir
        shell_cmd += "rm -f restart.nml *bin; " ##clean up before run
        shell_cmd += submit_cmd                 ##job_submitter
        shell_cmd += qg_exe+" . "               ##the qg model exe
        shell_cmd += ">& run.log"               ##output to log
        # print(shell_cmd, flush=True)

        self.run_process = subprocess.Popen(shell_cmd, shell=True, preexec_fn=os.setsid)

        ## Check the status of the process
        # while True:
        #     # Use poll() to check if the process has terminated
        #     status = self.run_process.poll()
        #     if status is not None:
        #         if status == 0:
        #             self.run_status = 'finished'
        #         elif status == -9:
        #             self.run_status = 'killed'
        #         elif status == -15:
        #             self.run_status = 'terminated'
        #         break
        #     time.sleep(1)
        self.run_process.wait()

        ##collect output
        ##check output valid here


    # def is_running(self):
    #     if self.run_status == 'running':
    #         return True
    #     return False


    def kill(self):
        os.killpg(os.getpgid(self.run_process.pid), signal.SIGKILL)

