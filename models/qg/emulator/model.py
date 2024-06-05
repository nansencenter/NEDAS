import numpy as np
import os
import sys
import inspect
import time
from pyproj import Proj

from config import parse_config
from grid import Grid
from utils.conversion import t2s, dt1h
from utils.netcdf_lib import nc_write_var

from ..util import read_data_bin, write_data_bin, grid2spec, spec2grid
from ..util import psi2zeta, psi2u, psi2v, psi2temp, uv2zeta, zeta2psi, temp2psi

from .netutils import Att_Res_UNet

class Model(object):
    """
    Class for configuring and running the qg model emulator
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
        restart_dt = self.total_counts * self.dt * 240
        self.restart_dt = restart_dt

        ##NB: total_counts = 200 used in training the model, so the emulator needs to run self.total_counts/200 iterations
        self.nsteps = self.total_counts // 200

        self.variables = {
            'velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            'streamfunc': {'name':'psi', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            'vorticity': {'name':'zeta', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            'temperature': {'name':'temp', 'dtype':'float', 'is_vector':False, 'restart_dt':restart_dt, 'levels':levels, 'units':'*'},
            }

        self.uniq_grid_key = ()
        self.uniq_z_key = ('k')
        self.z_units = '*'

        self.unet_model = Att_Res_UNet(**self.model_params).make_unet_model()
        self.unet_model.load_weights(self.weightfile)
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
        return self.grid


    def write_grid(self, grid, **kwargs):
        pass


    def read_mask(self, **kwargs):
        return self.mask


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


    def run(self, nens=1, task_id=0, task_nproc=1, **kwargs):
        assert task_nproc==1, f'qg emulator only support serial runs (got task_nproc={task_nproc})'

        self.run_status = 'running'

        if nens>1:
            ##running ensmeble together, ignore kwargs['member']
            members = list(range(nens))
        else:
            members = [kwargs['member']]

        time = kwargs['time']
        next_time = time + self.restart_dt * dt1h

        state = np.zeros((nens,self.ny,self.nx,self.nz))
        for m in range(nens):
            kwargs_in = {**kwargs, 'member':members[m], 'time':time}
            fname = self.filename(**kwargs_in)
            run_dir = os.path.dirname(fname)
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            for k in range(self.nz):
                state[m,...,k] = self.read_var(name='streamfunc', k=k, **kwargs_in)
                # psik = read_data_bin(input_file, self.kmax, self.nz, k)
                # state[m,...,k] = spec2grid(psik).T

        for i in range(self.nsteps):
            state = self.unet_model.predict(state, verbose=0)
        state_out = state

        for m in range(nens):
            kwargs_out = {**kwargs, 'member':members[m], 'time':next_time}
            fname = self.filename(**kwargs_out)
            if not os.path.exists(fname):
                with open(fname, 'wb'):
                    pass
            for k in range(self.nz):
                self.write_var(state_out[m,...,k], name='streamfunc', k=k, **kwargs_out)
                # psik = grid2spec(predict_state[0,...,k].T)
                # write_data_bin(output_file, psik, self.kmax, self.nz, k)

        # nc_write_var(output_file.replace('.bin','.nc'), {'t':None, 'z':self.nz, 'y':self.ny, 'x':self.nx}, 'psi', predict_state[0,...].transpose((2,0,1)), recno={'t':0})


    def kill(self):
        pass

