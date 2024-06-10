import numpy as np
import os
import inspect
import signal
import subprocess

from config import parse_config
from grid import Grid
from utils.conversion import t2s, s2t, dt1h

from .namelist import namelist


class Model(object):

    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        self.variables = {
                'atmos_velocity'
                'atmos_temp'
                'atmos_pres'
                'atmos_q_vapor'
                }

        self.run_process = None
        self.run_status = 'pending'

    def filename(self, **kwargs)
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'

        tstr = kwargs['time'].strftime('%Y-%m-%d_%H:%M:%S')
        return os.path.join(path, 'wrfout_'+tstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def write_grid(self, grid, **kwargs):
        pass

    def read_var(self, **kwargs):
        pass

    def write_var(self, **kwargs):
        pass

    def z_coords(self, **kwargs):
        pass

    def run(self, task_id=0, task_nproc=1, **kwargs):

        self.run_status = 'running'
        host = kwargs['host']
        nedas_dir = kwargs['nedas_dir']
        code_dir = kwargs['code_dir']
        data_dir = kwargs['data_dir']

        fname = self.filename(**kwargs)
        run_dir = os.path.dirname(fname)
        print('running wrf model in '+run_dir, flush=True)

        env_dir = os.path.join(nedas_dir, 'config', 'env', host)
        wrf_src = os.path.join(env_dir, 'wrf.src')
        wrf_exe = os.path.join(code_dir, 'wrf', 'WRF', 'main', 'wrf.exe')

        ##build the run command
        shell_cmd = "source "+wrf_src+"; "   ##enter wrf env

        self.run_process = subprocess.Popen(shell_cmd, shell=True, preexec_fn=os.setsid)
        self.run_process.wait()

        returncode = self.run_process.returncode
        if returncode is not None and returncode < 0:
            ##kill signal received, exit the run
            return

    def kill(self):
        os.killpg(os.getpgid(self.run_process.pid), signal.SIGKILL)

