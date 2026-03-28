import os
import numpy as np
from NEDAS.utils.conversion import dt1h
from NEDAS.grid import Grid
from NEDAS.core import Model
from NEDAS.core.types import VarDesc
from .namelist import namelist
from .util import read_data_bin, write_data_bin, grid2spec, spec2grid
from .util import psi2zeta, psi2u, psi2v, psi2temp, uv2zeta, zeta2psi, temp2psi

class QGFortranModel(Model):
    """
    Class for configuring and running the qg model
    """
    kmax: int
    nz: int
    restart_dt: float
    nproc_per_run: int
    psi_init_type: str
    model_code_dir: str
    model_env: str
    spinup_hours: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.io_mode == 'offline', f"{self.__class__.__name__} only support offline io mode"

        self.dx = kwargs['dx'] if 'dx' in kwargs else 1.0
        n = 2*(self.kmax+1)
        self.ny, self.nx = n, n
        x, y = np.meshgrid(np.arange(n), np.arange(n))
        self.grid = Grid(None, x, y, cyclic_dim='xy')
        self.grid.mask = np.full(self.grid.x.shape, False)  # no grid points are masked

        self.dz = kwargs['dz'] if 'dz' in kwargs else 1.0
        levels = np.arange(0, self.nz, self.dz)

        self.variables = {
            'velocity': VarDesc(name=('u', 'v'), dtype='float', is_vector=True, dt=self.restart_dt, levels=levels, units=1, z_units=1),
            'streamfunc': VarDesc(name='psi', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels, units=1, z_units=1),
            'vorticity': VarDesc(name='zeta', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels, units=1, z_units=1),
            'temperature': VarDesc(name='temp', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels, units=1, z_units=1),
        }

        assert self.nproc_per_run==1, f'qg model only support serial runs (got task_nproc={self.nproc_per_run})'

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        if kwargs['member'] is not None:
            mstr = '{:04d}'.format(kwargs['member']+1)
        else:
            mstr = ''

        assert kwargs['time'] is not None, 'missing time in kwargs'
        tstr = kwargs['time'].strftime('%Y%m%d_%H')

        return os.path.join(kwargs['path'], mstr, 'output_'+tstr+'.bin')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self, **kwargs):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)

        k = kwargs['k']
        k1 = int(k)
        if k1 < self.nz-1:
            k2 = k1+1
        else:
            k2 = k1

        psik1 = read_data_bin(fname, self.kmax, self.nz, k1)
        psik2 = read_data_bin(fname, self.kmax, self.nz, k2)

        if kwargs['name'] == 'streamfunc':
            var1 = spec2grid(psik1).T
            var2 = spec2grid(psik2).T

        elif kwargs['name'] == 'velocity':
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

        elif kwargs['name'] == 'vorticity':
            zetak1 = psi2zeta(psik1)
            var1 = spec2grid(zetak1).T
            zetak2 = psi2zeta(psik2)
            var2 = spec2grid(zetak2).T

        elif kwargs['name'] == 'temperature':
            tempk1 = psi2temp(psik1)
            var1 = spec2grid(tempk1).T
            tempk2 = psi2temp(psik2)
            var2 = spec2grid(tempk2).T

        else:
            raise ValueError('unknown variable name '+kwargs['name'])

        # vertical interp between var1 and var2
        if k1 < self.nz-1:
            return (var1*(k2-k) + var2*(k-k1)) / (k2-k1)
        else:
            return var1

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)

        k = kwargs['k']
        if k==int(k):
            if kwargs['name'] == 'streamfunc':
                psik = grid2spec(var.T)

            elif kwargs['name'] == 'velocity':
                uk = grid2spec(var[0,...].T)
                vk = grid2spec(var[1,...].T)
                psik = zeta2psi(uv2zeta(uk, vk))

            elif kwargs['name'] == 'vorticity':
                zetak = grid2spec(var.T)
                psik = zeta2psi(zetak)

            elif kwargs['name'] == 'temperature':
                tempk = grid2spec(var.T)
                psik = temp2psi(tempk)

            else:
                raise ValueError('unknown variable name '+kwargs['name'])

            write_data_bin(fname, psik, self.kmax, self.nz, int(k))

    def z_coords(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        z = np.full(self.grid.x.shape, kwargs['k'])
        return z

    def preprocess(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        restart_dir = kwargs['restart_dir']
        restart_file = self.filename(**{**kwargs, 'path':restart_dir})

        # restart file to be used in this experiment, in work_dir/cycle/...
        input_file = self.filename(**kwargs)
        input_dir = os.path.dirname(input_file)

        # just cp the prepared files to the work_dir location
        self.c.fs.make_dir(input_dir)
        self.c.fs.copy_file(restart_file, input_file)

    def postprocess(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        task_id = kwargs.get('worker_id', 0)

        self.run_status = 'running'

        input_file = self.filename(**kwargs)
        run_dir = os.path.dirname(input_file)
        self.c.fs.make_dir(run_dir)

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        output_file = self.filename(**{**kwargs, 'time':next_time})

        if time >= self.c.config.time_start:
            #this is during cycling
            psi_init_type = 'read'
            prep_input_cmd = 'ln -fs '+input_file+' input.bin; '
        else:
            # this is initial run for spin up
            psi_init_type = self.psi_init_type
            prep_input_cmd = ''

        qg_exe = os.path.join(self.model_code_dir, 'src', 'qg.exe')

        # build the shell command line
        shell_cmd = ". "+self.model_env+"; "   # enter the qg model env
        shell_cmd += "cd "+run_dir+"; "         # enter the run dir
        shell_cmd += "rm -f restart.nml; "      # clean up before run
        shell_cmd += prep_input_cmd             # prepare input file
        shell_cmd += "JOB_EXECUTE "+qg_exe+" . " # the qg model exe
        shell_cmd += "> run.log 2>&1"               # output to log

        log_file = os.path.join(run_dir, 'run.log')

        # give it several tries, each time decreasing time step
        for dt_ratio in [1, 0.6, 0.2]:
            namelist(vars(self), time, forecast_period, psi_init_type, kwargs['member'], dt_ratio, run_dir)

            self.c.run_job(shell_cmd, offset=task_id*self.nproc_per_run, **kwargs)

            # check output
            with open(log_file, 'rt') as f:
                if 'Calculation done' in f.read():
                    break

        # check output
        with open(log_file, 'rt') as f:
            if 'Calculation done' not in f.read():
                raise RuntimeError('errors in '+log_file)
        if not os.path.exists(os.path.join(run_dir, 'output.bin')):
            raise RuntimeError('output.bin file not found')

        shell_cmd = "cd "+run_dir+"; "
        shell_cmd += "mv output.bin "+output_file
        self.c.run_job(shell_cmd, offset=task_id*self.nproc_per_run, **kwargs)

    def generate_truth(self, *args, **kwargs) -> None:
        assert self.truth_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        kwargs['member'] = None
        debug = kwargs.get('debug', False)
        self.c.fs.make_dir(self.truth_dir)

        # check if truth files already exists in model.truth_dir
        complete = True
        kwargs['time'] = self.c.config.time_start
        ntask = 0
        while kwargs['time'] < self.c.config.time_end:
            current_file = f"output_{kwargs['time']:%Y%m%d_%H}.bin"
            if not os.path.exists(os.path.join(self.truth_dir, current_file)):
                complete = False
                break
            kwargs['time'] += kwargs['forecast_period'] * dt1h
            ntask += 1
        if complete:
            self.c.debug_message = f"truth files already exist in {self.truth_dir}, skipping"
            return

        # create the truth files
        self.c.debug_message = f"Creating truth run for qg model in {self.truth_dir}"
        run_dir = os.path.join(self.truth_dir, 'run')
        init_file = f"output_{self.c.config.time_start:%Y%m%d_%H}.bin"
        self.c.debug_message = f"Running the model for spinup period to get initial condition: {init_file}"
        kwargs['time'] = self.c.config.time_start - self.spinup_hours * dt1h
        self.run(**{**kwargs, 'path':run_dir, 'member':0, 'forecast_period':self.spinup_hours})

        kwargs['time'] = self.c.config.time_start
        self.c.total_tasks = ntask
        self.c.current_task = 0
        while kwargs['time'] < self.c.config.time_end:
            current_file = f"output_{kwargs['time']:%Y%m%d_%H}.bin"
            next_time = kwargs['time'] + kwargs['forecast_period'] * dt1h
            next_file = f"output_{next_time:%Y%m%d_%H}.bin"
            self.c.debug_message = f"Running the model from condition {current_file} to reach {next_file}"
            self.run(**{**kwargs, 'path':run_dir, 'member':0})
            kwargs['time'] = next_time
            self.c.current_task += 1

        # clean up
        self.c.fs.move_files_to_dir(os.path.join(run_dir, '*', 'output*.bin'), self.truth_dir)
        self.c.debug_message = f"removing temporary run directory: {run_dir}"
        self.c.fs.remove_dir(run_dir)

    def generate_init_ensemble(self, *args, **kwargs) -> None:
        assert self.ens_init_dir is not None
        kwargs = super().parse_kwargs(kwargs)
        debug = kwargs.get('debug', False)
        basename = f"output_{kwargs['time']:%Y%m%d_%H}.bin"
        mstr = f"{kwargs['member']+1:04d}"
        init_file = os.path.join(self.ens_init_dir, mstr, basename)

        # check if restart file can be found in model.ens_init_dir already
        if os.path.exists(init_file):
            if debug:
                print(f"Init file {init_file} already exists, skipping")
            return

        # create the initial ensemble members
        if debug:
            print(f"Creating initial condition for qg modeli member {kwargs['member']+1}:")
        init_time = kwargs['time'] - self.spinup_hours * dt1h
        next_time = kwargs['time']
        run_dir = self.c.fs.forecast_dir(init_time, 'qg.fortran')
        if debug:
            print(f"initial condition type: {self.psi_init_type}")
            print(f"spinup period: {self.spinup_hours} hours")

        if debug:
            print(f"Spinning up member {kwargs['member']+1} from {init_time} to {next_time}")
        self.run(**{**kwargs, 'path':run_dir, 'time':init_time, 'forecast_period':self.spinup_hours})

        if debug:
            print("Moving output files")
        src_file = os.path.join(run_dir, mstr, basename)
        self.c.fs.make_dir(os.path.dirname(init_file))
        self.c.fs.move_file(src_file, init_file)
