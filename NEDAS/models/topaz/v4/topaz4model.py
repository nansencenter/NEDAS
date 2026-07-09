import numpy as np
import os
import subprocess
from functools import lru_cache

from NEDAS.utils.conversion import units_convert, dt1h
from NEDAS.grid import RegularGrid
from NEDAS.core import Model
from NEDAS.core.types import VarDesc
from ..abfile import ABFileRestart, ABFileBathy
from ..model_grid import get_topaz_grid
from .namelist import namelist

class Topaz4Model(Model[RegularGrid]):
    io_mode = 'offline'
    basedir: str
    R: str
    T: str
    E: str
    V: str
    X: str
    onem: float
    z_units: str
    restart_dt: int
    forcing_frc: str
    era5_path: str
    priver: int
    jerlv0: int
    relax: int
    nproc: int
    nproc_per_run: int
    walltime: int|None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        levels = np.arange(1, 51, 1)
        levels_sfc = np.array([0])
        self.variables = {
            'ocean_velocity': VarDesc(name=('u', 'v'), dtype='float', is_vector=True, dt=self.restart_dt, levels=levels, units='m/s', z_units=self.z_units),
            'ocean_layer_thick': VarDesc(name='dp', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels, units='Pa', z_units=self.z_units),
            'ocean_temp': VarDesc(name='temp', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels, units='K', z_units=self.z_units),
            'ocean_saln': VarDesc(name='saln', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels, units='psu', z_units=self.z_units),
            'ocean_surf_height': VarDesc(name='msshb', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels_sfc, units='m', z_units=self.z_units),
            'ocean_surf_temp': VarDesc(name='sstb', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels_sfc, units='K', z_units=self.z_units),
            'ocean_b_velocity':  VarDesc(name=('ubavg', 'vbavg'), dtype='float', is_vector=True, dt=self.restart_dt, levels=levels_sfc, units='m/s', z_units=self.z_units),
            'ocean_b_press': VarDesc(name='pbavg', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels_sfc, units='Pa', z_units=self.z_units),
            'ocean_mixl_depth': VarDesc(name='dpmixl', dtype='float', is_vector=False, dt=self.restart_dt, levels=levels_sfc, units='Pa', z_units=self.z_units),
        }

        # model grid
        grid_info_file = os.path.join(self.basedir, 'topo', 'grid.info')
        self.grid = get_topaz_grid(grid_info_file)

        self.depthfile = os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}.a')
        f = ABFileBathy(self.depthfile, 'r', idm=self.grid.nx, jdm=self.grid.ny)
        depth = f.read_field('depth')
        f.close()
        self.depth = -depth.data
        self.grid.mask = np.asarray(depth.mask, dtype=bool)

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)

        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        tstr = kwargs['time'].strftime('%Y_%j_%H_0000')
        return os.path.join(kwargs['path'], mstr[1:], 'TP4restart'+tstr+mstr+'.a')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name].asdict()

        f = ABFileRestart(fname, 'r', idm=self.grid.nx, jdm=self.grid.ny)
        if rec['is_vector']:
            var1 = f.read_field(rec[name]['name'][0], level=kwargs['k'], tlevel=1, mask=None)
            var2 = f.read_field(rec[name]['name'][1], level=kwargs['k'], tlevel=1, mask=None)
            var = np.array([var1, var2])
        else:
            var = f.read_field(rec[name]['name'], level=kwargs['k'], tlevel=1, mask=None)
        f.close()

        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name].asdict()

        # open the restart file for over-writing
        # the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile
        f = ABFileRestart(fname, 'r+', idm=self.grid.nx, jdm=self.grid.ny)

        # convert units back if necessary
        var = units_convert(kwargs['units'], rec['units'], var)

        if rec['is_vector']:
            for i in range(2):
                f.overwrite_field(var[i,...], None, rec['name'][i], level=kwargs['k'], tlevel=1)
        else:
            f.overwrite_field(var, None, rec['name'], level=kwargs['k'], tlevel=1)
        f.close()

    def z_coords(self, **kwargs) -> np.ndarray:
        """
        Calculate vertical coordinates given the 3D model state.
        Returns:
            np.ndarray: The corresponding z field.

        Public entry point -- accepts the flexible **kwargs calling
        convention used throughout the codebase (FieldRecord.asdict(),
        ObsRecord.asdict(), obs_seq entries merged in by core/obs.py, etc.),
        but the cached implementation only ever needs member/time/k/path to
        locate and read a field (name/units get overridden internally, see
        _z_coords_cached; everything else has a default -- see base Model's
        parse_kwargs). Extracting exactly those four here, into
        _z_coords_cached's own explicit (non-**kwargs) signature, means
        nothing else the caller's kwargs happens to contain -- an obs_seq
        entry's x/y/z coordinate arrays today, whatever gets added tomorrow
        -- ever reaches the lru_cache boundary at all: it's excluded by not
        being a parameter, not by being matched against a list of known-bad
        fields. Same fix as topaz/v5. 'path' specifically must be one of
        the four kept: see io_backends/offline.py's call_method, which
        resolves the correct forecast-cycle path (using model_src from the
        *original* kwargs) and injects it before calling z_coords -- drop
        it and parse_kwargs falls back to config.work_dir alone (missing
        the /cycle/<time>/<model> suffix), silently pointing at the wrong
        directory instead of raising.
        """
        return self._z_coords_cached(
            member=kwargs.get('member'),
            time=kwargs.get('time'),
            k=kwargs.get('k', 0),
            path=kwargs.get('path'),
        )

    @lru_cache(maxsize=3)
    def _z_coords_cached(self, member, time, k, path):
        # not checked for correctness yet
        z = np.zeros(self.grid.x.shape)

        if k == 0:
            return z
        else:
            rec = {'member': member, 'time': time, 'k': k, 'path': path}
            rec['name'] = 'ocean_layer_thick'
            rec['units'] = self.variables['ocean_layer_thick'].units
            if self.z_units == 'm':
                dz = - self.read_var(**rec) / self.onem
            elif self.z_units == 'Pa':
                dz = self.read_var(**rec)
            else:
                raise ValueError('do not know how to calculate z_coords for z_units = '+self.z_units)

            z_prev = self._z_coords_cached(member=member, time=time, k=k-1, path=path)
            return z_prev + dz

    def preprocess(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(kwargs)

        init_file = self.filename(**{**kwargs, 'path':self.ens_init_dir})
        input_file = self.filename(**kwargs)
        os.system("mkdir -p "+os.path.dirname(input_file))
        os.system("cp "+init_file+" "+input_file)
        os.system("cp "+init_file.replace('.a', '.b')+" "+input_file.replace('.a', '.b'))

    def postprocess(self, task_id=0, **kwargs):
        pass

    def run(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(kwargs)
        self.run_status = 'running'

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        input_file = self.filename(**kwargs)
        run_dir = os.path.dirname(input_file)
        os.system("mkdir -p "+run_dir)
        os.chdir(run_dir)
        log_file = os.path.join(run_dir, "run.log")
        os.system('touch '+log_file)

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        kwargs_out = {**kwargs, 'time':next_time}
        output_file = self.filename(**kwargs_out)

        # create namelist config files
        namelist(self, time, forecast_period, run_dir)

        # link files
        partit_file = os.path.join(self.basedir, 'topo', 'partit', f'depth_{self.R}_{self.T}.{self.nproc:04d}')
        os.system("cp "+partit_file+" patch.input")

        for ext in ['.a', '.b']:
            os.system("ln -fs "+os.path.join(self.basedir, 'topo', 'regional.grid'+ext)+" regional.grid"+ext)
            os.system("ln -fs "+os.path.join(self.basedir, 'topo', f'depth_{self.R}_{self.T}'+ext)+" regional.depth"+ext)
            os.system("ln -fs "+os.path.join(self.basedir, 'topo', 'tbaric'+ext)+" tbaric"+ext)
        os.system("ln -fs "+os.path.join(self.basedir, 'topo', 'grid.info')+" grid.info")

        # TODO: switches for other forcing options
        forcing_path = None
        if self.forcing_frc == 'era5':
            forcing_path = self.era5_path
        if self.forcing_frc == 'era40':
            pass
        assert forcing_path is not None
        os.system("ln -fs "+forcing_path+" .")
        os.system("ln -fs "+os.path.join(self.basedir, 'force', 'other', 'iwh_tabulated.dat')+" .")
        for ext in ['.a', '.b']:
            if self.priver == 1:
                os.system("ln -fs "+os.path.join(self.basedir, 'force', 'rivers', self.E, 'rivers'+ext)+" forcing.rivers"+ext)
            if self.jerlv0 == 0:
                os.system("ln -fs "+os.path.join(self.basedir, 'force', 'seawifs', 'kpar'+ext)+" forcing.kpar"+ext)
            if self.relax == 1:
                for comp in ['saln', 'temp', 'intf', 'rmu']:
                    os.system("ln -fs "+os.path.join(self.basedir, 'relax', self.E, 'relax_'+comp[:3]+ext)+" relax."+comp+ext)
            os.system("ln -fs "+os.path.join(self.basedir, 'relax', self.E, 'thkdf4'+ext)+" thkdf4"+ext)
        os.system("ln -fs "+os.path.join(self.basedir, 'relax', self.E, 'clim_tran.txt')+" .")
        # if self.gpflag: TODO
        # if self.nestoflag:
        # if self.nestiflag:
        # if self.tideflag:

        model_src = os.path.join(self.basedir, 'setup.src')
        model_exe = os.path.join(self.basedir, f'Build_V{self.V}_X{self.X}', 'hycom')

        # build the shell command line
        shell_cmd =  ". "+model_src+"; "   # enter topaz v4 env
        shell_cmd += "cd "+run_dir+"; "          # enter run directory
        shell_cmd += f"JOB_EXECUTE {model_exe} {kwargs['member']+1} >& run.log"

        for tr in range(2):  # number of tries
            with open(log_file, 'rt') as f:
                if '(normal)' in f.read():
                    break
            self.run_process = subprocess.Popen(shell_cmd, shell=True)
            self.run_process.wait()
            self.c.run_job(shell_cmd, job_name='topaz4_run', run_dir=run_dir,
                    nproc=self.nproc, offset=task_id*self.nproc_per_run,
                    walltime=self.walltime, **kwargs)

        with open(log_file, 'rt') as f:
            if '(normal)' not in f.read():
                raise RuntimeError('errors in '+log_file)
        if not os.path.exists(output_file):
            raise RuntimeError('output file not found: '+output_file)
