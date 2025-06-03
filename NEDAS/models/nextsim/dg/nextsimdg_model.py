import os
import subprocess
import numpy as np
from pyproj import Proj
from NEDAS.grid import Grid
from NEDAS.utils.conversion import units_convert, t2s, dt1h
from NEDAS.utils.shell_utils import run_job, makedir
from NEDAS.utils.netcdf_lib import nc_read_var, nc_write_var
from NEDAS.models import Model
from . import restart, forcing, namelist, dgLimit

class NextsimDGModel(Model):
    restart_dt: float
    dg_comp: int

    proj: str
    xstart: float
    xend: float
    ystart: float
    yend: float
    dx: float
    mask_file: str

    files: dict
    perturb: dict

    model_env: str
    model_config_file: str
    nproc_per_run: int
    walltime: int
    parallel_mode: str
    run_separate_jobs: bool
    use_job_array: bool

    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        self.native_variables = {
             'seaice_conc_dg': {'name':'data/cice', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':np.arange(self.dg_comp), 'units':1},
             'seaice_thick_dg': {'name':'data/hice', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':np.arange(self.dg_comp), 'units':'m'},
             'seaice_damage': {'name':'data/damage', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1},
             'snow_thick': {'name':'data/hsnow', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m'},
             'seaice_temp_k': {'name':'data/tice', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0,1,2], 'units':'K'},
             'seaice_velocity': {'name':('data/u', 'data/v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'m/s'},
             }
        self.diag_variables = {
             'seaice_conc': {'name':'data/sic', 'operator':self.get_seaice_conc, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':1},
             'seaice_thick': {'name':'data/sit', 'operator':self.get_seaice_thick, 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m'},
             }
        self.atmos_forcing_variables = {
            'atmos_surf_velocity': {'name':('data/u', 'data/v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'m/s'},
            'atmos_surf_press': {'name':'data/pair', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
            'atmos_surf_temp': {'name':'data/tair', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'C'},
            'atmos_surf_dewpoint': {'name':'data/dew2m', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'C'},
            'atmos_down_shortwave': {'name':'data/sw_in', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'W/m2'},
            'atmos_down_longwave': {'name':'data/lw_in', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'W/m2'},
        }
        self.ocean_forcing_variables = {
            'ocean_surf_velocity': {'name':('data/u', 'data/v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'m/s'},
            'ocean_surf_temp': {'name':'data/sst', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
            'ocean_surf_salinity': {'name':'data/sss', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
            'ocean_mixl_depth': {'name':'data/mld', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
        }
        self.variables = {**self.native_variables, **self.diag_variables, **self.atmos_forcing_variables, **self.ocean_forcing_variables}

        self.z_units = 'm'

        # construct grid obj based on config
        self.grid = Grid.regular_grid(Proj(self.proj), self.xstart, self.xend, self.ystart, self.yend, self.dx)

        # mask for grid points not
        self.grid.mask = self.read_mask()

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        member = kwargs['member']
        if member is None:
            ens_mem_dir = ''
            ens_mem_id = 1
        else:
            ens_mem_id = member + 1
            ens_mem_dir = f'ens_{str(ens_mem_id).zfill(2)}'

        time = kwargs['time']
        assert time is not None, 'nextsim.dg.filename: time needs to be specified, wildcard searching is not implemented.'

        name = kwargs['name']  ##name of the variable
        if name in self.native_variables:
            fname = restart.get_restart_filename(self.files['restart'], ens_mem_id, time)

        elif name in self.diag_variables:
            fname = f"{name}_k{kwargs['k']}_{t2s(kwargs['time'])}.npy"

        elif name in self.atmos_forcing_variables:
            fname = forcing.get_forcing_filename(self.files['forcing']['atmosphere'], ens_mem_id, time)

        elif name in self.ocean_forcing_variables:
            fname = forcing.get_forcing_filename(self.files['forcing']['ocean'], ens_mem_id, time)

        else:
            raise ValueError(f"variable {name} is not defined for nextsimdg model.")

        ##fname is given by the format defined in config file
        ##the source file is copied to path (cycle directory) in preprocess
        ##this filename function will return the copy, not the original location defined by fname
        return os.path.join(kwargs['path'], ens_mem_dir, os.path.basename(fname))

    def read_grid(self, **kwargs):
        ## the nextsim dg grid is fixed, no need to update the grid at runtime
        pass

    def read_mask(self):
        ## the nextsim dg grid is fixed, no need to update the mask at runtime
        if os.path.exists(self.mask_file):
            mask = nc_read_var(self.mask_file, 'data/mask')[:]
            return (mask==0)
        return np.full(self.grid.x.shape, False)

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        if name in self.native_variables:
            if rec['is_vector']:
                u = nc_read_var(fname, rec['name'][0])
                v = nc_read_var(fname, rec['name'][1])
                var = np.array([u, v])
            else:
                if rec['name'] in ['data/cice', 'data/hice', 'data/damage']:
                    tmp = nc_read_var(fname, rec['name'])
                    if len(tmp.shape) == 3:
                        var = tmp[..., kwargs['k']]
                    else:
                        var = tmp
                elif rec['name'] in ['data/tice']:
                    var = nc_read_var(fname, rec['name'])[kwargs['k'], ...]
                else:
                    var = nc_read_var(fname, rec['name'])

        elif name in self.diag_variables:
            var = rec['operator'](**kwargs)
            np.save(fname, var)

        else:
            if name in self.atmos_forcing_variables:
                forcing_name = 'atmosphere'
            elif name in self.ocean_forcing_variables:
                forcing_name = 'ocean'
            else:
                raise KeyError(f"variable {name} is not a forcing variable")
            file_options = self.files['forcing'][forcing_name]
            itime = forcing.get_time_index(fname,
                                           file_options['time_name'],
                                           file_options['time_units_name'],
                                           kwargs['time']
                                           )
            if rec['is_vector']:
                u = forcing.read_var(fname, [rec['name'][0],], itime)[0,...].data
                v = forcing.read_var(fname, [rec['name'][1],], itime)[0,...].data
                var = np.array([u, v])
            else:
                var = forcing.read_var(fname, [rec['name'],], itime)[0,...].data

        var = units_convert(rec['units'], kwargs['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        var = units_convert(kwargs['units'], rec['units'], var)

        if name in self.native_variables:
            if rec['is_vector']:
                dims = {'ydim':self.grid.ny, 'xdim':self.grid.nx}
                for i in range(2):
                    nc_write_var(fname, dims, rec['name'][i], var[i,...], comm=kwargs['comm'])
            else:
                if rec['name'] in ['data/cice', 'data/hice']:
                    dims = {'ydim':self.grid.ny, 'xdim':self.grid.nx, 'dg_comp':None}
                    recno = {'dg_comp':kwargs['k']}
                    nc_write_var(fname, dims, rec['name'], var, recno=recno, comm=kwargs['comm'])
                elif rec['name'] in ['data/tice']:
                    dims = {'zdim':None, 'ydim':self.grid.ny, 'xdim':self.grid.nx}
                    recno = {'zdim':kwargs['k']}
                    nc_write_var(fname, dims, rec['name'], var, recno=recno, comm=kwargs['comm'])
                else:
                    dims = {'ydim':self.grid.ny, 'xdim':self.grid.nx}
                    nc_write_var(fname, dims, rec['name'], var, comm=kwargs['comm'])

        elif name in self.diag_variables:
            np.save(fname, var)

        else:
            if name in self.atmos_forcing_variables:
                forcing_name = 'atmosphere'
            elif name in self.ocean_forcing_variables:
                forcing_name = 'ocean'
            else:
                raise KeyError(f"variable {name} is not a forcing variable")
            file_options = self.files['forcing'][forcing_name]
            itime = forcing.get_time_index(fname,
                                           file_options['time_name'],
                                           file_options['time_units_name'],
                                           kwargs['time']
                                           )
            if rec['is_vector']:
                forcing.write_var(fname, list(rec['name']), np.ma.array(var), itime)
            else:
                forcing.write_var(fname, [rec['name'],], np.ma.array([var,]), itime)

    def z_coords(self, **kwargs):
        return np.zeros(self.grid.x.shape)

    def get_seaice_conc(self, **kwargs):
        return self.read_var(**{**kwargs, 'name':'seaice_conc_dg', 'k':0, 'units':1})

    def get_seaice_thick(self, **kwargs):
        return self.read_var(**{**kwargs, 'name':'seaice_thick_dg', 'k':0, 'units':'m'})

    def preprocess(self, task_id:int=0, **kwargs):
        """Preprocessing method for nextsim.dg
        Parameters
        ----------
        task_id : int
            task id for parallel execution
        self.nproc_per_util : int
            number of processors for each task
        **kwargs : dict
            keyword arguments for the model configuration
            Keywords defined when the function is called:
            - member : int
                ensemble member id
            - time : datetime
                start time of the forecast
            - time_start: datetime
                initial time of the forecast cycles
            - path : str
                path to the working directory of the ensemble member
            - forecast_period : int
                number of hours being forecasted by model
            - restart_dir : str
                the saved restart directory from the previous cycle,
                which is the model run directory from
                the previous cycle. In the initial cycle, this
                directory is given as `ens_init_dir` defined in
                `nextsim.dg`` section of the `model_def` section
                in nedas config file.
        These are defined in the `config_file` entry of model_def:nextsim.dg section,
        and parse_config will bring them in this class as
            - self.files : dict
                This section contains the filenames for the restart file.
                This must have a `restart` key that defines the filename `format`
                and the strftime `time_format` code of the restart filename.
                If perturbation is used, this section must also have a `lon_name`
                and `lat_name` key
            - self.perturb : dict
                perturbation options for the initial conditions.
                See example configuration file for required keys and explanations.
                This section is not necessary if the model does not use perturbation
                for the initial conditions or forcings.
        """
        kwargs = super().parse_kwargs(**kwargs)

        # get the current ensemble member id
        ens_mem_id:int = kwargs['member'] + 1
        # ensemble member directory for the current member
        ens_mem_dir:str = f'ens_{str(ens_mem_id).zfill(2)}'

        # directory where files are being collected to, and where the model will be run
        run_dir = os.path.join(kwargs['path'], ens_mem_dir)
        makedir(run_dir)

        # get all required filenames for the initial ensemble
        # 1. get current and next time
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h
        time_start = kwargs['time_start']
        debug = kwargs['debug']

        # 2. get the restart and forcing filename
        file_options_restart = self.files['restart']
        # obtain restart file at initial cycling
        fname_restart_init:str = restart.get_restart_filename(file_options_restart, ens_mem_id, time)
        fname_restart:str = os.path.join(kwargs['restart_dir'], ens_mem_dir, os.path.basename(fname_restart_init))
        if not os.path.exists(fname_restart):
            fname_restart = fname_restart_init

        file_options_forcing:dict[str, dict] = self.files['forcing']
        fname_forcing:dict[str, str] = dict()
        for forcing_name in file_options_forcing:
            fname_forcing[forcing_name] = forcing.get_forcing_filename(file_options_forcing[forcing_name], ens_mem_id, time)

        # no need for perturbation if not specified in yaml file
        if not hasattr(self, 'perturb'):
            print('We do no perturbations as perturb section '
                  'is not specified in the model configuration.',
                  flush=True)
            # we we do not perturb the restart file
            # simply link the restart files
            os.system(f'ln -fs {fname_restart} {run_dir}')
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -fs {fname_forcing[forcing_name]} {run_dir}')
            return

        # 3. add perturbations
        np.random.seed(time.second+ens_mem_id)
        # here, if 'restart section is not under perturb section
        # we only link the restart file to each ensemble directory
        if 'restart' not in self.perturb or kwargs['time'] != kwargs['time_start']:
            # we we do not perturb the restart file
            # simply copy over the restart files
            os.system(f'cp -L {fname_restart} {run_dir}')
        else:
            restart_options = self.perturb['restart']
            # copy restart files to the ensemble member directory
            fname = os.path.join(run_dir, os.path.basename(fname_restart))
            subprocess.run(['cp', fname_restart, fname])
            # prepare the restart file options for the perturbation
            file_options_rst = {'fname': fname,
                            'lon_name':file_options_restart['lon_name'],
                            'lat_name':file_options_restart['lat_name']}
            # perturb the restart file
            restart.perturb_restart(restart_options, file_options_rst, debug)

        if 'forcing' not in self.perturb:
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -fs {fname_forcing[forcing_name]} {run_dir}')
        else:
            forcing_options = self.perturb['forcing']
            for forcing_name in forcing_options:
                # we ignore entries that are not in the files options
                # e.g., path
                if forcing_name not in fname_forcing: continue
                fname = os.path.join(run_dir,
                                    os.path.basename(fname_forcing[forcing_name])
                                    )
                # the forcing file options for the perturbation
                file_options_forcing[forcing_name]['fname_src'] = fname_forcing[forcing_name]
                file_options_forcing[forcing_name]['fname'] = fname
            # add forcing perturbations
            forcing.perturb_forcing(forcing_options, file_options_forcing, ens_mem_id, time, next_time, debug)

    def postprocess(self, task_id=0, **kwargs):
        """Postprocessing method for nextsim.dg
        Parameters: same as preprocess
        """
        kwargs = super().parse_kwargs(**kwargs)

        # get the current ensemble member id
        ens_mem_id:int = kwargs['member'] + 1
        # ensemble member directory for the current member
        ens_mem_dir:str = f'ens_{str(ens_mem_id).zfill(2)}'

        # directory where files are being collected to, and where the model will be run
        run_dir = os.path.join(kwargs['path'], ens_mem_dir)

        file = restart.get_restart_filename(self.files['restart'], ens_mem_id, kwargs['time'])
        restartfile = os.path.join(run_dir, os.path.basename(file))

        # read cice hice from restart file
        cice = nc_read_var(restartfile, 'data/cice')
        hice = nc_read_var(restartfile, 'data/hice')

        ##limit the cice between 0-1, and hice>0, before running the next forecast
        cice = dgLimit.limit_max(cice, 1.0)
        cice = dgLimit.limit_min(cice, 0.0)
        hice = dgLimit.limit_min(hice, 0.0)

        # write back to restart file
        dims = {'ydim':self.grid.ny, 'xdim':self.grid.nx, 'dg_comp':self.dg_comp}
        nc_write_var(restartfile, dims, 'data/cice', cice, comm=kwargs['comm'])
        nc_write_var(restartfile, dims, 'data/hice', hice, comm=kwargs['comm'])

    def run(self, task_id=0, **kwargs):
        """Run nextsim.dg model forecast"""
        kwargs = super().parse_kwargs(**kwargs)

        nproc = self.nproc_per_run
        offset = task_id*self.nproc_per_run

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        member = kwargs['member']
        if member is not None:
            run_dir:str = os.path.join(kwargs['path'], f"ens_{member+1:02}")
        else:
            run_dir = kwargs['path']
        makedir(run_dir)
        namelist.make_namelist(self.files, self.model_config_file, run_dir, **kwargs)

        ##build shell commands for running the model
        shell_cmd = ""
        if self.model_env:
            shell_cmd += f". {self.model_env}; "
        shell_cmd += f"cd {run_dir}; "
        shell_cmd += "JOB_EXECUTE $NDG_BLD_DIR/nextsim --config-file nextsim.cfg > time.step"

        run_job(shell_cmd, job_name='nextsim.dg.run', parallel_mode=self.parallel_mode, nproc=nproc, offset=offset, run_dir=run_dir, **kwargs)

        ##check if the restart file at next_time is produced
        fname_restart = restart.get_restart_filename(self.files['restart'], 1, next_time)
        fname_out = os.path.join(run_dir, os.path.basename(fname_restart))
        if not os.path.exists(fname_out):
            raise RuntimeError(f"nextsim.dg.run: failed to produce {fname_out}, check {run_dir}")

    def run_batch(self, task_id=0, **kwargs):
        """Run nextsim.dg model ensemble forecast, use job array to spawn the member runs"""
        kwargs = super().parse_kwargs(**kwargs)
        assert self.use_job_array, \
            "use_job_array shall be True if running ensemble in batch mode."

        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        run_dir:str = kwargs['path']
        nens:int = kwargs['nens']
        for member in range(nens):
            ens_dir = os.path.join(run_dir, f"ens_{member+1:02}")
            makedir(ens_dir)

            kwargs['member'] = member
            ##this creates run_dir/ens_??/nextsim.cfg
            namelist.make_namelist(self.files, self.model_config_file,
                                   ens_dir, **kwargs)

        ##build shell commands for running the model using job array
        shell_cmd = "echo starting the script...; "
        shell_cmd += f". {self.model_env}; "
        shell_cmd += f"cd {run_dir}; "
        shell_cmd += f"echo {run_dir}; "
        shell_cmd += "cd ens_$(printf '%02d' JOB_ARRAY_INDEX); "
        shell_cmd += "echo $(pwd); "
        shell_cmd += "cp $NDG_BLD_DIR/nextsim .; "
        if self.parallel_mode == 'openmp':
            shell_cmd += f"export OMP_NUM_THREADS={self.nproc_per_run}; "
            shell_cmd += "./nextsim --config-file nextsim.cfg > time.step"
        elif self.parallel_mode == 'mpi':
            shell_cmd += "JOB_EXECUTE ./nextsim --config-file nextsim.cfg > time.step"
        else:
            raise TypeError(f"unknown parallel mode '{self.parallel_mode}'")

        run_job(shell_cmd, job_name='nextsim.dg.ens_run', use_job_array=self.use_job_array, nproc=self.nproc_per_run, walltime=self.walltime, array_size=nens, run_dir=run_dir, **kwargs)

        ##check if the restart files at next_time are produced
        fname_restart = restart.get_restart_filename(self.files['restart'], 1, next_time)
        for member in range(nens):
            ens_dir = os.path.join(run_dir, f"ens_{member+1:02}")
            fname_out = os.path.join(ens_dir, os.path.basename(fname_restart))
            if not os.path.exists(fname_out):
                raise RuntimeError(f"nextsim.dg.run_batch: failed to produce {fname_out}, check {ens_dir}")
