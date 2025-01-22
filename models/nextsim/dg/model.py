import numpy as np
import os
import subprocess
from pyproj import Proj
from grid import Grid
from utils.conversion import units_convert, dt1h
from utils.shell_utils import run_job, makedir
from . import restart, forcing, namelist
from ...model_config import ModelConfig

class Model(ModelConfig):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        level_sfc = np.array([0])
        self.seaice_variables = {  ##TODO: check units here and any missing variables
             'seaice_conc': {'name':'cice', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'%'},
             'seaice_thick': {'name':'hice', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m'},
             'snow_thick': {'name':'hsnow', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'m'},
             'seaice_temp': {'name':'tice', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'K'},
             'seaice_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'km/day'},
             }
        self.atmos_forcing_variables = {
            'atmos_surf_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'m/s'},
            'atmos_surf_press': {'name':'pair', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
            'atmos_surf_temp': {'name':'tair', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'K'},
            'atmos_surf_dewpoint': {'name':'dew2m', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'K'},
            'atmos_down_shortwave': {'name':'sw_in', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'W/m2'},
            'atmos_down_longwave': {'name':'lw_in', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'W/m2'},
        }
        self.ocean_forcing_variables = {
            'ocean_surf_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'dt':self.restart_dt, 'levels':[0], 'units':'m/s'},
            'ocean_surf_temp': {'name':'sst', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
            'ocean_surf_salinity': {'name':'sss', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
            'ocean_mixl_depth': {'name':'mld', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'Pa'},
        }
        self.variables = {**self.seaice_variables, **self.atmos_forcing_variables, **self.ocean_forcing_variables}

        self.z_units = 'm'

        # construct grid obj based on config
        self.grid = Grid.regular_grid(
            Proj(self.grid_def['proj']),
            float(self.grid_def['xstart']),
            float(self.grid_def['xend']),
            float(self.grid_def['ystart']),
            float(self.grid_def['yend']),
            float(self.grid_def['dx']))
        self.mask = np.full(self.grid.x.shape, False)  ##model grid points that are masked (land?)

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)

        member = kwargs['member']
        if member is None:
            ens_mem_dir = ''
            ens_mem_id = 1
        else:
            ens_mem_id = member + 1
            ens_mem_dir = os.path.join(kwargs['path'], f'ens_{str(ens_mem_id).zfill(2)}')

        time = kwargs['time']
        assert time is not None, 'nextsim.dg.filename: time needs to be specified, wildcard searching is not implemented.'

        name = kwargs['name']  ##name of the variable
        if name in self.seaice_variables:
            fname = restart.get_restart_filename(self.files['restart'], ens_mem_id, time)

        elif name in self.atmos_forcing_variables:
            fname = forcing.get_forcing_filename(self.files['forcing']['atmosphere'], ens_mem_id, time)

        elif name in self.ocean_forcing_variables:
            fname = forcing.get_forcing_filename(self.files['forcing']['ocean'], ens_mem_id, time)
        
        ##fname is given by the format defined in config file
        ##the source file is copied to path (cycle directory) in preprocess
        ##this filename function will return the copy, not the original location defined by fname
        return os.path.join(kwargs['path'], ens_mem_dir, os.path.basename(fname))
            
    def read_grid(self, **kwargs):
        ## the nextsim dg grid is fixed, no need to update the grid at runtime
        pass

    def read_mask(self):
        ## the nextsim dg grid is fixed, no need to update the mask at runtime
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        if name in self.seaice_variables:
            if rec['is_vector']:
                u = restart.read_var(fname, [rec[name][0],])[0,...].data
                v = restart.read_var(fname, [rec[name][1],])[0,...].data
                var = np.array([u, v])
            else:
                var = restart.read_var(fname, [rec[name],])[0,...].data

        else:
            if name in self.atmos_forcing_variables:
                forcing_name = 'atmosphere'
            elif name in self.ocean_forcing_variables:
                forcing_name = 'ocean'
            file_options = self.files['forcing'][forcing_name]
            itime = forcing.get_time_index(fname,
                                           file_options['time_name'],
                                           file_options['time_units_name'],
                                           kwargs['time']
                                           )
            if rec['is_vector']:
                u = forcing.read_var(fname, [rec[name][0],], itime)[0,...].data
                v = forcing.read_var(fname, [rec[name][1],], itime)[0,...].data
                var = np.array([u, v])
            else:
                var = forcing.read_var(fname, [rec[name],], itime)[0,...].data
    
        var = units_convert(kwargs['units'], rec['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        if name in self.seaice_variables:
            if rec['is_vector']:
                restart.write_var(fname, list(rec[name]), np.ma.arary(var))
            else:
                restart.write_var(fname, [rec[name],], np.ma.array([var,]))
        else:
            if name in self.atmos_forcing_variables:
                forcing_name = 'atmosphere'
            elif name in self.ocean_forcing_variables:
                forcing_name = 'ocean'
            file_options = self.files['forcing'][forcing_name]
            itime = forcing.get_time_index(fname,
                                           file_options['time_name'],
                                           file_options['time_units_name'],
                                           kwargs['time']
                                           )
            if rec['is_vector']:
                forcing.write_var(fname, list(rec[name]), np.ma.array(var), itime)
            else:
                forcing.write_var(fname, [rec[name],], np.ma.array([var,]))

    def z_coords(self, **kwargs):
        return np.zeros(self.grid.x.shape)

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
            os.system(f'ln -s {fname_restart} {run_dir}')
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -s {fname_forcing[forcing_name]} {run_dir}')
            return

        # 3. add perturbations
        # here, if 'restart section is not under perturb section
        # we only link the restart file to each ensemble directory
        if 'restart' not in self.perturb or kwargs['time'] != kwargs['time_start']:
            # we we do not perturb the restart file
            # simply link the restart files
            os.system(f'ln -s {fname_restart} {run_dir}')
        else:
            restart_options = self.perturb['restart']
            # copy restart files to the ensemble member directory
            fname = os.path.join(run_dir, os.path.basename(fname_restart))
            subprocess.run(['cp', '-v', fname_restart, fname])
            # prepare the restart file options for the perturbation
            file_options_rst = {'fname': fname,
                            'lon_name':file_options_restart['lon_name'],
                            'lat_name':file_options_restart['lat_name']}
            # perturb the restart file
            restart.perturb_restart(restart_options, file_options_rst)

        if 'forcing' not in self.perturb:
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -s {fname_forcing[forcing_name]} {run_dir}')
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
            forcing.perturb_forcing(forcing_options, file_options_forcing, ens_mem_id, time, next_time)

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
        shell_cmd = "echo starting the script...; "
        shell_cmd += f"source {self.model_env}; "
        shell_cmd += f"cd {run_dir}; "
        shell_cmd += f"echo {run_dir}; "
        shell_cmd += "cp $NDG_BLD_DIR/nextsim .; "
        if self.parallel_mode == 'openmp':
            shell_cmd += f"export OMP_NUM_THREADS={nproc}; "
            shell_cmd += "./nextsim --config-file nextsim.cfg > time.step"
        elif self.parallel_mode == 'mpi':
            shell_cmd += "JOB_EXECUTE ./nextsim --config-file nextsim.cfg > time.step"
        else:
            raise TypeError(f"unknown parallel mode '{self.parallel_mode}'")

        run_job(shell_cmd, job_name='nextsim.dg.run', nproc=nproc, offset=offset, run_dir=run_dir, **kwargs)

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
        shell_cmd += f"source {self.model_env}; "
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

        run_job(shell_cmd, job_name='nextsim.dg.ens_run', use_job_array=self.use_job_array, nproc=self.nproc_per_run, array_size=nens, run_dir=run_dir, **kwargs)

        ##check if the restart files at next_time are produced
        fname_restart = restart.get_restart_filename(self.files['restart'], 1, next_time)
        for member in range(nens):
            ens_dir = os.path.join(run_dir, f"ens_{member+1:02}")
            fname_out = os.path.join(ens_dir, os.path.basename(fname_restart))
            if not os.path.exists(fname_out):
                raise RuntimeError(f"nextsim.dg.run_batch: failed to produce {fname_out}, check {ens_dir}")
