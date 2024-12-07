import numpy as np
import os
import subprocess

from utils.conversion import units_convert, dt1h
from utils.shell_utils import run_job, makedir
from . import restart, forcing, namelist
from ...model_config import ModelConfig

class Model(ModelConfig):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        level_sfc = np.array([0])
        self.variables = {  ##TODO: a list of model variables defined here
             'seaice_conc': {'name':'M_conc', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'%'},
             }
        # restart
        # forcing
        
        self.z_units = 'm'

        #self.grid construct grid obj based on config

    def filename(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        if kwargs['time'] is not None:
            tstr = kwargs['time'].strftime('%Y_%j_%H')
        else:
            tstr = '????????'
        if kwargs['member'] is not None:
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
        return os.path.join(kwargs['path'], mstr[1:], 'init_25km_NH_'+tstr+mstr+'.nc')

    def read_grid(self, **kwargs):
        pass

    def read_mask(self):
        pass

    def read_var(self, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##TODO: read model variable logic here, can come from restart.py

        var = units_convert(kwargs['units'], rec['units'], var)
        return var

    def write_var(self, var, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        fname = self.filename(**kwargs)
        name = kwargs['name']
        rec = self.variables[name]

        ##open the restart file for over-writing

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
            - path : str
                path to the working directory of the ensemble member
        These are defined in the configuration file model_def:nextsim.dg section,
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
        ens_mem_dir:str = os.path.join(kwargs['path'],
                                        f'ens_{str(ens_mem_id).zfill(2)}')
        # create directory for the ensemble member
        os.makedirs(ens_mem_dir, exist_ok=True)
        
        # get all required filenames for the initial ensemble
        # 1. get current and next time
        time = kwargs['time']
        forecast_period = kwargs['forecast_period']
        next_time = time + forecast_period * dt1h

        # 2. get the restart and forcing filename
        file_options_restart = self.files['restart']
        fname_restart:str = restart.get_restart_filename(file_options_restart, ens_mem_id, time)

        file_options_forcing:dict[str, str] = self.files['forcing']
        fname_forcing:dict[str, str] = dict()
        for forcing_name in file_options_forcing:
            fname_forcing[forcing_name] = forcing.get_forcing_filename(file_options_forcing[forcing_name], ens_mem_id, time)

        # no need for perturbation if not specified in yaml file
        if not hasattr(self, 'perturb'):
            print ('We do no perturbations as perturb section is not specified in the model configuration.', flush=True)
            # we we do not perturb the restart file
            # simply link the restart files
            os.system(f'ln -s {fname_restart} {ens_mem_dir}')
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -s {fname_forcing[forcing_name]} {ens_mem_dir}')
            return

        # here, if 'restart section is not under perturb section
        # we only link the restart file to each ensemble directory
        if 'restart' not in self.perturb:
            # we we do not perturb the restart file
            # simply link the restart files
            os.system(f'ln -s {fname_restart} {ens_mem_dir}')
            return
        if 'forcing' not in self.perturb:
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -s {fname_forcing[forcing_name]} {ens_mem_dir}')
            return

        # 3. add perturbations
        restart_options = self.perturb['restart']
        # copy restart files to the ensemble member directory
        fname = os.path.join(ens_mem_dir, os.path.basename(fname_restart))
        subprocess.run(['cp', '-v', fname_restart, fname])
        # prepare the restart file options for the perturbation
        file_options = {'fname': fname,
                        'lon_name':file_options_restart['lon_name'],
                        'lat_name':file_options_restart['lat_name']}
        # perturb the restart file
        restart.perturb_restart(restart_options, file_options)

        forcing_options = self.perturb['forcing']
        # construct file options for forcing
        file_options:dict = dict()
        for forcing_name in forcing_options:
            # we ignore entries that are not in the files options
            # e.g., path
            if forcing_name not in fname_forcing: continue
            fname = os.path.join(ens_mem_dir,
                                os.path.basename(fname_forcing[forcing_name])
                                )
            # the forcing file options for the perturbation
            file_options[forcing_name] = {'fname_src': fname_forcing[forcing_name],
                                           'fname': fname,
                                           **file_options_forcing[forcing_name]}
        # add forcing perturbations
        forcing.perturb_forcing(forcing_options, file_options, ens_mem_id, time, next_time)

    def run(self, task_id=0, **kwargs):
        """Run nextsim.dg model forecast"""
        kwargs = super().parse_kwargs(**kwargs)
        nproc = self.nproc_per_run
        offset = task_id*self.nproc_per_run

        member = kwargs['member']
        if member is not None:
            run_dir:str = os.path.join(kwargs['path'], f"ens_{member+1:02}")
        else:
            run_dir = kwargs['path']
        makedir(run_dir)
        namelist.make_namelist(run_dir, **kwargs)

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

    def run_batch(self, task_id=0, **kwargs):
        """Run nextsim.dg model ensemble forecast, use job array to spawn the member runs"""
        kwargs = super().parse_kwargs(**kwargs)
        assert kwargs['use_job_array'], "use_job_array shall be True if running ensemble in batch mode."

        run_dir:str = kwargs['path']
        nens:int = kwargs['nens']
        for member in range(nens):
            ens_dir = os.path.join(run_dir, f"ens_{member+1:02}")
            makedir(ens_dir)

            ##this creates run_dir/ens_??/nextsim.cfg
            namelist.make_namelist(ens_dir, member=member, **kwargs)

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

        run_job(shell_cmd, job_name='nextsim.dg.ens_run', nproc=self.nproc_per_run, array_size=nens, run_dir=run_dir, **kwargs)

