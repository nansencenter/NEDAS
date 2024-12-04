import numpy as np
import os
import subprocess
from datetime import datetime
from time import sleep

from config import parse_config
from utils.conversion import units_convert, dt1h
from utils.shell_utils import run_command, run_job, makedir
from . import restart, forcing, namelist
from ...model_config import ModelConfig

class Model(ModelConfig):
    def __init__(self, config_file=None, parse_args=False, **kwargs):
        super().__init__(config_file, parse_args, **kwargs)

        level_sfc = np.array([0])
        self.variables = {  ##TODO: a list of model variables defined here
             'seaice_conc': {'name':'M_conc', 'dtype':'float', 'is_vector':False, 'dt':self.restart_dt, 'levels':[0], 'units':'%'},
             }
        self.z_units = 'm'

        self.run_process = None
        self.run_status = 'pending'

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
        """generate initial condition for a single ensemble member.

        Parameters
        ----------
        task_id : int
            task id for parallel execution
        task_nproc : int
            number of processors for each task
        **kwargs : dict
            keyword arguments for the model configuration
            Keywords defined when the function is called:
            - member : int
                ensemble member id
            - time_start : datetime
                start time of the forecast
            - path : str
                path to the working directory of the ensemble member
            These are defined in the configuration file of model_def/nextsim.dg section
            - files : dict
                This section contains the filenames for the restart file.
                This must have a `restart` key that defines the filename `format`
                and the strftime `time_format` code of the restart filename.
                If perturbation is used, this section must also have a `lon_name`
                and `lat_name` key
            - perturb : dict
                perturbation options for the initial conditions.
                See example configuration file for required keys and explanations.
                This section is not necessary if the model does not use perturbation
                for the initial conditions or forcings.

        """
        # get the current ensemble member id
        ens_mem_id:int = kwargs['member'] + 1
        # ensemble member directory for the current member
        ens_init_dir:str = os.path.join(kwargs['path'],
                                        f'ens_{str(ens_mem_id).zfill(2)}')
        # create directory for the initial ensemble
        os.makedirs(ens_init_dir, exist_ok=True)
        # get all required filenames for the initial ensemble
        # 1. get current time, which is supposed to be the start time
        time = kwargs['time_start']
        # 2. get the restart filename
        file_options_restart = kwargs['files']['restart']
        fname_restart:str = restart.get_restart_filename(file_options_restart, ens_mem_id, time)

        # no need for perturbation if not specified in yaml file
        if 'perturb' not in kwargs:
            print ('We do no perturbations as perturb section is not specified in the model configuration.')
            # we we do not perturb the restart file
            # simply link the restart files
            os.system(f'ln -s {fname_restart} {ens_init_dir}')
            return

        # get perturbation options
        perturb_options = kwargs['perturb']

        # here, if 'restart section is not under perturb section
        # we only link the restart file to each ensemble directory
        if 'restart' not in perturb_options:
            # we we do not perturb the restart file
            # simply link the restart files
            os.system(f'ln -s {fname_restart} {ens_init_dir}')
            return

        # 3. add perturbations
        restart_options = perturb_options['restart']
        # copy restart files to the ensemble member directory
        fname = os.path.join(ens_init_dir, os.path.basename(fname_restart))
        subprocess.run(['cp', '-v', fname_restart, fname])
        # prepare the restart file options for the perturbation
        file_options = {'fname': fname,
                        'lon_name':file_options_restart['lon_name'],
                        'lat_name':file_options_restart['lat_name']}
        # perturb the restart file
        restart.perturb_restart(restart_options, file_options)

    def prepare_forcing(self, task_id:int, **kwargs):
        """Prepare forcing file for the next forecast for a single ensemble member.

        Parameters
        ----------
        task_id : int
            task id for parallel execution
        **kwargs : dict
            keyword arguments for the model configuration
            Keywords defined when the function is called:
            - member : int
                ensemble member id
            - time : datetime
                start time of the forecast
            - next_time : datetime
                end time of the forecast
            - path : str
                path to the working directory of the ensemble member
            These are defined in the configuration file of model_def/nextsim.dg section
            - files : dict
                This section contains the filenames for the forcing file.
                This must have a `forcing` key with each subsection the
                atmosphere/ocean. Within these subsections, the filename `format`
                and the strftime `time_format` code of the restart filename.
                If perturbation is used, this section must also have a `lon_name`
                and `lat_name` key
            - perturb : dict
                perturbation options for the initial conditions.
                See example configuration file for required keys and explanations.
                This section is not necessary if the model does not use perturbation
                for the initial conditions or forcings.

        """
        # get the current ensemble member id
        ens_mem_id:int = kwargs['member'] + 1
        # ensemble member directory for the current member
        ens_init_dir:str = os.path.join(kwargs['path'],
                                        f'ens_{str(ens_mem_id).zfill(2)}')
        # create directory for the initial ensemble
        os.makedirs(ens_init_dir, exist_ok=True)
        # get all required filenames for the initial ensemble
        # 1. get current time and the end time of the forecast
        time:datetime = kwargs['time']
        next_time:datetime = kwargs['next_time']
        # 2. get the forcing filenames
        file_options_forcing:dict[str, str] = kwargs['files']['forcing']
        fname_forcing:dict[str, str] = dict()
        for forcing_name in file_options_forcing:
            fname_forcing[forcing_name] = forcing.get_forcing_filename(file_options_forcing[forcing_name],
                                                         ens_mem_id, time)

        if 'perturb' not in kwargs:
            print ('We do no perturbations as perturb section is not specified in the model configuration.',
                   flush=True)
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -s {fname_forcing[forcing_name]} {ens_init_dir}')
            return

        # get perturbation options
        perturb_options = kwargs['perturb']

        if 'forcing' not in perturb_options:
            # we we do not perturb the forcing file
            # simply link the forcing files
            for forcing_name in file_options_forcing:
                os.system(f'ln -s {fname_forcing[forcing_name]} {ens_init_dir}')
            return

        forcing_options = perturb_options['forcing']
        # construct file options for forcing
        file_options:dict = dict()
        for forcing_name in forcing_options:
            # we ignore entries that are not in the files options
            # e.g., path
            if forcing_name not in fname_forcing: continue
            fname = os.path.join(ens_init_dir,
                                    os.path.basename(fname_forcing[forcing_name])
                                    )
            # the forcing file options for the perturbation
            file_options[forcing_name] = {'fname_src': fname_forcing[forcing_name],
                                           'fname': fname,
                                           **file_options_forcing[forcing_name]}
        # add forcing perturbations
        forcing.perturb_forcing(forcing_options, file_options, ens_mem_id, time, next_time)

    def run(self, task_id=0, **kwargs):
        ##TODO: add the logic to run only 1 member here
        pass

    def run_batch(self, task_id=0, **kwargs):
        kwargs = super().parse_kwargs(**kwargs)
        self.run_status = 'running'
        
        run_dir:str = kwargs['path']
        nens:int = kwargs['nens']

        for member in range(nens):
            ens_dir = os.path.join(run_dir, f"ens_{member:02}")
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
        shell_cmd += "./nextsim --config-file nextsim.cfg > time.step"

        run_job(shell_cmd, nproc=self.nproc_per_run, ppn=32, array_size=nens, run_dir=run_dir, **kwargs)

