import numpy as np
import configparser
import os
import shutil
import subprocess
import inspect
from datetime import datetime
from functools import lru_cache
from time import sleep

from utils.conversion import units_convert, t2s, s2t, dt1h
from config import parse_config
from . import restart
from . import forcing

class Model(object):
    def __init__(self, config_file=None, parse_args=False, **kwargs):

        ##parse config file and obtain a list of attributes
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)
        for key, value in config_dict.items():
            setattr(self, key, value)

        levels = np.arange(1, 51, 1)
        level_sfc = np.array([0])
        self.variables = {
                'ocean_velocity': {'name':('u', 'v'), 'dtype':'float', 'is_vector':True, 'levels':levels, 'units':'m/s'},
                'ocean_layer_thick': {'name':'dp', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'Pa'},
                'ocean_temp': {'name':'temp', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'K'},
                'ocean_saln': {'name':'saln', 'dtype':'float', 'is_vector':False, 'levels':levels, 'units':'psu'},
                'ocean_surf_height': {'name':'msshb', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'m'},
                'ocean_surf_temp': {'name':'sstb', 'dtype':'float', 'is_vector':False, 'levels':[0], 'units':'K'},
                'ocean_b_velocity':  {'name':('ubavg', 'vbavg'), 'dtype':'float', 'is_vector':True, 'levels':level_sfc, 'units':'m/s'},
                'ocean_b_press': {'name':'pbavg', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'Pa'},
                'ocean_mixl_depth': {'name':'dpmixl', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'Pa'},
                'seaice_velocity': {'name':('uice', 'vice'), 'dtype':'float', 'is_vector':True, 'levels':level_sfc, 'units':'m/s'},
                'seaice_conc': {'name':'ficem', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'%'},
                'seaice_thick': {'name':'hicem', 'dtype':'float', 'is_vector':False, 'levels':level_sfc, 'units':'m'},
                }
        self.z_units = 'm'
#        self.read_grid()

        self.run_process = None
        self.run_status = 'pending'
        self.grid=''


    def filename(self, **kwargs):
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = '.'
        if 'time' in kwargs and kwargs['time'] is not None:
            assert isinstance(kwargs['time'], datetime), 'time shall be a datetime object'
            tstr = kwargs['time'].strftime('%Y_%j_%H')
        else:
            tstr = '????????'
        if 'member' in kwargs and kwargs['member'] is not None:
            assert kwargs['member'] >= 0, 'member index shall be >= 0'
            mstr = '_mem{:03d}'.format(kwargs['member']+1)
            mdir = '{:03d}'.format(kwargs['member']+1)
        else:
            mstr = ''
            mdir = ''
        return os.path.join(path, mdir, 'init_25km_NH_'+tstr+mstr+'.nc')


    def read_grid(self, **kwargs):
        pass


    def write_grid(self, **kwargs):
        pass


    def read_mask(self):
        pass


    def read_var(self, **kwargs):
        ##check name in kwargs and read the variables from file
        assert 'name' in kwargs, 'please specify which variable to get, name=?'
        name = kwargs['name']
        assert name in variables, 'variable name '+name+' not listed in variables'
        fname = filename(path, **kwargs)

        if 'k' in kwargs:
            ##Note: ocean level indices are negative in assim_tools.state
            ##      but in abfiles, they are defined as positive indices
            k = -kwargs['k']
        else:
            k = -variables[name]['levels'][-1]  ##get the first level if not specified
        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            mask = None

        if 'is_vector' in kwargs:
            is_vector = kwargs['is_vector']
        else:
            is_vector = variables[name]['is_vector']

        if 'units' in kwargs:
            units = kwargs['units']
        else:
            units = variables[name]['units']

        var = units_convert(units, variables[name]['units'], var)
        return var


    def write_var(path, grid, var, **kwargs):
        ##check name in kwargs
        assert 'name' in kwargs, 'please specify which variable to write, name=?'
        name = kwargs['name']
        assert name in variables, 'variable name '+name+' not listed in variables'
        fname = filename(path, **kwargs)

        ##same logic for setting level indices as in read_var()
        if 'k' in kwargs:
            k = -kwargs['k']
        else:
            k = -variables[name]['levels'][-1]
        if 'mask' in kwargs:
            mask = kwargs['mask']
        else:
            mask = None

        ##open the restart file for over-writing
        ##the 'r+' mode and a new overwrite_field method were added in the ABFileRestart in .abfile


    @lru_cache(maxsize=3)
    def z_coords(self, **kwargs):
        """calculate vertical coordinates given the 3D model state
        """
        ##check if level is provided
        assert 'k' in kwargs, 'missing level index in kwargs for z_coords calc, level=?'

        z = np.zeros(grid.x.shape)
        if kwargs['k'] == 0:
            ##if level index is 0, this is the surface, so just return zeros
            return z
        else:
            ##get layer thickness above level k, convert to z_units, and integrate to total depth
            for k in [k for k in levels if k>=kwargs['k']]:
                rec = kwargs.copy()
                rec['name'] = 'ocean_layer_thick'
                rec['units'] = variables['ocean_layer_thick']['units']
                rec['k'] = k
                d = read_var(path, grid, **rec)
                if kwargs['units'] == 'm':
                    onem = 9806.
                    z -= d/onem  ##accumulate depth in meters, negative relative to surface
                else:
                    raise ValueError('do not know how to calculate z_coords for z_units = '+kwargs['units'])
            return z


    def generate_initial_condition(self, task_id:int=0, task_nproc:int=1, **kwargs):
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


    def prepare_forcing(self, task_id:int, task_nproc:int, **kwargs):
        """Prepare forcing file for the next forecast for a single ensemble member.

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


    def run(self, task_id=0, task_nproc=16, **kwargs):
        self.run_status = 'running'

        job_submit_cmd:str = kwargs['job_submit_cmd']
        # the job submission must through job_submit_node
        # if it is -t devel, the node is oar-dahu3
        # else it is f-dahu
        job_submit_node:str = kwargs['job_submit_node']
        run_dir:str = kwargs['path']
        n_ens:int = kwargs['n_ens']
        os.chdir(run_dir)
        # copy the job submission script to the run directory
        os.system('cp -fv ' +  job_submit_cmd + ' run.sh')
        # specify the number of ensemble members
        os.system(f'sed -i "s/--array N/--array {n_ens}/g" run.sh')
        # copy the model configuration file to the run directory
        os.system('cp -fv ' +  kwargs['model_config_file'] + ' default.cfg')
        # configure the configuration file for current cycle
        #   specify start and enf time of the job
        #   get all required filenames for the initial ensemble
        #     1. get current time, which is supposed to be the start time
        time = kwargs['time']
        #     2. get the restart filename
        file_options_restart = kwargs['files']['restart']
        fname_restart:str = restart.get_restart_filename(file_options_restart,
                                                         1,
                                                         time)
        fname_restart = os.path.basename(fname_restart)
        # Can we do it better?
        # read the config file
        model_config = configparser.ConfigParser()
        model_config.optionxform = str
        model_config.read('default.cfg')
        model_config['model']['init_file'] = fname_restart
        model_config['model']['start'] = kwargs["time"].strftime("%Y-%m-%dT%H:%M:%SZ")
        model_config['model']['stop'] = kwargs["next_time"].strftime("%Y-%m-%dT%H:%M:%SZ")
        model_config['ConfigOutput']['start'] = kwargs["time"].strftime("%Y-%m-%dT%H:%M:%SZ")
        # changing the forcing file in ERA5Atmosphere
        file_options_forcing:dict[str, str] = kwargs['files']['forcing']
        fname_atmos_forcing = forcing.get_forcing_filename(file_options_forcing['atmosphere'],
                                                         1, time)
        fname_atmos_forcing = os.path.basename(fname_atmos_forcing)
        model_config['ERA5Atmosphere']['file'] = fname_atmos_forcing
        # changing the forcing file in ERA5Atmosphere
        fname_ocn_forcing = forcing.get_forcing_filename(file_options_forcing['ocean'],
                                                         1, time)
        fname_ocn_forcing = os.path.basename(fname_ocn_forcing)
        model_config['TOPAZOcean']['file'] = fname_ocn_forcing
        # dump the config to new file
        with open('default.cfg', 'w') as configfile:
            model_config.write(configfile)
        # submit the job
        # the job submission must through job_submit_node
        # specified in yaml file
        # this should f-dahu, or dahu-oar3
        # as oarsub is only available there
        # the job submission is done in the run directory
        process = subprocess.run(['ssh', job_submit_node,
                                  f'cd {run_dir} && '
                                  'oarsub -S ./run.sh'],
                                capture_output=True)
        # obtain the array job id for checking the jobs
        s = process.stdout.decode('utf-8')
        print (s, flush=True)
        s = s.split('OAR_ARRAY_ID=')[-1]
        array_job_id = int(s)
        # check the status of the jobs
        while True:
            # checking this every 300 seconds
            sleep(60)
            # get the status of the jobs
            process = subprocess.run(['ssh', job_submit_node,
                                      'oarstat', '-s', f'-a {array_job_id}'],
                                    capture_output=True)
            s = process.stdout.decode('utf-8').split('\n')[:-1]
            jobs_status = [job.split(':')[-1].replace(' ', '') for job in s]
            # end this loop if all jobs are terminated
            if all([status == 'Terminated' for status in jobs_status]):
                break
            if all([status == 'Error' for status in jobs_status]):
                raise RuntimeError(f'Error job array {array_job_id} in {run_dir}')
        # move the restart file to output directory (next cycle)
        os.chdir(run_dir)
        file_options_restart = kwargs['files']['restart']
        fname_restart:str = restart.get_restart_filename(file_options_restart,
                                                         1,
                                                         kwargs['next_time'])
        fname_restart = os.path.basename(fname_restart)
        for i in range(n_ens):
            os.makedirs(os.path.join(kwargs['output_dir'],
                                     f'ens_{str(i+1).zfill(2)}'), exist_ok=True)
            subprocess.run(['mv', os.path.join(f'ens_{str(i+1).zfill(2)}', fname_restart),
                            os.path.join(kwargs['output_dir'],
                                     f'ens_{str(i+1).zfill(2)}')], check=True)