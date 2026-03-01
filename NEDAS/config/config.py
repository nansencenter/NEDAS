import os
import inspect
from typing import Optional, Literal, Any
import yaml
import dateutil.parser
from datetime import datetime, timezone
import NEDAS
from .parse_config import parse_config

class Config:
    """
    Class to manage the configuration for running the NEDAS analysis.
    Configuration entries are described in details in :doc:`config_file`.

    Args:
        config_file (str, optional): Path to the configuration file.
        parse_args (bool, optional): If true, parse command line arguments to collect configuration. Default is False.
        **kwargs: Additional key-value pairs to be passed to parse_config. Can be used to override values in the config file.
    """
    work_dir: str
    python_env: Optional[str]
    io_mode: Literal['online', 'offline']
    job_submit: dict
    directories: Optional[dict[str, str]]
    nproc: int
    nproc_mem: int
    nproc_rec: int
    nproc_util: int
    pid: int
    pid_mem: int
    pid_rec: int
    pid_show: int
    nens: int
    run_preproc: bool
    run_forecast: bool
    run_analysis: bool
    run_postproc: bool
    run_diagnose: bool
    debug: bool
    timer: bool
    step: Optional[str]
    time: datetime
    time_start: datetime
    time_end: datetime
    time_analysis_start: datetime
    time_analysis_end: datetime
    cycle_period: float
    obs_time_steps: list[float]
    obs_time_scale: float
    state_time_steps: list[float]
    state_time_scale: float
    grid_def: dict
    state_def: Optional[dict]
    model_def: dict
    obs_def: Optional[dict]
    dataset_def: dict
    shuffle_obs: bool
    z_coords_from: Literal['mean', 'member']
    perturb: Optional[dict]
    scheme: str
    niter: int
    iter: int
    resolution_level: list[int]
    character_length: list[float]
    localize_scale_fac: list[float]
    obs_err_scale_fac: list[float]
    assimilator_def: dict
    updator_def: dict
    covariance_def: dict
    inflation_def: dict
    localization_def: dict
    transform_def: dict
    alignment: Optional[dict]
    diag: Optional[dict]

    def __init__(self, config_file: Optional[str]=None, parse_args: bool=False, **kwargs):
        # parse the yaml config file to obtain the values
        code_dir = os.path.dirname(inspect.getfile(self.__class__))
        config_dict = parse_config(code_dir, config_file, parse_args, **kwargs)

        # replace placeholders in dir paths with actual values
        config_dict['work_dir'] = os.path.abspath(config_dict['work_dir'])
        self.work_dir = config_dict['work_dir']
        self.nedas_root = NEDAS.__path__[0]
        config_dict = self._parse_directories(config_dict)

        # check a few attributes, setting default values if not specified in yaml file
        config_dict = self._check_time_scheme(config_dict)
        config_dict = self._check_parallel_scheme(config_dict)

        # set current iteration to 0 if undefined
        if 'iter' not in config_dict or config_dict['iter'] is None:
            config_dict['iter'] = 0

        # set the attributes
        self.__dict__.update(config_dict)
    
    def _parse_directories(self, data: Any) -> Any:
        """
        Parse the directories or file names defined in :code:`data`
        and replace the placeholders {work_dir} and {nedas_root} with the actual values.
        """
        if isinstance(data, dict):
            return {key: self._parse_directories(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._parse_directories(element) for element in data]
        elif isinstance(data, str):
            return data.replace('{work_dir}', self.work_dir).replace('{nedas_root}', self.nedas_root)
        else:
            return data

    def _check_time_scheme(self, config_dict: dict) -> dict:
        """
        Initialize the time variables for the analysis.

        Checks if the mandatory :code:`time_*` entries are defined in the config file.
        If :code:`time` is not set, set it to :code:`time_start` by default.
        YAML file recognizes 2001-01-01T00:00:00 format and convert directly to datetime object.
        If time is a formatted string, will try to parse it using dateutil.parser.
        """
        # check if mandatory time keys are defined in config file
        for key in ['time', 'time_start', 'time_end', 'time_analysis_start', 'time_analysis_end']:
            if key not in config_dict:
                raise KeyError(f"'{key}' is missing in config file")
            if isinstance(config_dict[key], str):
                try:
                    config_dict[key] = dateutil.parser.parse(config_dict[key])
                except Exception:
                    raise ValueError(f"Failed to convert string {key}={config_dict[key]} to datetime")
            # add default tzinfo
            if config_dict[key] and config_dict[key].tzinfo is None:
                config_dict[key] = config_dict[key].replace(tzinfo=timezone.utc)

        if config_dict['time'] is None:
            ##initialize current time to start time, if not available
            config_dict['time'] = config_dict['time_start'].replace()

        return config_dict

    def _check_parallel_scheme(self, config_dict: dict) -> dict:
        """
        Check the number of processors for parallelization
        """
        ##nproc is the total number of processpors
        ##if not defined, set to 1 (serial program) by default
        if 'nproc' not in config_dict or config_dict['nproc'] is None:
            config_dict['nproc'] = 1

        ##In parallel schemes, the communicator is divided into mem/rec groups
        ##nproc_mem and nproc_rec are the number of groups in each direction
        ##set default values if they are not defined
        if 'nproc_mem' not in config_dict or config_dict['nproc_mem'] is None:
            config_dict['nproc_mem'] = config_dict['nproc']
        ##check if division works
        if config_dict['nproc'] % config_dict['nproc_mem'] != 0:
            raise ValueError(f"nproc={config_dict['nproc']} is not evenly divided by nproc_mem={config_dict['nproc_mem']}")
        config_dict['nproc_rec'] = int(config_dict['nproc']/config_dict['nproc_mem'])

        ##nproc_util (optional) is nproc to use for utility functions
        if 'nproc_util' not in config_dict or config_dict['nproc_util'] is None:
            config_dict['nproc_util'] = config_dict['nproc']

        return config_dict

    def dump_yaml(self, config_file: str):
        """
        Dump the current configuration to a YAML file.

        Args:
            config_file (str): Path to the output configuration file.
        """
        with open(config_file, 'w') as f:
            yaml.dump(self.__dict__, f, sort_keys=False)


    def show_summary(self):
        """
        Print a summary of the configuration.
        """
        print(f"""Initializing config...
 working directory: {self.work_dir}
 parallel scheme: nproc = {self.nproc}, nproc_mem = {self.nproc_mem}
 cycling from {self.time_start} to {self.time_end}
 analysis start at {self.time_analysis_start}
 cycle_period = {self.cycle_period} hours
 current time: {self.time}
 nens: {self.nens}
 Assimilation scheme: TODO
 """, flush=True)