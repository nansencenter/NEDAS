from code import interact
import os
import inspect
from typing import Literal, Any
import yaml
import dateutil.parser
from datetime import datetime, timezone
import NEDAS
from NEDAS.utils import progress
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
    directories: dict[str, str]
    python_env: str|None
    io_mode: Literal['online', 'offline']
    job_submit: dict|None

    # parallel scheme
    nproc: int
    nproc_mem: int
    nproc_rec: int
    nproc_util: int
    pid: int
    pid_mem: int
    pid_rec: int
    pid_show: int  # avail in context

    # experiment design parameters
    nens: int
    run_preproc: bool
    run_forecast: bool
    run_analysis: bool
    run_postproc: bool
    run_diagnose: bool
    step: str|None

    # runtime logging options
    call_stack: list[dict]|None
    debug: bool
    timer: bool
    quiet: bool
    interactive: bool|None
    is_notebook: bool|None
    call_stack_max_level: int|None
    cols: int
    anchor: int
    tabspace: int
    progress_bar_width: int

    # time control
    time: datetime  # avail in context
    time_start: datetime
    time_end: datetime
    time_analysis_start: datetime
    time_analysis_end: datetime
    cycle_period: float
    forecast_period: float
    obs_time_steps: list[float]
    obs_time_scale: float
    state_time_steps: list[float]
    state_time_scale: float

    # some definitions
    grid_def: dict
    state_def: dict|None
    model_def: dict|None
    obs_def: dict|None
    dataset_def: dict|None
    shuffle_obs: bool
    z_coords_from: Literal['mean', 'member']
    interp_method: str
    perturb: dict|None

    # more details in assimilation algorithm
    scheme: str
    niter: int
    iter: int  # avail in context
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
    alignment: dict|None
    diag: dict|None

    def __init__(self, config_file: str|None=None, parse_args: bool=False, **kwargs):
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
            # initialize current time to start time, if not available
            config_dict['time'] = config_dict['time_start'].replace()
        if config_dict['time_analysis_start'] is None:
            # initialize analysis start time if not available
            config_dict['time_analysis_start'] = config_dict['time_start'].replace()
        if config_dict['time_analysis_end'] is None:
            # initialize analysis end time if not available
            config_dict['time_analysis_end'] = config_dict['time_end'].replace()

        return config_dict

    def _check_parallel_scheme(self, config_dict: dict) -> dict:
        """
        Check the number of processors for parallelization
        """
        # nproc is the total number of processpors
        # if not defined, set to 1 (serial program) by default
        if 'nproc' not in config_dict or config_dict['nproc'] is None:
            config_dict['nproc'] = 1

        # In parallel schemes, the communicator is divided into mem/rec groups
        # nproc_mem and nproc_rec are the number of groups in each direction
        # set default values if they are not defined
        if 'nproc_mem' not in config_dict or config_dict['nproc_mem'] is None:
            config_dict['nproc_mem'] = config_dict['nproc']
        # check if division works
        if config_dict['nproc'] % config_dict['nproc_mem'] != 0:
            raise ValueError(f"nproc={config_dict['nproc']} is not evenly divided by nproc_mem={config_dict['nproc_mem']}")
        config_dict['nproc_rec'] = int(config_dict['nproc']/config_dict['nproc_mem'])

        # nproc_util (optional) is nproc to use for utility functions
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

    def summary(self) -> str:
        """
        Return a comprehensive summary of the NEDAS configuration.
        """
        # Format the active flags for the workflow
        workflow = []
        if self.run_preproc: workflow.append("Preprocess")
        if self.run_analysis: workflow.append("Analysis")
        if self.run_postproc: workflow.append("Postprocess")
        if self.run_forecast: workflow.append("Forecast")
        if self.run_diagnose: workflow.append("Diagnose")
        workflow_str = " -> ".join(workflow) if workflow else "None"

        fcst_str = f"{self.forecast_period}h" if hasattr(self, 'forecast_period') and self.forecast_period else "N/A"
        js = self.job_submit or {}
        loc = self.localization_def or {}
        h_loc = loc.get('horizontal', {}).get('type', 'N/A')
        v_loc = loc.get('vertical', {}).get('type', 'N/A')
        inf = self.inflation_def or {}
        inf_str = f"{inf.get('type', 'None')} (coef: {inf.get('coef', 1.0)}, adaptive: {inf.get('adaptive', False)})"
        state_vars = [f"{d.get('name')} ({d.get('model_src')})" for d in (self.state_def or [])]
        obs_vars = [f"{d.get('name')} ({d.get('dataset_src')})" for d in (self.obs_def or [])]

        # Construct the summary block
        summary_text = f"""
CONFIGURATION SUMMARY
{'='*21}
Directories:
  Work Dir:      {self.work_dir}
  NEDAS Root:    {self.nedas_root}

Time Configuration:
  Current Time:  {self.time}
  Experiment:    [{self.time_start}] to [{self.time_end}]
  Analysis:      [{self.time_analysis_start}] to [{self.time_analysis_end}]
  Periods:       Cycle: {self.cycle_period}h | Forecast: {fcst_str}

Parallel Scheme:
  Total Procs:   {self.nproc}
  Decomposition: {self.nproc_mem} (mem) x {self.nproc_rec} (rec)
  Procs for utility funcs: {self.nproc_util}
  Host:          {js.get('host', 'local')}
  Scheduler:     {js.get('scheduler', 'None')} | Project: {js.get('project', 'N/A')}
  Queue/Mode:    {js.get('queue', 'N/A')} | Parallel mode: {js.get('parallel_mode', 'serial')}

Analysis Scheme:
  General:       Scheme: {self.scheme} | Ensemble Size: {self.nens} | IO: {self.io_mode}
  Grid Type:     {self.grid_def.get('type', 'N/A') if self.grid_def else 'N/A'}
  Iteration:     {self.iter + 1} of {self.niter} (Outer Loops)
  Assimilator:   Type: {self.assimilator_def.get('type') if self.assimilator_def else 'None'}
  Updator:       Type: {self.updator_def.get('type') if self.updator_def else 'None'}
  Inflation:     {inf_str}
  Localization:  H: {h_loc} | V: {v_loc} | T: {loc.get('temporal', {}).get('type', 'N/A')}
  Multiscale:    Resolution Levels: {self.resolution_level} | Character Lengths: {self.character_length}
                 Localization Factor: {self.localize_scale_fac} | Obs Err Factor: {self.obs_err_scale_fac}

Definitions:
  Models Used:   {", ".join(self.model_def.keys()) if self.model_def else 'None'}
  Datasets:      {", ".join(self.dataset_def.keys()) if self.dataset_def else 'None'}
  State Vector:  {', '.join(state_vars) if state_vars else 'None'}
  Observations:  {', '.join(obs_vars) if obs_vars else 'None'}

Workflow Status:
  Active Steps:  {workflow_str}
  Debug Mode:    {self.debug} | Timer: {self.timer} | Interactive: {self.interactive}

"""
        return summary_text
