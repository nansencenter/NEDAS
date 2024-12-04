import os
import configparser
from utils.conversion import dt1h
from . import restart, forcing

def make_namelist(path='.', **kwargs):
    ens_mem_id:int = kwargs['member'] + 1
    time = kwargs['time']
    forecast_period = kwargs['forecast_period']
    next_time = time + forecast_period * dt1h

    # read the config file
    model_config = configparser.ConfigParser()
    model_config.optionxform = str
    model_config.read('template.cfg')

    ##change the restart file name
    file_options_restart = kwargs['files']['restart']
    fname_restart:str = restart.get_restart_filename(file_options_restart, ens_mem_id, time)
    model_config['model']['init_file'] = fname_restart
    model_config['model']['start'] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    model_config['model']['stop'] = next_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    model_config['ConfigOutput']['start'] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
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
    config_file = os.path.join(path, 'nextsim.cfg')
    with open(config_file, 'w') as configfile:
        model_config.write(configfile)

