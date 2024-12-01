import os
import configparser
from . import restart, forcing

def namelist(**kwargs):

    # read the config file
    model_config = configparser.ConfigParser()
    model_config.optionxform = str
    model_config.read('template.cfg')
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
    config_file = os.path.join(run_dir, 'nextsim.cfg')
    with open(config_file, 'w') as configfile:
        model_config.write(configfile)

