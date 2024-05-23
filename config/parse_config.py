import os, inspect
import argparse
import yaml
from distutils.util import strtobool

def parse_config(**kwargs):
    """
    Load configuration from yaml files and runtime arguments

    The default.yml file is expected in the directory of the code calling
    the load_config function. The values defined within will be used.

    If user provides --config_file=file.yml then values defined in file.yml
    will overwrite the default values

    The default.yml provides the list of configuration variables,
    a argparse.ArgumentParser will be created to parse these variables.
    If user provide a runtime argument --variable=value, then value will be
    used to replace the default value for that variable.
    """

    ##parser for config_file input
    parser = argparse.ArgumentParser(description='Parse configuration', add_help=False)
    parser.add_argument('-c', '--config_file')
    parser.add_argument('-h', '--help', action='store_true')

    ##if a config_file is specified, load the content into config_from_file
    args, remaining_args = parser.parse_known_args()

    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            config_from_file = yaml.safe_load(f)
    else:
        config_from_file = {}

    ##continue building the parser with remaining arguments from default.yml
    parser = argparse.ArgumentParser()

    ##the directory where the code calling this function is located
    ##expect to find a default.yml in this directory, and
    ##load the content into config_default
    code_dir, code_file = os.path.split(inspect.stack()[1].filename)

    with open(code_dir+'/default.yml', 'r') as f:
        config_default = yaml.safe_load(f)

    for key, value in config_default.items():

        ##default value for this argument
        ##given by the default.yml:
        key_default = value

        ##if new value available from the specified config_file:
        if key in config_from_file:
            key_default = config_from_file[key]

        ##if new value available from kwargs
        if key in kwargs:
            key_default = kwargs[key]

        ##of course if the runtime arguments provided new values,
        ##they will overwrite the default value

        ##variable type for this argument
        if isinstance(key_default, bool):
            key_type = lambda x: bool(strtobool(x))
        else:
            key_type = type(key_default)

        ##help message shows the default value and type for this argument
        key_help = f"type = {type(key_default).__name__}, default = {key_default}"

        ##add the argument to parser
        parser.add_argument('--'+key, default=key_default, type=key_type, help=key_help)

    ##show help message
    if args.help:
        print(f"""Parsing configuration:

Default values are defined in {os.path.join(code_dir, 'default.yml')}

You can specify a yaml config file to overwrite (part of) the default configuration
usage: {code_file} [-c YAML_FILE] [--config_file YAML_FILE]

The following arguments also allow you to update a configuration variable at runtime:
""")
        parser.print_help()

    ##run the parser to get the config namespace object
    config = parser.parse_args(remaining_args)

    return config


