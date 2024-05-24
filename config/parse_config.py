import sys
import os
import inspect
import argparse
import yaml
from distutils.util import strtobool
from IPython import get_ipython

def parse_config(**kwargs):
    """
    Load configuration from yaml files and runtime arguments

    The default.yml file is expected in the directory of the code calling
    the parse_config function. The values defined within will be used.

    If user provides --config_file=file.yml then values defined in file.yml
    will overwrite the default values

    The default.yml provides the list of configuration variables,
    an ArgumentParser will be created to parse these variables.
    If user provide a runtime argument --variable=value, then value will be
    used to replace the default value for that variable.

    Returns: Namespace object with config variables
    """

    ##if running from jupyter notebook, ignore all arguments
    if get_ipython() is not None:
        input_args = {}
    else:
        input_args = sys.argv[1:]

    ##try to find a default.yml with definitions of config variables
    stack = inspect.stack()  ##the call stack
    if len(stack) > 1:
        ##this is where the code that calls parse_config() resides
        code_dir = os.path.dirname(stack[1].filename)
    else:
        ##or try to find it in current directory
        code_dir = os.getcwd()
    default_config_file = os.path.join(code_dir, 'default.yml')

    if os.path.exists(default_config_file):
        with open(default_config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {}

    ##optionally, a config file can be specified at runtime
    ##through the config_file argument, build a parser for this
    parser = argparse.ArgumentParser(description='Parse configuration', add_help=False)
    parser.add_argument('-c', '--config_file')
    parser.add_argument('-h', '--help', action='store_true')

    ##parse --config_file and --help first
    args, remaining_args = parser.parse_known_args(input_args)

    ##update config_dict if new config_file is provided
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            config_dict = {**config_dict, **yaml.safe_load(f)}

    if 'config_file' in kwargs:
        with open(kwargs['config_file'], 'r') as f:
            config_dict = {**config_dict, **yaml.safe_load(f)}

    ##continue building the parser with additional arguments to update
    ##individual config variables in config_dict
    parser = argparse.ArgumentParser()

    for key, value in config_dict.items():

        ##variable type for this argument
        if isinstance(value, bool):
            key_type = lambda x: bool(strtobool(x))
        else:
            key_type = type(value)

        ##help message shows the default value and type for this argument
        key_help = f"type: {type(value).__name__}, default: {value}"

        ##add the argument to parser
        parser.add_argument('--'+key, default=value, type=key_type, help=key_help)

    ##show help message
    if args.help or 'help' in kwargs:
        print(f"""Parsing configuration:

Default values are defined in default.yml

You can specify a yaml config file to overwrite (part of) the default configuration
usage: [-c YAML_FILE] [--config_file YAML_FILE]

The following arguments also allow you to update a configuration variable at runtime:
""")
        parser.print_help()

    ##run the parser to get the config namespace object
    config = parser.parse_args(remaining_args)

    return config


