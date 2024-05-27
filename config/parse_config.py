import sys
import os
import inspect
import argparse
import yaml
from distutils.util import strtobool
from IPython import get_ipython

def parse_config(code_dir='.', config_file=None):
    """
    Load configuration from yaml files and runtime arguments

    Inputs:
    - code_dir: path to the default.yml
    - config_file: alternative yaml config file to overwrite the default settings

    Return:
    - config_dict: dict[key, value]
      a dictionary with config variables

    If user provides --config_file=file.yml then values defined in file.yml
    will overwrite the default values

    The default.yml provides the list of configuration variables,
    an ArgumentParser will be created to parse these variables.
    If user provide a runtime argument --variable=value, then value will be
    used to replace the default value for that variable.
    """

    ##if running from jupyter notebook, ignore all runtime arguments
    if get_ipython() is not None:
        input_args = {}
    else:
        input_args = sys.argv[1:]

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
    if config_file is not None:
        with open(config_file, 'r') as f:
            config_dict = {**config_dict, **yaml.safe_load(f)}

    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
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
    if args.help:
        print(f"""Parsing configuration:

Default values can be defined in default.yml in the code directory

You can specify a yaml config file to overwrite (part of) the default configuration
usage: code.py [-c YAML_FILE] [--config_file YAML_FILE]

The following arguments also allow you to update a configuration variable at runtime:
""")
        parser.print_help()

    ##run the parser to get the config namespace object
    config = parser.parse_args(remaining_args)

    ##return the dict with config variables
    return vars(config)


