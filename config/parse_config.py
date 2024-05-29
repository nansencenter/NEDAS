import sys
import os
import inspect
import argparse
import yaml
from distutils.util import strtobool

def parse_config(code_dir='.', config_file=None, parse_args=False, **kwargs):
    """
    Load configuration from yaml files and runtime arguments

    Inputs:
    - code_dir: str
      path to the default.yml
    - config_file: str
      alternative yaml config file to overwrite the default settings
    - parse_args: bool
      if true, parse runtime arguments with argparse
      NB: only enable this once in a program to avoid namespace confusion
    - **kwargs
      config_dict entries can also be set here

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

    if parse_args:
        input_args = sys.argv[1:]
    else:
        input_args = {}

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

        ##update value if they are specified in kwargs
        if key in kwargs:
            value = kwargs[key]

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
        exit()

    ##run the parser to get the config namespace object
    config, remaining_args = parser.parse_known_args(remaining_args)

    ##return the dict with config variables
    return vars(config)


