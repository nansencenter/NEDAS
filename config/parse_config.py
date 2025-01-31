import numpy
import sys
import os
import re
import inspect
import argparse
import yaml

def convert_notation(data):
    if isinstance(data, dict):
        return {key: convert_notation(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_notation(element) for element in data]
    elif isinstance(data, str):
        ## Match scientific notation pattern and convert to float
        if re.match(r'^-?\d+(\.\d*)?[eE]-?\d+$', data):
            return float(data)
        ## convert "inf" to numpy.inf
        elif data == "inf":
            return numpy.inf
        elif data == "-inf":
            return -numpy.inf
        ## convert string to bool
        elif data.lower() in ['y', 'yes', 'on', 't', 'true', '.true.']:
            return True
        elif data.lower() in ['n', 'no', 'off', 'f', 'false', '.false.']:
            return False
        else:
            return data
    else:
        return data

def str2bool(data):
    if data.lower() in ['y', 'yes', 'on', 't', 'true', '.true.', '1']:
        return 1
    elif data.lower() in ['n', 'no', 'off', 'f', 'false', '.false.', '0']:
        return 0
    else:
        raise ValueError(f"Invalid input value for str2bool: {data}")

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
      config_dict entries can also be added/updated through kwargs

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
        if len(input_args) == 0:
            print(f"Usage: {sys.argv[0]} -c YAML_config_file")
            exit()
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

    ##append new config variables defined in kwargs
    config_dict = {**config_dict, **kwargs}

    ##continue building the parser with additional arguments to update
    ##individual config variables in config_dict
    parser = argparse.ArgumentParser()

    for key, value in config_dict.items():

        value = convert_notation(value)

        ##bool variable type needs special treatment for parsing runtime input string
        if isinstance(value, bool):
            key_type = lambda x: bool(str2bool(x))
        else:
            key_type = type(value)

        ##help message shows the default value and type for this argument
        key_help = f"type: {type(value).__name__}, default: {value}"

        ##add the argument to parser
        parser.add_argument('--'+key, default=value, type=key_type, help=key_help)

    ##show help message
    if args.help:
        print(f"""
Default configuration variables are defined in 'default.yml' in the code directory.

You can specify a YAML config file by [-c YAML_FILE] or [--config_file YAML_FILE] to overwrite the default configuration.

Furthermore, you can also overwrite some configuration variables by specifying them at runtime:
""")
        parser.print_help()
        exit()

    ##run the parser to get the config namespace object
    config, remaining_args = parser.parse_known_args(remaining_args)

    ##return the dict with config variables
    return vars(config)


