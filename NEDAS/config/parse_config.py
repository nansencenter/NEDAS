import numpy
import sys
import os
import re
import argparse
import yaml

def convert_notation(data):
    """
    Parse values in data, convert strings to appropriate types if possible.

    Goes through the data recursively if it is a list or a dictionary. If the string can be interpreted as a
    scientific notation, inf, or boolean flag, convert it to the appropriate type.

    Args:
        data (any): The data to be converted.

    Returns:
        any: The converted data.
    """
    if isinstance(data, dict):
        return {key: convert_notation(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_notation(element) for element in data]
    elif isinstance(data, str):
        ## Match scientific notation pattern and convert to float
        if re.match(r'^-?\d+(\.\d*)?[eE]-?\d+$', data):
            return float(data)
        ## convert "inf" to numpy.inf
        elif data.lower() == "inf":
            return numpy.inf
        elif data.lower() == "-inf":
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

def str2bool(data:str) -> int:
    """
    Convert a string to a boolean value (0 or 1).
    """
    if data.lower() in ['y', 'yes', 'on', 't', 'true', '.true.']:
        return 1
    elif data.lower() in ['n', 'no', 'off', 'f', 'false', '.false.']:
        return 0
    else:
        raise ValueError(f"Invalid input value for str2bool: {data}")

def parse_config(code_dir='.', config_file=None, parse_args=False, **kwargs):
    """
    Load configuration from YAML files and runtime arguments.

    This function loads configuration settings from a default YAML file, located in `code_dir/default.yml`.
    If `config_file` is provided, then values defined in that file will overwrite those in the default YAML file.
    If additional key=value pairs are provided, either through runtime command-line arguments or as `kwargs`,
    those will also overwrite the existing values.

    The default YAML file also serves as a template for the ArgumentParser, which parses the key-value pairs,
    if the value is not None, parser will add the argument with type and default value in the help message.

    Args:
        code_dir (str, optional): Directory containing the `default.yml` file. Default is the current directory.
        config_file (str, optional): Alternative YAML config file to overwrite default settings.
        parse_args (bool, optional): If True, parse runtime arguments with argparse. Default is False.
        **kwargs: Additional configuration key-value pairs that can overwrite existing settings.

    Returns:
        dict: A dictionary containing all configuration variables.
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
    args, remaining_args = parser.parse_known_args(input_args) # type: ignore

    ##update config_dict if new config_file is provided
    if config_file is not None:
        with open(config_file, 'r') as f:
            config_dict = {**config_dict, **yaml.safe_load(f)}

    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            config_dict = {**config_dict, **yaml.safe_load(f)}

    if not isinstance(config_dict, dict):
        config_dict = {}

    ##append new config variables defined in kwargs
    config_dict = {**config_dict, **kwargs}

    ##continue building the parser with additional arguments to update
    ##individual config variables in config_dict
    parser = argparse.ArgumentParser()

    for key, value in config_dict.items():
        key_rec = {}
        value = convert_notation(value)
        key_rec['default'] = value
        if value is not None:
            ##bool variable type needs special treatment for parsing runtime input string
            if isinstance(value, bool):
                key_rec['type'] = lambda x: bool(str2bool(x))
            else:
                key_rec['type'] = type(value)
            ##help message shows the default value and type for this argument
            key_rec['help'] = f"type: {type(value).__name__}, default: {value}".replace('%','%%')

        ##add the argument to parser
        parser.add_argument('--'+key, **key_rec)

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