import importlib
from NEDAS.config import Config
from NEDAS.core import Scheme

registry = {
    'filter': 'FilterAnalysisScheme',
    'forecast': 'ForecastScheme',
}

def get_scheme(config: Config|None=None,
               config_file: str|None=None, parse_args: bool=False, **kwargs) -> Scheme:
    """
    Factory function to get the correct analysis scheme instance.

    Args:
        config (Config, optional): configuration object.
        config_file (str, optional): the top-level configuration file
        parse_args (bool, optional): whether to parse runtime argument from cli, default to false.
        **kwargs: other key value pairs to rewrite the configuration

    Returns:
        Scheme: The analysis scheme class instance.
    """
    if not config:
        config = Config(config_file=config_file, parse_args=parse_args, **kwargs)
    if not hasattr(config, 'scheme'):
        raise KeyError("Configuration object needs to define 'scheme'")

    scheme_name = config.scheme.lower()
    if scheme_name not in registry:
        raise NotImplementedError(f"Scheme '{scheme_name}' is not available")

    module = importlib.import_module('NEDAS.schemes.'+scheme_name)
    SchemeClass = getattr(module, registry[scheme_name])

    return SchemeClass(config=config, config_file=config_file, parse_args=parse_args, **kwargs)

__all__ = ['registry', 'get_scheme']