import importlib
from NEDAS.config import Config
from .base import Inflation

registry = {
    'multiplicative': 'MultiplicativeInflation',
    'RTPP': 'RTPPInflation',
}

def get_inflation_func(c: Config) -> Inflation:
    """
    Get the correct Inflation subclass instance based on configuration

    Args:
        c (Config): Confugration object.

    Returns:
        Inflation: Corresponding Inflation subclass instance.
    """
    if not hasattr(c, 'inflation_def'):
        raise AttributeError("'inflation_def' missing from configuration")
    if not isinstance(c.inflation_def, dict):
        c.inflation_def = {}
    if 'type' not in c.inflation_def.keys():
        raise KeyError("'type' needs to be specified in inflation_def")
    inflation_type = c.inflation_def['type'].split(',')

    prior = ('prior' in inflation_type)
    posterior = ('posterior' in inflation_type)

    adaptive = c.inflation_def.get('adaptive', False)
    coef = c.inflation_def.get('coef', 1.0)

    for key in registry.keys():
        if key in inflation_type:
            module = importlib.import_module('NEDAS.assim_tools.inflation.'+key)
            InflationClass = getattr(module, registry[key])
            return InflationClass(coef, adaptive, prior, posterior)

    raise RuntimeError("No valid inflation class found, check c.inflation_def")
