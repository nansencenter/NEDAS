from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context, Inflation

registry = {
    'multiplicative': 'MultiplicativeInflation',
    'RTPP': 'RTPPInflation',
}

def get_inflation_func(c: Context) -> Inflation:
    """
    Get the correct Inflation subclass instance based on configuration

    Args:
        c (Context): the runtime context.

    Returns:
        Inflation: Corresponding Inflation subclass instance.
    """
    if not hasattr(c.config, 'inflation_def'):
        raise AttributeError("'inflation_def' missing from configuration")
    if not isinstance(c.config.inflation_def, dict):
        c.config.inflation_def = {}
    if 'type' not in c.config.inflation_def.keys():
        raise KeyError("'type' needs to be specified in inflation_def")
    inflation_type = c.config.inflation_def['type'].split(',')

    prior = ('prior' in inflation_type)
    post = ('post' in inflation_type)

    adaptive = c.config.inflation_def.get('adaptive', False)
    coef = c.config.inflation_def.get('coef', 1.0)

    for key in registry.keys():
        if key in inflation_type:
            module = importlib.import_module('NEDAS.assim_tools.inflation.'+key)
            InflationClass = getattr(module, registry[key])
            return InflationClass(coef, adaptive, prior, post)

    raise RuntimeError("No valid inflation class found, check c.inflation_def")

__all__ = ['registry', 'get_inflation_func']
