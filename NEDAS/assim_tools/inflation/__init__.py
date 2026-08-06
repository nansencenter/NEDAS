from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
from NEDAS.utils.conversion import resolve_iter_dict
if TYPE_CHECKING:
    from NEDAS.core import Context, Inflation

registry = {
    'multiplicative': 'MultiplicativeInflation',
    'RTPP': 'RTPPInflation',
    'RTPS': 'RTPSInflation',
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
    # 'type', 'coef' and 'adaptive' can each be a plain value (used at every outer-loop
    # iteration, unchanged behavior) or a per-iteration dict, e.g.
    # {iter0: 'prior,multiplicative', iter1: 'post,multiplicative,once_after_outer_loop'}.
    # Note 'once_after_outer_loop' is inherently a cross-iteration concept (it defers
    # to schemes/filter.py's final_inflation(), which runs once after all iterations) --
    # varying it by iteration only makes sense if every iteration agrees on it.
    inflation_type = resolve_iter_dict(c.config.inflation_def['type'], c.iter, c.config.niter).split(',')

    prior = ('prior' in inflation_type)
    post = ('post' in inflation_type)
    timing = 'once_after_outer_loop' if 'once_after_outer_loop' in inflation_type else 'per_iteration'

    adaptive = resolve_iter_dict(c.config.inflation_def.get('adaptive', False), c.iter, c.config.niter)
    coef = resolve_iter_dict(c.config.inflation_def.get('coef', 1.0), c.iter, c.config.niter)

    for key in registry.keys():
        if key in inflation_type:
            module = importlib.import_module('NEDAS.assim_tools.inflation.'+key)
            InflationClass = getattr(module, registry[key])
            kwargs = {}
            if key == 'multiplicative' and 'max_coef' in c.config.inflation_def:
                kwargs['max_coef'] = c.config.inflation_def['max_coef']
            if key == 'multiplicative' and 'post_infl_formula' in c.config.inflation_def:
                kwargs['post_infl_formula'] = c.config.inflation_def['post_infl_formula']
            return InflationClass(coef, adaptive, prior, post, timing, **kwargs)

    raise RuntimeError("No valid inflation class found, check c.inflation_def")

__all__ = ['registry', 'get_inflation_func']
