from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context

"""Localization funcs needs to be pure functions (they need to be used in numba njit)"""

registry = {
    'gaspari_cohn': ('distance_based', 'gaspari_cohn_func'),
    'step': ('distance_based', 'step_func'),
    'exponential': ('distance_based', 'exponential_func'),
    'gcv': ('correlation_based', 'gcv_localization_func'),
    'polo': ('correlation_based', 'prior_optimal_localization_func')
}


def get_localization_funcs(c: Context) -> dict:
    local_funcs = {}

    assert c.config.localization_def is not None, \
        "c.localization_def needs to be defined in the config file"

    # Distance-based localization
    for key in ['horizontal', 'vertical', 'temporal']:
        assert key in c.config.localization_def, \
            f"{key} needs to be defined in c.localization_def"

        if c.config.localization_def[key]:
            if 'type' not in c.config.localization_def[key]:
                raise KeyError(
                    f"'type' needs to be specified for "
                    f"c.localization_def['{key}']"
                )
            local_funcs[key] = get_localization_func_component(
                c.config.localization_def[key]['type']
            )
        else:
            local_funcs[key] = None

    # Correlation-based localization is optional
    correlation_def = c.config.localization_def.get('correlation')

    if correlation_def and correlation_def.get('type') is not None:
        local_funcs['correlation'] = get_correlation_localization_func(
            correlation_def['type']
        )
    else:
        local_funcs['correlation'] = None

    return local_funcs


def get_localization_func_component(localization_types):
    localization_types = localization_types.lower().split(',')

    # Distance-based localization schemes
    if 'gaspari_cohn' in localization_types:
        from .distance_based import gaspari_cohn_func as local_func
    elif 'step' in localization_types:
        from .distance_based import step_func as local_func
    elif 'exponential' in localization_types:
        from .distance_based import exponential_func as local_func
    else:
        raise ValueError(f"Unknown localization type {localization_types}")

    return local_func


def get_correlation_localization_func(localization_type):
    localization_type = localization_type.lower()

    if localization_type == 'gcv':
        from .correlation_based import gcv_localization_func as local_func
    elif localization_type == 'polo':
        from .correlation_based import prior_optimal_localization_func as local_func
    else:
        raise ValueError(
            f"Unknown correlation localization type {localization_type}"
        )

    return local_func


__all__ = ['registry', 'get_localization_funcs']
