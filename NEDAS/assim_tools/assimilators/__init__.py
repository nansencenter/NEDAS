from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
from NEDAS.utils.conversion import resolve_iter_dict
if TYPE_CHECKING:
    from NEDAS.core import Context, Assimilator

registry = {
    'ETKF': 'ETKFAssimilator',
    'EAKF': 'EAKFAssimilator',
    'TopazDEnKF': 'TopazDEnKFAssimilator',
    'QCEF': 'QCEFAssimilator',
    #'PDAF': 'PDAFAssimilator',
    #'RHF'
}

def get_assimilator(c: Context) -> Assimilator:
    """
    Get the correct Assimilator subclass instance based on the configuration.

    Args:
        c (Context): the runtime context object.

    Returns:
        Assimilator: Corresponding Assimilator subclass instance.
    """
    if c.config.assimilator_def is None:
        raise ValueError("assimilator_def not found in Config")
    if 'type' not in c.config.assimilator_def.keys():
        raise KeyError("'type' needs to be specified in assimilator_def")
    # 'type' can be a plain string (used at every outer-loop iteration, unchanged
    # behavior) or a per-iteration dict, e.g. {iter0: ETKF, iter1: TopazDEnKF}
    assimilator_type = resolve_iter_dict(c.config.assimilator_def['type'], c.iter, c.config.niter)

    if assimilator_type not in registry:
        raise NotImplementedError(f"Assimilator type '{assimilator_type}' is not implemented")

    module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_type)
    AssimilatorClass = getattr(module, registry[assimilator_type])

    return AssimilatorClass(c)

__all__ = ['registry', 'get_assimilator']
