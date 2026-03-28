from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context, Updator

registry = {
    'additive': 'AdditiveUpdator',
    'alignment': 'AlignmentUpdator',
    'alignment_interp': 'AdditiveUpdator',
}

def get_updator(c: Context) -> Updator:
    """
    Get the correct Updator subclass instance based on the configuration.

    Args:
        c (Context): the runtime context

    Returns:
        Updator: Corresponding Updator subclass instance.
    """
    if not hasattr(c.config, 'updator_def'):
        raise AttributeError("'updator_def' missing in configuration")
    if not isinstance(c.config.updator_def, dict):
        c.config.updator_def = {}
    if 'type' not in c.config.updator_def.keys():
        c.config.updator_def['type'] = 'additive'
    updator_type = c.config.updator_def['type'].lower()

    if updator_type not in registry:
        raise NotImplementedError(f"updator type '{updator_type}' is not implemented")

    # TODO: last scale component doesn't need alignment, find a better general logic
    assert c.config.niter is not None
    if c.iter == c.config.niter-1:
        updator_type = 'additive'
    module = importlib.import_module('NEDAS.assim_tools.updators.'+updator_type)
    UpdatorClass = getattr(module, registry[updator_type])

    return UpdatorClass(c)

__all__ = ['registry', 'get_updator']
