from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
from NEDAS.utils.conversion import resolve_iter_dict
if TYPE_CHECKING:
    from NEDAS.core import Context, Updator

registry = {
    'additive': 'AdditiveUpdator',
    'alignment': 'AlignmentUpdator',
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
    type_value = c.config.updator_def['type']

    assert c.config.niter is not None
    if isinstance(type_value, dict):
        # explicit per-iteration override, e.g. {iter0: alignment, iter1: alignment,
        # iter2: additive} -- user is fully in control here, so no implicit
        # last-iteration override below (that special case exists only to preserve
        # old single-type configs' historical behavior, see the else branch).
        updator_type = resolve_iter_dict(type_value, c.iter, c.config.niter).lower()
    else:
        updator_type = type_value.lower()

    if updator_type not in registry:
        raise NotImplementedError(f"updator type '{updator_type}' is not implemented")

    module = importlib.import_module('NEDAS.assim_tools.updators.'+updator_type)
    UpdatorClass = getattr(module, registry[updator_type])

    return UpdatorClass(c)

__all__ = ['registry', 'get_updator']
