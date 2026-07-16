from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context, Preconditioner

registry = {
    'identity': 'Identity',
    'alignment': 'AlignmentPreconditioner',
}

def get_preconditioner(c: Context) -> Preconditioner:
    """
    Get the correct Preconditioner subclass instance based on the configuration.

    Args:
        c (Context): the runtime context object.

    Returns:
        Preconditioner: Corresponding Preconditioner subclass instance.
    """
    if c.config.preconditioner_def is None:
        c.config.preconditioner_def = {'type': 'identity'}

    if 'type' not in c.config.preconditioner_def.keys():
        raise KeyError("'type' needs to be specified in preconditioner_def")
    preconditioner_type = c.config.preconditioner_def['type'].lower()

    if preconditioner_type not in registry:
        raise NotImplementedError(f"Preconditioner type '{preconditioner_type}' is not implemented")

    module = importlib.import_module('NEDAS.assim_tools.preconditioners.'+preconditioner_type)
    PreconditionerClass = getattr(module, registry[preconditioner_type])

    return PreconditionerClass(c, **c.config.preconditioner_def)

__all__ = ['registry', 'get_preconditioner']
