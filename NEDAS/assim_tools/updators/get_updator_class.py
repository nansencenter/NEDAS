import importlib
from typing import Type
from .base import Updator

registry = {
    'additive': 'AdditiveUpdator',
    'alignment': 'AlignmentUpdator',
    'alignment_interp': 'AdditiveUpdator',
}

def get_updator_class(updator_type: str) -> Type['Updator']:
    """
    Get the correct Updator subclass based on the input type.

    Args:
        updator_type (str): Updator type.

    Returns:
        Type[Updator]: Corresponding Updator subclass.
    """
    if updator_type not in registry:
        raise NotImplementedError(f"updator '{updator_type}' is not available")

    module = importlib.import_module('NEDAS.assim_tools.updators.'+updator_type)
    UpdatorClass = getattr(module, registry[updator_type])
    return UpdatorClass
