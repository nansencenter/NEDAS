import importlib
from typing import Type
from .base import Assimilator

registry = {
    'ETKF': 'ETKFAssimilator',
    'EAKF': 'EAKFAssimilator',
    'TopazDEnKF': 'TopazDEnKFAssimilator',
    #'PDAF': 'PDAFAssimilator',
    #'RHF'
    #'QCEF'
}

def get_assimilator_class(assimilator_type: str) -> Type[Assimilator]:
    """
    Get the correct Assimilator subclass based on the input type.

    Args:
        assimilator_type (str): Assimilator type.

    Returns:
        Type[Assimilator]: Corresponding Assimilator subclass.
    """
    if assimilator_type not in registry:
        raise NotImplementedError(f"Assimilator '{assimilator_type}' is not available")

    module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_type)
    AssimilatorClass = getattr(module, registry[assimilator_type])
    return AssimilatorClass
