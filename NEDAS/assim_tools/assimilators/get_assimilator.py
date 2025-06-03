import importlib
from NEDAS.config import Config
from .base import Assimilator

# make sure keys are lower-case
registry = {
    'ETKF': 'ETKFAssimilator',
    'EAKF': 'EAKFAssimilator',
    'TopazDEnKF': 'TopazDEnKFAssimilator',
    #'PDAF': 'PDAFAssimilator',
    #'RHF'
}

def get_assimilator(c: Config) -> Assimilator:
    """
    Get the correct Assimilator subclass instance based on the configuration.

    Args:
        c (Config): Configuration object.

    Returns:
        Assimilator: Corresponding Assimilator subclass instance.
    """
    if 'type' not in c.assimilator_def.keys():
        raise KeyError("'type' needs to be specified in assimilator_def")
    assimilator_type = c.assimilator_def['type']

    if assimilator_type not in registry:
        raise NotImplementedError(f"Assimilator type '{assimilator_type}' is not implemented")

    module = importlib.import_module('NEDAS.assim_tools.assimilators.'+assimilator_type)
    AssimilatorClass = getattr(module, registry[assimilator_type])

    return AssimilatorClass(c)
