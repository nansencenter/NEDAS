import importlib
from NEDAS.config import Config
from .base import Updator

registry = {
    'additive': 'AdditiveUpdator',
    'alignment': 'AlignmentUpdator',
    'alignment_interp': 'AdditiveUpdator',
}

def get_updator(c: Config) -> Updator:
    """
    Get the correct Updator subclass instance based on the configuration.

    Args:
        c (Config): Configuration object.

    Returns:
        Updator: Corresponding Updator subclass instance.
    """
    if 'type' not in c.updator_def.keys():
        raise KeyError("'type' needs to be specified in updator_def")
    updator_type = c.updator_def['type'].lower()

    if updator_type not in registry:
        raise NotImplementedError(f"updator type '{updator_type}' is not implemented")

    ##TODO: find a better logic
    if c.iter == c.niter-1:
        updator_type = 'additive' 
    module = importlib.import_module('NEDAS.assim_tools.updators.'+updator_type)
    UpdatorClass = getattr(module, registry[updator_type])

    return UpdatorClass(c)
