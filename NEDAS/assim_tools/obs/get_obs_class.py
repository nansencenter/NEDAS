import importlib
from typing import Type
from .base import Obs

registry = {
}

def get_obs_class(obs_type: str) -> Type['Obs']:
    if obs_type not in registry.keys():
        return Obs
    module = importlib.import_module('NEDAS.assim_tools.obs.'+obs_type)
    ObsClass = getattr(module, registry[obs_type])
    return ObsClass

