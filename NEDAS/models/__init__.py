import importlib
from typing import Type
from NEDAS.core import Model

registry = {
    'lorenz96': 'Lorenz96Model',
    'qg.fortran': 'QGFortranModel',
    'qg.fortran.emulator': 'QGFortranModelEmulator',
    'vort2d': 'Vort2DModel',
    'topaz.v5': 'Topaz5Model',
    'nextsim.v1': 'NextsimModel',
    'nextsim.dg': 'NextsimDGModel',
    'wrf': 'WRFModel',
}

def get_model_class(model_name: str) -> Type["Model"]:
    """
    Factory function to return the correct Model subclass.

    Args:
        model_name (str): Model name

    Returns:
        Type["Model"]: Corresponding Model subclass
    """
    model_name = model_name.lower()

    if model_name not in registry.keys():
        raise NotImplementedError(f"Model class not implemented for '{model_name}'")

    module = importlib.import_module('NEDAS.models.'+model_name)
    ModelClass = getattr(module, registry[model_name])

    return ModelClass

__all__ = ['registry', 'get_model_class']