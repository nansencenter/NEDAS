import importlib
from NEDAS.config import Config
from NEDAS.utils.conversion import ensure_list
from .base import Transform

# make sure keys are lower-case
registry = {
    'identity': 'Identity',
    'scale_bandpass': 'ScaleBandpass',
}

def get_transform_funcs(c: Config) -> list[Transform]:
    if c.transform_def is None:
        c.transform_def = {'type':'identity'}
    
    transform_funcs = []
    for transform_func_def in ensure_list(c.transform_def):

        if 'type' not in transform_func_def.keys():
            raise KeyError("'type' needs to be specified in transform_def entries")
        transform_func_type = transform_func_def['type'].lower()

        if transform_func_type not in registry.keys():
            raise NotImplementedError("Transform function type '{transform_func_type}' is not implemented.")
        
        module = importlib.import_module('NEDAS.assim_tools.transforms.'+transform_func_type)
        TransformClass = getattr(module, registry[transform_func_type])
        transform_funcs.append(TransformClass(**transform_func_def))

    return transform_funcs