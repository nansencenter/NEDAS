from __future__ import annotations
import importlib
from NEDAS.utils.conversion import ensure_list
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context, Transform

registry = {
    'identity': 'Identity',
    'scale_bandpass': 'ScaleBandpass',
}

def get_transform_funcs(c: Context) -> list[Transform]:
    if c.config.transform_def is None:
        c.config.transform_def = {'type':'identity'}
    
    transform_funcs = []
    for transform_func_def in ensure_list(c.config.transform_def):

        if 'type' not in transform_func_def.keys():
            raise KeyError("'type' needs to be specified in transform_def entries")
        transform_func_type = transform_func_def['type'].lower()

        if transform_func_type not in registry.keys():
            raise NotImplementedError(f"Transform function type '{transform_func_type}' is not implemented.")
        
        module = importlib.import_module('NEDAS.assim_tools.transforms.'+transform_func_type)
        TransformClass = getattr(module, registry[transform_func_type])
        transform_func = TransformClass(**transform_func_def)
        transform_funcs.append(transform_func)

    return transform_funcs

__all__ = ['registry', 'get_transform_funcs']