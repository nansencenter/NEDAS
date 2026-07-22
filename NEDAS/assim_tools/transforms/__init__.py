from __future__ import annotations
import importlib
from NEDAS.utils.conversion import ensure_list, resolve_iter_dict, is_iter_keyed_dict
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

    # transform_def's own list already means "chain of transforms within one
    # iteration" (assim_tools/transforms/__init__.py, unchanged below) -- that
    # meaning must not be reused for per-iteration variation. Instead the WHOLE
    # transform_def value can be wrapped in a per-iteration dict, e.g.
    # {iter0: [{type: scale_bandpass, ...}], iter1: [{type: identity}]}. Detected
    # via is_iter_keyed_dict (not resolve_iter_dict's own dict-vs-other check) because
    # transform_def's old format can ALREADY legitimately be a bare dict (a single
    # transform spec, e.g. {'type': 'scale_bandpass', 'decompose_obs': False}) --
    # only a dict whose keys are ALL iterN/default is treated as the new wrapper.
    transform_def = c.config.transform_def
    if is_iter_keyed_dict(transform_def):
        transform_def = resolve_iter_dict(transform_def, c.iter, c.config.niter)

    transform_funcs = []
    for transform_func_def in ensure_list(transform_def):

        if 'type' not in transform_func_def.keys():
            raise KeyError("'type' needs to be specified in transform_def entries")
        transform_func_type = transform_func_def['type'].lower()

        if transform_func_type not in registry.keys():
            raise NotImplementedError(f"Transform function type '{transform_func_type}' is not implemented.")

        module = importlib.import_module('NEDAS.assim_tools.transforms.'+transform_func_type)
        TransformClass = getattr(module, registry[transform_func_type])
        transform_func = TransformClass(c, **transform_func_def)
        transform_funcs.append(transform_func)

    return transform_funcs

__all__ = ['registry', 'get_transform_funcs']