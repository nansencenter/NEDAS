from __future__ import annotations
import importlib
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context

registry = {
  'ensemble': 'EnsembleCovariance',
  # 'static'
}

def get_covariance(c: Context):
    config = c.config

    if config.covariance_def is None:
        config.covariance_def = {'type':'ensemble'}
    if 'type' not in config.covariance_def.keys():
        raise KeyError("'type' needs to be specified in covariance_def entries")
    covariance_type = config.covariance_def['type'].lower()

    if covariance_type not in registry.keys():
        raise NotImplementedError(f"Covariance model '{covariance_type}' not implemented")

    module = importlib.import_module("NEDAS.assim_tools.covariance."+covariance_type)
    CovarianceClass = getattr(module, registry[covariance_type])
    cov = CovarianceClass(**config.covariance_def)

    return cov

__all__ = ['registry', 'get_covariance']
