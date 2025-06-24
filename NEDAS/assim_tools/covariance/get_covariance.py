import importlib

registry = {
  'ensemble': 'EnsembleCovariance',
  # 'static'
}

def get_covariance(c):

    if c.covariance not in registry.keys():
        raise NotImplementedError(f"Covariance model '{c.covariance}' not implemented")

    module = importlib.import_module("NEDAS.assim_tools.covariance."+c.covariance)

    Covariance = getattr(module, registry[c.covariance])

    return Covariance(c)
