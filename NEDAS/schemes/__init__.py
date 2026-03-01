import importlib
from NEDAS.config import Config
from NEDAS.core import Scheme

registry = {
    'filter': 'FilterAnalysisScheme',
    'forecast': 'ForecastScheme',
}

def get_scheme(config: Config) -> Scheme:
    """
    Factory function to get the correct analysis scheme instance.

    Args:
        config (Config): Configuration.

    Returns:
        Scheme: The analysis scheme class instance.
    """
    if not hasattr(config, 'scheme'):
        raise KeyError("Configuration object needs to define 'scheme'")

    scheme_name = config.scheme.lower()
    if scheme_name not in registry:
        raise NotImplementedError(f"Scheme '{scheme_name}' is not available")

    module = importlib.import_module('NEDAS.schemes.'+scheme_name)
    SchemeClass = getattr(module, registry[scheme_name])

    return SchemeClass(config)

__all__ = ['registry', 'get_scheme']