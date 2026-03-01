import importlib
from NEDAS.config import Config
from NEDAS.core import Scheme

registry = {
    #'offline_filter': 'OfflineFilterAnalysisScheme',
    #'online_filter': 'OnlineFilterAnalysisScheme',
    #'forecast': 'ForecastScheme',
}

def get_scheme(config: Config) -> Scheme:
    """
    Factory function to get the correct analysis scheme instance.

    Args:
        config (Config): Configuration.

    Returns:
        Scheme: The analysis scheme class instance.
    """
    if config.io_mode not in ('online', 'offline'):
        raise ValueError(f"Unknown io_mode: {config.io_mode}")
    if not hasattr(config, 'scheme'):
        raise KeyError("Configuration object needs to define 'scheme'")

    scheme_name = config.scheme.lower()
    if scheme_name not in registry:
        raise NotImplementedError(f"Analysis scheme '{scheme_name}' is not available")

    module = importlib.import_module('NEDAS.schemes.'+scheme_name)
    SchemeClass = getattr(module, registry[scheme_name])

    scheme = SchemeClass(config)

    # check if scheme has correct io_mode
    if config.io_mode not in scheme.supported_io_modes:
        raise ValueError(f"Scheme '{scheme.__class__.__name__}' doesn't support the config io_mode {config.io_mode}")

    return scheme

__all__ = ['registry', 'get_scheme']