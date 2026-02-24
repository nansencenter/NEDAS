import importlib
from NEDAS.config import Config
from NEDAS.core import Scheme

"""
Registry is a dict with key=module_name value=class_name
module_name is f"{io_mode}_{analysis_scheme}" from the config file
"""
registry = {
    'offline_filter': 'OfflineFilterAnalysisScheme',
    'online_filter': 'OnlineFilterAnalysisScheme',
    'offline_forecast': 'ForecastScheme',
}

def get_analysis_scheme(cf: Config) -> Scheme:
    """
    Factory function to get the correct analysis scheme instance.

    Args:
        cf (Config): Configuration.

    Returns:
        Scheme: The analysis scheme class instance.
    """
    if cf.io_mode not in ('online', 'offline'):
        raise ValueError(f"Unknown io_mode: {cf.io_mode}")
    if not hasattr(cf, 'analysis_scheme'):
        raise KeyError("Configuration object needs to define 'analysis_scheme'")

    scheme_name = cf.io_mode+'_'+cf.analysis_scheme.lower()
    if scheme_name not in registry:
        raise NotImplementedError(f"Analysis scheme '{scheme_name}' is not available")

    module = importlib.import_module('NEDAS.schemes.'+scheme_name)
    SchemeClass = getattr(module, registry[scheme_name])

    return SchemeClass()
