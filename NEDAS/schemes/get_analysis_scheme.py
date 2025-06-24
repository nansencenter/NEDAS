import importlib
from typing import Any

registry = {
    'offline_filter': 'OfflineFilterAnalysisScheme',
}

def get_analysis_scheme(c) -> Any:
    """
    Factory function to get the correct analysis scheme instance.

    Args:
        c (Config): Configuration.

    Returns:
        The analysis scheme class instance.
    """
    if not hasattr(c, 'analysis_scheme'):
        raise KeyError("Configuration object needs to define 'analysis_scheme'")
    scheme_name = c.analysis_scheme.lower()
    if scheme_name not in registry:
        raise NotImplementedError(f"Analysis scheme '{scheme_name}' is not available")

    module = importlib.import_module('NEDAS.schemes.'+scheme_name)
    Scheme = getattr(module, registry[scheme_name])

    return Scheme()
