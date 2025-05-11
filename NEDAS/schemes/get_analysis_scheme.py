import importlib
from NEDAS.schemes.base import AnalysisScheme

registry = {
    'offline_filter': 'OfflineFilterAnalysisScheme',
}

def get_analysis_scheme(analysis_scheme: str) -> AnalysisScheme:
    """
    Factory function to get the correct analysis scheme instance.

    Args:
        analysis_scheme (str): Configuration.

    Returns:
        AnalysisScheme: The analysis scheme subclass instance.
    """
    if analysis_scheme not in registry:
        raise NotImplementedError(f"Analysis scheme '{analysis_scheme}' is not available")

    module = importlib.import_module('NEDAS.schemes.'+analysis_scheme)
    Scheme = getattr(module, registry[analysis_scheme])
    return Scheme()
