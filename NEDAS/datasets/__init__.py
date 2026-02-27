import importlib
from typing import Type
from NEDAS.core.dataset import Dataset

registry = {
    'ecmwf.era5': 'ERA5Data',
    'ecmwf.forecast': 'EcmwfForecastData',
    'ifremer.argo': 'ArgoObs',
    'osisaf.ice_conc': 'OsisafSeaIceConcObs',
    'osisaf.ice_drift': 'OsisafSeaIceDriftObs',
    'amsr2': 'AMSR2Obs',
    'cs2smos': 'Cs2SmosObs',
    'rgps': 'RgpsObs',
    'topaz': 'TopazPrepObs',
    'vort2d': 'Vort2DObs',
    'synthetic': 'SyntheticObs',
}

def get_dataset_class(dataset_name: str) -> Type["Dataset"]:
    """
    Factory function to return the correct Dataset subclass.

    Args:
        dataset_name (str): Dataset name

    Returns:
        Type["Dataset"]: Corresponding Dataset subclass.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in registry.keys():
        raise NotImplementedError(f"Dataset class not implemented for '{dataset_name}'")

    module = importlib.import_module('NEDAS.datasets.'+dataset_name)
    DatasetClass = getattr(module, registry[dataset_name])

    return DatasetClass

__all__ = ['registry', 'get_dataset_class']