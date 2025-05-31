import importlib
from typing import Type, TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.datasets import Dataset

registry = {
    'lorenz96': 'Lorenz96Obs',
    'qg': 'QGObs',
    'vort2d': 'Vort2DObs',
    'ecmwf.era5': 'ERA5Data',
    'ecmwf.forecast': 'EcmwfForecastData',
    'ifremer.argo': 'ArgoObs',
    'osisaf.ice_conc': 'OsisafSeaIceConcObs',
    'osisaf.ice_drift': 'OsisafSeaIceDriftObs',
    'cs2smos': 'Cs2SmosObs',
    'rgps': 'RgpsObs',
    'topaz': 'TopazPrepObs',
}

def get_dataset_class(dataset_name: str) -> Type["Dataset"]:
    """
    Factory function to return the correct Dataset subclass.

    Args:
        dataset_name (str): Dataset name

    Returns:
        Type[Dataset]: Corresponding Dataset subclass.
    """
    dataset_name = dataset_name.lower()

    if dataset_name not in registry.keys():
        raise NotImplementedError(f"Dataset class not implemented for '{dataset_name}'")

    module = importlib.import_module('NEDAS.datasets.'+dataset_name)
    DatasetClass = getattr(module, registry[dataset_name])

    return DatasetClass
