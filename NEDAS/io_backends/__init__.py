import importlib
from NEDAS.core.io_backend import IOBackend
from NEDAS.core.types import IOMode

registry = {
    'offline': 'OfflineIO',
    'online': 'OnlineIO',
}

def get_io_backend(io_mode: IOMode) -> IOBackend:
    """
    Factory function to return the correct IOBackend subclass instance.

    Args:
        io_mode (IOMode): IO mode

    Returns:
        IOBackend: Corresponding IOBackend subclass instance
    """
    if io_mode not in registry.keys():
        raise NotImplementedError(f"Unsupported io_mode '{io_mode}'.")
    module = importlib.import_module('NEDAS.io_backends.'+io_mode)
    IOClass = getattr(module, registry[io_mode])

    return IOClass()

__all__ = ['registry', 'get_io_backend']
