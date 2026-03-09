import importlib
from typing import Type
from NEDAS.core.types import IOMode
from NEDAS.core.runtime import Runtime

registry = {
    'offline': 'OfflineRuntime',
    'online': 'OnlineRuntime',
}

def get_runtime_class(io_mode: IOMode) -> Type['Runtime']:
    """
    Factory function to return the correct Runtime subclass instance.

    Args:
        io_mode (IOMode): io mode string: 'online' or 'offline'

    Returns:
        Runtime class: Corresponding runtime subclass
    """
    if io_mode.lower() not in registry.keys():
        raise NotImplementedError(f"Unsupported io_mode '{io_mode}'.")
    module = importlib.import_module('NEDAS.runtimes.'+io_mode)
    RuntimeClass = getattr(module, registry[io_mode])

    return RuntimeClass

__all__ = ['registry', 'get_runtime_class']
