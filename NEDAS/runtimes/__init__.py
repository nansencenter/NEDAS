from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from NEDAS.core import Context, Runtime
import importlib

registry = {
    'offline': 'OfflineRuntime',
    'online': 'OnlineRuntime',
}

def get_runtime(c: Context) -> Runtime:
    """
    Factory function to return the correct Runtime subclass instance.

    Args:
        c (Context): runtime context

    Returns:
        Runtime: Corresponding runtime subclass instance
    """
    if c.config.io_mode.lower() not in registry.keys():
        raise NotImplementedError(f"Unsupported io_mode '{c.config.io_mode}'.")
    module = importlib.import_module('NEDAS.runtimes.'+c.config.io_mode)
    RuntimeClass = getattr(module, registry[c.config.io_mode])

    return RuntimeClass(c)

__all__ = ['registry', 'get_runtime']
