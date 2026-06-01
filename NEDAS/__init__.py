try:
    from ._version import version as __version__
except ImportError:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version("NEDAS")
    except PackageNotFoundError:
        __version__ = "unknown"

from .schemes import get_scheme

__all__ = ['get_scheme', '__version__']
