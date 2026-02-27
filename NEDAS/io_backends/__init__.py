from NEDAS.core import Context, IOBackend

def get_io_backend(c: Context) -> IOBackend:
    """
    Factory function to return the correct IOBackend subclass instance.

    Args:
        c (Context): the runtime context

    Returns:
        IOBackend: Corresponding io backend subclass instance
    """
    cf = c.config
    if cf.io_mode == 'offline':
        from NEDAS.io_backends.file_io import FileIO
        return FileIO(c)

    elif cf.io_mode == 'online':
        from NEDAS.io_backends.memory_io import MemoryIO
        return MemoryIO(c)

    else:
        raise ValueError(f"Unsupported io_mode '{cf.io_mode}', only 'online' or 'offline'.")

__all__ = ['get_io_backend']