try:
    from numba import njit

except ImportError:
    print("Warning: numba is not found in your environment, will skip the njit precompiling of functions.")
    ##define njit as en empty decorator
    from functools import wraps
    def njit(cache=True):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator

