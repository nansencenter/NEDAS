try:
    from numba import njit

except ImportError:
    print("Warning: numba is not found in your environment, will skip the njit precompiling of functions.", flush=True)

    ##define njit as en empty decorator
    from functools import wraps

    def njit(*njit_args, **njit_kwargs):
        #@njit() - decorator with arguments
        if len(njit_args) == 0:
            def decorator(func):
                @wraps(func)
                def wrapper(*func_args, **func_kwargs):
                    return func(*func_args, **func_kwargs)
                return wrapper
            return decorator

        #@njit - decorator without arguments
        elif callable(njit_args[0]):
            func = njit_args[0]
            @wraps(func)
            def wrapper(*func_args, **func_kwargs):
                return func(*func_args, **func_kwargs)
            return wrapper

