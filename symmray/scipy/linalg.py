import autoray as ar


def expm(x):
    """Matrix exponential of a AbelianArray."""
    if x.ndim != 2:
        raise NotImplementedError(
            "expm only implemented for 2D AbelianArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )
    _expm = ar.get_lib_fn(x.backend, "scipy.linalg.expm")
    new = x.copy()
    new.apply_to_arrays(_expm)
    return new


# register non-scipy version
ar.register_function("symmray", "linalg.expm", expm)
