import autoray as ar


def expm(x):
    """Matrix exponential of a 2D symmray array, block by block."""
    if x.ndim != 2:
        raise NotImplementedError(
            "expm only implemented for 2D AbelianArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )
    _expm = ar.get_lib_fn(x.backend, "scipy.linalg.expm")
    # expm is not linear, so lazy fermionic phases must be applied first
    new = x.phase_sync() if x.fermionic else x.copy()
    # with a |x><x| inner index every matrix product flips odd sectors.
    # Writing P for this flip, exp(A) = P expm(P A) blockwise
    flip = x.fermionic and not x.indices[1].dual
    if flip:
        new.phase_flip(1, inplace=True)
        new.phase_sync(inplace=True)
    new.apply_to_blocks(_expm)
    if flip:
        new.phase_flip(1, inplace=True)
    return new


# register non-scipy version
ar.register_function("symmray", "linalg.expm", expm)
