"""Common interface functions for `symmray` array objects."""

import autoray as ar


def conj(x, **kwargs):
    """Conjugate a `symmray` array."""
    return x.conj(**kwargs)


def max(x):
    """Return the maximum value of a `symmray` array."""
    try:
        return x.max()
    except AttributeError:
        # called on non symmray array
        return ar.do("max", x)


def min(x):
    """Return the minimum value of a `symmray` array."""
    try:
        return x.min()
    except AttributeError:
        # called on non symmray array
        return ar.do("min", x)


def sum(x):
    """Return the sum of a `symmray` array."""
    try:
        return x.sum()
    except AttributeError:
        # called on non symmray array
        return ar.do("sum", x)


def all(x):
    """Check if all elements of a `symmray` array are true."""
    try:
        return x.all()
    except AttributeError:
        # called on non symmray array
        return ar.do("all", x)


def any(x):
    """Check if any elements of a `symmray` array are true."""
    try:
        return x.any()
    except AttributeError:
        # called on non symmray array
        return ar.do("any", x)


def isfinite(x):
    """Check if a `symmray` array contains only finite values."""
    try:
        return x.isfinite()
    except AttributeError:
        # called on non symmray array
        return ar.do("isfinite", x)


def abs(x):
    """Return the absolute value of a `symmray` array."""
    try:
        return x.abs()
    except AttributeError:
        # called on non symmray array
        return ar.do("abs", x)


def sqrt(x):
    """Return the square root of a `symmray` array."""
    try:
        return x.sqrt()
    except AttributeError:
        # called on non symmray array
        return ar.do("sqrt", x)


def log(x):
    """Return the natural logarithm of a `symmray` array."""
    try:
        return x.log()
    except AttributeError:
        # called on non symmray array
        return ar.do("log", x)


def log2(x):
    """Return the base-2 logarithm of a `symmray` array."""
    try:
        return x.log2()
    except AttributeError:
        # called on non symmray array
        return ar.do("log2", x)


def log10(x):
    """Return the base-10 logarithm of a `symmray` array."""
    try:
        return x.log10()
    except AttributeError:
        # called on non symmray array
        return ar.do("log10", x)


def real(x):
    """Return the real part of a `symmray` array."""
    try:
        return x.real
    except AttributeError:
        # called on non symmray array
        return ar.do("real", x)


def imag(x):
    """Return the imaginary part of a `symmray` array."""
    try:
        return x.imag
    except AttributeError:
        # called on non symmray array
        return ar.do("imag", x)


def clip(x, a_min, a_max):
    """Clip the values of a `symmray` array."""
    return x.clip(a_min, a_max)


def squeeze(x, axis=None):
    """Remove single-dimensional entries from the shape of a `symmray` array."""
    return x.squeeze(axis)


def expand_dims(x, axis):
    """Expand the shape of a `symmray` array."""
    return x.expand_dims(axis)


def reshape(a, newshape, **kwargs):
    """Reshape a `symmray` array, via fusing or unfusing."""
    return a.reshape(newshape, **kwargs)


def tensordot(a, b, axes=2, **kwargs):
    """Contract two `symmray` arrays along the specified axes.

    Parameters
    ----------
    a : AbelianArray or FermionicArray
        First array to contract.
    b : AbelianArray or FermionicArray
        Second array to contract, with same type as `a`.
    axes : int or tuple of int, optional
        If an integer, the number of axes to contract. If a tuple, the axes
        to contract. Default is 2.
    """
    try:
        return a.tensordot(b, axes, **kwargs)
    except AttributeError:
        if getattr(a, "ndim", 0) == 0:
            # likely called as effective scalar multiplication of block array
            return a * b
        else:
            raise TypeError(f"Expected SymmrayCommon, got {type(a).__name__}.")


def einsum(eq, x):
    """Perform an Einstein summation on a `symmray` array."""
    return x.einsum(eq)


def transpose(a, axes=None, **kwargs):
    """Transpose a `symmray` array."""
    return a.transpose(axes, **kwargs)


def trace(a):
    """Return the trace of a `symmray` array."""
    return a.trace()


# non-standard 'composed' functions


def multiply_diagonal(x, v, axis):
    """Multiply a `symmray` array by a vector as if contracting a diagonal
    matrix into one axis.
    """
    return x.multiply_diagonal(v, axis)


ar.register_function("symmray", "multiply_diagonal", multiply_diagonal)


def ldmul(v, x):
    """Left multiply a `symmray` matrix `x` by a vector `v`."""
    return x.ldmul(v)


ar.register_function("symmray", "ldmul", ldmul)


def rdmul(x, v):
    """Right multiply a `symmray` matrix `x` by a vector `v`."""
    return x.rdmul(v)


ar.register_function("symmray", "rdmul", rdmul)


def lddiv(v, x):
    """Left divide a `symmray` matrix `x` by a vector `v`."""
    return x.lddiv(v)


ar.register_function("symmray", "lddiv", lddiv)


def rddiv(x, v):
    """Right divide a `symmray` matrix `x` by a vector `v`."""
    return x.rddiv(v)


ar.register_function("symmray", "rddiv", rddiv)


def align_axes(x, y, axes):
    """ """
    return x.align_axes(y, axes)


ar.register_function("symmray", "align_axes", align_axes)


def fuse(x, *axes_groups):
    """Fuse multiple axes of a `symmray` array."""
    return x.fuse(*axes_groups)


ar.register_function("symmray", "fuse", fuse)
