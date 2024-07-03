"""Common interface functions for `symmray` array objects."""

import functools

import autoray as ar


def conj(x, **kwargs):
    """Conjugate a `symmray` array."""
    return x.conj(**kwargs)


def max(x):
    """Return the maximum value of a `symmray` array."""
    return x.max()


def min(x):
    """Return the minimum value of a `symmray` array."""
    return x.min()


def sum(x):
    """Return the sum of a `symmray` array."""
    return x.sum()


def all(x):
    """Check if all elements of a `symmray` array are true."""
    return x.all()


def any(x):
    """Check if any elements of a `symmray` array are true."""
    return x.any()


def isfinite(x):
    """Check if a `symmray` array contains only finite values."""
    return x.isfinite()


def abs(x):
    """Return the absolute value of a `symmray` array."""
    return x.abs()


def reshape(a, newshape, **kwargs):
    """Reshape a `symmray` array, via fusing or unfusing."""
    return a.reshape(newshape, **kwargs)


@functools.singledispatch
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
    raise NotImplementedError(
        f"Not implemented for types {type(a)} and {type(b)}."
    )


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


def align_axes(x, y, axes):
    """ """
    return x.align_axes(y, axes)


ar.register_function("symmray", "align_axes", align_axes)


def fuse(x, *axes_groups):
    """Fuse multiple axes of a `symmray` array."""
    return x.fuse(*axes_groups)


ar.register_function("symmray", "fuse", fuse)
