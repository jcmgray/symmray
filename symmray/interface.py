"""Common interface functions for `symmray` array objects."""

import functools


def conj(x, **kwargs):
    """Conjugate a `symmray` array."""
    return x.conj(**kwargs)


def reshape(a, newshape, **kwargs):
    """Reshape a `symmray` array, via fusing or unfusing."""
    return a.reshape(newshape, **kwargs)


@functools.singledispatch
def tensordot(a, b, axes=2, **kwargs):
    """Contract two `symmray` arrays along the specified axes.

    Parameters
    ----------
    a : BlockArray or FermionicArray
        First array to contract.
    b : BlockArray or FermionicArray
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
