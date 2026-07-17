"""Functional interface for `symmray` array objects."""

import functools

import autoray as ar
import cotengra as ctg


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
    """Remove single-dimensional entries from shape of a `symmray` array."""
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


@functools.lru_cache(2**12)
def _parse_tensordot_eq(eq):
    """Try to convert the two term einsum `eq` into tensordot axes and a
    final output permutation.

    Returns None if not possible, i.e. if the eq features batched, summed,
    or repeated indices, else a tuple `(axes_a, axes_b, perm)`, where `perm`
    is a final permutation of the tensordot output axes, or None if not
    needed.
    """
    lhs, out = eq.split("->")
    left, right = lhs.split(",")

    if (
        len(set(left)) != len(left)
        or len(set(right)) != len(right)
        or len(set(out)) != len(out)
    ):
        # repeated indices
        return None

    sleft, sright, sout = set(left), set(right), set(out)
    if sout != sleft ^ sright:
        # summed (missing from output) or batched (on all three) indices
        return None

    axes_a = tuple(i for i, q in enumerate(left) if q in sright)
    axes_b = tuple(right.index(left[i]) for i in axes_a)

    # tensordot output is kept `a` indices then kept `b` indices
    td_out = [q for q in left if q not in sright]
    td_out.extend(q for q in right if q not in sleft)
    perm = tuple(map(td_out.index, out))
    if perm == tuple(range(len(perm))):
        perm = None

    return axes_a, axes_b, perm


@functools.lru_cache(2**12)
def _parse_multiply_diagonal_eq(eq):
    """Check whether the two term einsum `eq` is a diagonal multiplication
    pattern, e.g. "i,ijkl->ijkl", suitable for `multiply_diagonal`.

    Returns None if it is not, else a tuple `(which, axis, perm)`, where
    `which` is "left", "right", or "both" depending on which term(s) are
    vector-like, `axis` is the axis of the array term that the vector
    multiplies into, and `perm` is a final permutation of the array term
    axes, or None if not needed.
    """
    lhs, out = eq.split("->")
    left, right = lhs.split(",")

    if len(left) == 1 and len(right) == 1:
        if left == right == out:
            # e.g. "i,i->i"
            return "both", 0, None
        return None
    elif len(left) == 1:
        which, tv, tx = "left", left, right
    elif len(right) == 1:
        which, tv, tx = "right", right, left
    else:
        return None

    if len(set(tx)) != len(tx) or tv not in tx or sorted(out) != sorted(tx):
        return None

    axis = tx.index(tv)
    perm = None if out == tx else tuple(map(tx.index, out))
    return which, axis, perm


def einsum(*args, **kwargs):
    """Perform an Einstein summation on a `symmray` array, this simply uses
    `cotengra` to dispatch the full expression into pairwise tensordot (or
    einsum if necessary) calls.
    """
    if not isinstance(args[0], str):
        # convert from interleaved
        eq, arrays = ctg.utils.convert_from_interleaved(args)
    else:
        eq, *arrays = args

    if len(arrays) == 1:
        # use symmray for single term
        return arrays[0].einsum(eq, **kwargs)

    if len(arrays) == 2:
        from .vector_common import VectorCommon

        a, b = arrays
        a_isvec = isinstance(a, VectorCommon)
        b_isvec = isinstance(b, VectorCommon)
        if a_isvec or b_isvec:
            # only allow diagonal multiplication
            info = _parse_multiply_diagonal_eq(eq)
            if info is None:
                raise NotImplementedError(
                    f"einsum eq {eq!r} with a symmray vector operand is only "
                    "allowed for diagonal multiplications like 'i,ijkl->ijkl'."
                )

            which, axis, perm = info
            if a_isvec and b_isvec:
                # blockwise vector multiplication, e.g. "i,i->i"
                return a * b
            elif a_isvec and which != "right":
                v, x = a, b
            else:
                v, x = b, a

            # vector acts as a diagonal matrix on one array axis
            x = x.multiply_diagonal(v, axis=axis)
            if perm is not None:
                x = x.transpose(perm)
            return x

        # both proper arrays: dispatch directly to tensordot
        info = _parse_tensordot_eq(eq)
        if info is None:
            raise NotImplementedError(
                f"einsum eq {eq!r} for two symmray arrays is only supported "
                "for pure pairwise contractions, i.e. without batched, summed,"
                " or repeated indices."
            )

        axes_a, axes_b, perm = info
        x = tensordot(a, b, axes=(axes_a, axes_b))
        if perm is not None:
            x = x.transpose(perm)
        return x

    # else dispatch >2 terms to pairwise contractions using cotengra
    return ctg.einsum(eq, *arrays, **kwargs)


def transpose(a, axes=None, **kwargs):
    """Transpose a `symmray` array."""
    return a.transpose(axes, **kwargs)


def trace(a):
    """Return the trace of a `symmray` array."""
    return a.trace()


def take(a, indices, axis, **kwargs):
    """Take elements from a `symmray` array along an axis."""
    return a.take(indices, axis, **kwargs)


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
