"""Interface to linear algebra submodule functions."""

import functools

import autoray as ar


@functools.singledispatch
def eigh(x, *args, **kwargs):
    raise NotImplementedError("eigh is not implemented for this type.")


@functools.singledispatch
def eigh_truncated(x, *args, **kwargs):
    raise NotImplementedError(
        "eigh_truncated is not implemented for this type."
    )


@functools.singledispatch
def norm(x, *args, **kwargs):
    return x.norm(*args, **kwargs)


@functools.singledispatch
def qr(x, *args, **kwargs):
    raise NotImplementedError("qr is not implemented for this type.")


@functools.singledispatch
def qr_stabilized(x, *args, **kwargs):
    raise NotImplementedError(
        "qr_stabilized is not implemented for this type."
    )


@functools.singledispatch
def solve(x, *args, **kwargs):
    raise NotImplementedError("solve is not implemented for this type.")


@functools.singledispatch
def svd(x, *args, **kwargs):
    raise NotImplementedError("svd is not implemented for this type.")


@functools.singledispatch
def svd_truncated(x, *args, **kwargs):
    raise NotImplementedError(
        "svd_truncated is not implemented for this type."
    )


# used by quimb
ar.register_function("symmray", "eigh_truncated", eigh_truncated)
ar.register_function("symmray", "qr_stabilized", qr_stabilized)
ar.register_function("symmray", "svd_truncated", svd_truncated)
