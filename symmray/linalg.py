"""Interface to linear algebra submodule functions."""

from .sparse_linalg import (
    eigh,
    eigh_truncated,
    norm,
    qr,
    qr_stabilized,
    solve,
    svd,
    svd_truncated,
)

__all__ = (
    "eigh_truncated",
    "eigh",
    "norm",
    "qr_stabilized",
    "qr",
    "solve",
    "svd_truncated",
    "svd",
)
