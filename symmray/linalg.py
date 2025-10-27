"""Interface to linear algebra submodule functions."""

import autoray as ar


def eigh(x, *args, **kwargs):
    """Hermitian eigen-decomposition of an assumed hermitian symmray array.

    Returns
    -------
    w : VectorCommon
        The eigenvalues as a vector.
    u : AbelianCommon
        The array of eigenvectors.
    """
    return x.eigh(*args, **kwargs)


def eigh_truncated(x, *args, **kwargs):
    """Truncated hermitian eigen-decomposition of an assumed hermitian symmray
    array.

    Parameters
    ----------
    cutoff : float, optional
        Absolute eigenvalue cutoff threshold.
    cutoff_mode : int or str, optional
        How to perform the truncation:

        - 1 or 'abs': trim values below ``cutoff``
        - 2 or 'rel': trim values below ``s[0] * cutoff``
        - 3 or 'sum2': trim s.t. ``sum(s_trim**2) < cutoff``.
        - 4 or 'rsum2': trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
        - 5 or 'sum1': trim s.t. ``sum(s_trim**1) < cutoff``.
        - 6 or 'rsum1': trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}
        How to absorb the eigenvalues.

        - -1 or 'left': absorb into the left factor (U).
        - 0 or 'both': absorb the square root into both factors.
        - 1 or 'right': absorb into the right factor (VH).
        - None: do not absorb, return eigenvalues as a BlockVector.

    renorm : {0, 1}
        Whether to renormalize the eigenvalues (depends on `cutoff_mode`).

    Returns
    -------
    u : AbelianCommon
        The abelian array of left eigenvectors.
    w : VectorCommon or None
        The vector of eigenvalues, or None if absorbed.
    uh : AbelianCommon
        The abelian array of right eigenvectors.
    """
    return x.eigh_truncated(*args, **kwargs)


def norm(x, *args, **kwargs):
    return x.norm(*args, **kwargs)


def qr(x, *args, **kwargs):
    return x.qr(*args, **kwargs)


def qr_stabilized(x, *args, **kwargs):
    q, r = x.qr(*args, stabilized=True, **kwargs)
    return q, None, r


def solve(x, *args, **kwargs):
    return x.solve(*args, **kwargs)


def svd(x, *args, **kwargs):
    return x.svd(*args, **kwargs)


def svd_truncated(x, *args, **kwargs):
    """Truncated singular value decomposition of a symmray array.

    Parameters
    ----------
    cutoff : float, optional
        Singular value cutoff threshold.
    cutoff_mode : int or str, optional
        How to perform the truncation:

        - 1 or 'abs': trim values below ``cutoff``
        - 2 or 'rel': trim values below ``s[0] * cutoff``
        - 3 or 'sum2': trim s.t. ``sum(s_trim**2) < cutoff``.
        - 4 or 'rsum2': trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
        - 5 or 'sum1': trim s.t. ``sum(s_trim**1) < cutoff``.
        - 6 or 'rsum1': trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}
        How to absorb the singular values.

        - -1 or 'left': absorb into the left factor (U).
        - 0 or 'both': absorb the square root into both factors.
        - 1 or 'right': absorb into the right factor (VH).
        - None: do not absorb, return singular values as a BlockVector.

    renorm : {0, 1}
        Whether to renormalize the singular values (depends on `cutoff_mode`).

    Returns
    -------
    u : AbelianCommon
        The abelian array of left singular vectors.
    s : VectorCommon or None
        The vector of singular values, or None if absorbed.
    vh : AbelianCommon
        The abelian array of right singular vectors.
    """
    return x.svd_truncated(*args, **kwargs)


# used by quimb
ar.register_function("symmray", "eigh_truncated", eigh_truncated)
ar.register_function("symmray", "qr_stabilized", qr_stabilized)
ar.register_function("symmray", "svd_truncated", svd_truncated)
