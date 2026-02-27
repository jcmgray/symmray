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


def svd_via_eig_truncated(x, *args, **kwargs):
    """Truncated singular value decomposition of a symmray array, using
    eigen-decomposition of the gram (xdag @ x or x @ xdag) matrix. This can
    be faster, but also incur a loss of precision due to the squaring.

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
    return x.svd_via_eig_truncated(*args, **kwargs)


def cholesky(x, *args, **kwargs):
    """Cholesky decomposition of an assumed positive-definite symmray array.

    Parameters
    ----------
    x : AbelianCommon
        The 2D block-symmetric array to decompose.
    upper : bool, optional
        Whether to return the upper triangular Cholesky factor.
        Default is False, returning the lower triangular factor.

    Returns
    -------
    l_or_r : AbelianCommon
        The Cholesky factor. Lower triangular if ``upper=False``,
        upper triangular if ``upper=True``.
    """
    return x.cholesky(*args, **kwargs)


def cholesky_regularized(x, *args, **kwargs):
    """Cholesky decomposition with optional diagonal regularization,
    returning results in an SVD-like ``(left, None, right)`` format
    for compatibility with tensor network split drivers.

    Parameters
    ----------
    x : AbelianCommon
        The 2D block-symmetric array to decompose. Must be positive
        (semi-)definite.
    absorb : {-12, 0, 12}, optional
        How to return the factors:

        - ``0`` (``'both'``): return ``(L, None, L^H)``.
        - ``-12`` (``'lsqrt'``): return ``(L, None, None)``.
        - ``12`` (``'rsqrt'``): return ``(None, None, L^H)``.

    shift : float, optional
        Diagonal regularization shift. If negative, auto-compute
        proportional to dtype machine epsilon. If positive, take as
        relative shift to the trace of each block. Default is -1.0
        (auto-compute).

    Returns
    -------
    left : AbelianCommon or None
        The lower Cholesky factor, or None.
    s : None
        Always None (no singular values).
    right : AbelianCommon or None
        The conjugate transpose of the Cholesky factor, or None.
    """
    return x.cholesky_regularized(*args, **kwargs)


# used by quimb
ar.register_function("symmray", "eigh_truncated", eigh_truncated)
ar.register_function("symmray", "qr_stabilized", qr_stabilized)
ar.register_function("symmray", "svd_truncated", svd_truncated)
ar.register_function("symmray", "svd_via_eig_truncated", svd_via_eig_truncated)
ar.register_function("symmray", "cholesky_regularized", cholesky_regularized)
