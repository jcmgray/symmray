"""Interface to linear algebra submodule functions."""

import autoray as ar

from .array_common import ArrayCommon
from .vector_common import VectorCommon


def eigh(x: ArrayCommon, *args, **kwargs):
    """Hermitian eigen-decomposition of an assumed hermitian symmray array.

    Returns
    -------
    w : VectorCommon
        The eigenvalues as a vector.
    u : AbelianCommon
        The array of eigenvectors.
    """
    return x.eigh(*args, **kwargs)


def eigh_truncated(x: ArrayCommon, *args, **kwargs):
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


def norm(x: ArrayCommon, *args, **kwargs):
    return x.norm(*args, **kwargs)


def qr(x: ArrayCommon, *args, **kwargs):
    """QR decomposition of a symmray array."""
    kwargs.setdefault("stabilized", False)
    return x.qr(*args, **kwargs)


def qr_stabilized(x: ArrayCommon, *args, **kwargs):
    """Stabilized QR decomposition of a symmray array, returning results in an
    SVD-like ``(Q, None, R)`` format for compatibility with tensor network
    split drivers.
    """
    kwargs.setdefault("stabilized", True)
    q, r = x.qr(*args, **kwargs)
    return q, None, r


def lq(x: ArrayCommon, *args, **kwargs):
    """LQ decomposition of a symmray array."""
    kwargs.setdefault("stabilized", False)
    return x.lq(*args, **kwargs)


def lq_stabilized(x: ArrayCommon, *args, **kwargs):
    """Stabilized LQ decomposition of a symmray array, returning results in an
    SVD-like ``(L, None, Q)`` format for compatibility with tensor network
    split drivers.
    """
    kwargs.setdefault("stabilized", True)
    l, q = x.lq(*args, **kwargs)
    return l, None, q


def solve(x: ArrayCommon, *args, **kwargs):
    return x.solve(*args, **kwargs)


def svd(
    x: ArrayCommon, *args, **kwargs
) -> tuple[ArrayCommon, VectorCommon, ArrayCommon]:
    """Singular value decomposition of a symmray array, returning all singular
    values and vectors without truncation.
    """
    return x.svd(*args, **kwargs)


def svd_truncated(x: ArrayCommon, *args, **kwargs):
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


def svd_rand_truncated(x: ArrayCommon, *args, **kwargs):
    """Truncated singular value decomposition of a symmray array, using
    randomized projection. This is efficient for low-rank approximations when a
    target ``max_bond`` is known.

    Parameters
    ----------
    max_bond : int
        Target rank / maximum bond dimension.
    absorb : {-1, 0, 1, None}
        How to absorb the singular values.

        - -1 or 'left': absorb into the left factor (U).
        - 0 or 'both': absorb the square root into both factors.
        - 1 or 'right': absorb into the right factor (VH).
        - None: do not absorb, return singular values.

    oversample : int, optional
        Extra sketch dimensions for accuracy. Default is 10.
    num_iterations : int, optional
        Number of power iterations for accuracy. Default is 2.
    seed : int, Generator or None, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    u : AbelianCommon or None
        The array of left singular vectors.
    s : VectorCommon or None
        The singular values, or None if absorbed.
    vh : AbelianCommon or None
        The array of right singular vectors.
    """
    return x.svd_rand_truncated(*args, **kwargs)


def svd_via_eig_truncated(x: ArrayCommon, *args, **kwargs):
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


def cholesky(x: ArrayCommon, *args, **kwargs):
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


def cholesky_regularized(x: ArrayCommon, *args, **kwargs):
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
        Diagonal regularization shift. If True or negative, auto-compute
        from dtype machine epsilon. The shift is always applied as a
        relative shift scaled by the trace of each block. Default is True.

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


def lq_via_cholesky(x: ArrayCommon, *args, **kwargs):
    """LQ decomposition via Cholesky factorization of ``x @ x^H``.

    Computes ``x = L @ Q`` where ``L`` is lower triangular and ``Q`` is
    isometric.

    Parameters
    ----------
    x : AbelianCommon
        The 2D block-symmetric array to decompose.
    absorb : {-1, -10, -11} or str, optional
        How to return the factors:

        - ``-1`` (``'left'``): return ``(L, None, Q)``.
        - ``-10`` (``'lfactor'``): return ``(L, None, None)``.
        - ``-11`` (``'rorthog'``): return ``(None, None, Q)``.

    shift : float, optional
        Diagonal regularization shift. If True or negative, auto-compute
        from dtype machine epsilon. The shift is always applied as a
        relative shift scaled by the trace of each block. Default is True.
    solve_triangular : bool, optional
        Whether to use triangular solve (faster) or general solve to
        compute Q. Default is True.

    Returns
    -------
    L : AbelianCommon or None
        The lower triangular factor.
    s : None
        Always None.
    Q : AbelianCommon or None
        The isometric factor.
    """
    return x.lq_via_cholesky(*args, **kwargs)


def qr_via_cholesky(x: ArrayCommon, *args, **kwargs):
    """QR decomposition via Cholesky factorization of ``x^H @ x``.

    Computes ``x = Q @ R`` where ``Q`` is isometric and ``R`` is upper
    triangular.

    Parameters
    ----------
    x : AbelianCommon
        The 2D block-symmetric array to decompose.
    absorb : {1, 11, 10} or str, optional
        How to return the factors:

        - ``1`` (``'right'``): return ``(Q, None, R)``.
        - ``11`` (``'rfactor'``): return ``(None, None, R)``.
        - ``10`` (``'lorthog'``): return ``(Q, None, None)``.

    shift : float, optional
        Diagonal regularization shift. If True or negative, auto-compute
        from dtype machine epsilon. The shift is always applied as a
        relative shift scaled by the trace of each block. Default is True.
    solve_triangular : bool, optional
        Whether to use triangular solve (faster) or general solve.
        Default is True.

    Returns
    -------
    Q : AbelianCommon or None
        The isometric factor.
    s : None
        Always None.
    R : AbelianCommon or None
        The upper triangular factor.
    """
    return x.qr_via_cholesky(*args, **kwargs)


# used by quimb
ar.register_function("symmray", "eigh_truncated", eigh_truncated)
ar.register_function("symmray", "qr_stabilized", qr_stabilized)
ar.register_function("symmray", "lq_stabilized", lq_stabilized)
ar.register_function("symmray", "svd_truncated", svd_truncated)
ar.register_function("symmray", "svd_rand_truncated", svd_rand_truncated)
ar.register_function("symmray", "svd_via_eig_truncated", svd_via_eig_truncated)
ar.register_function("symmray", "cholesky_regularized", cholesky_regularized)
ar.register_function("symmray", "lq_via_cholesky", lq_via_cholesky)
ar.register_function("symmray", "qr_via_cholesky", qr_via_cholesky)
