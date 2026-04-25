"""Common linear algebra utilities shared across backends."""

import functools

import autoray as ar


@functools.cache
def get_quimb_array_split():
    try:
        from quimb.tensor.decomp import array_split

        return array_split
    except ImportError:
        return None


def array_split(*args, **kwargs):
    # NOTE: wrap for three reasons:
    # 1) avoid hard dependency and eager import
    # 2) might want to provide fallback implementations of some methods
    # 3) centralized place to add any informational warnings or errors
    return get_quimb_array_split()(*args, **kwargs)


class Absorb:
    """Absorb mode constants and parsing for SVD-like decompositions.

    Attributes give canonical integer codes for each mode. Use
    ``Absorb.parse`` to normalize user-facing string or integer aliases
    to the canonical code.

    Modes
    -----
    U_s_VH : None
        Return all three factors unmodified ('full').
    s : 2
        Return only the singular values ('svals').
    Usq : -12
        Absorb sqrt(s) into U, return (U√s, None, None) ('lsqrt').
    VH : -11
        Return only VH ('rorthog').
    Us : -10
        Absorb s into U, return (Us, None, None) ('lfactor').
    Us_VH : -1
        Absorb s into U, return (Us, None, VH) ('left').
    Usq_sqVH : 0
        Absorb sqrt(s) into both, return (U√s, None, √sVH) ('both').
    U_sVH : 1
        Absorb s into VH, return (U, None, sVH) ('right').
    U : 10
        Return only U ('lorthog').
    sVH : 11
        Absorb s into VH, return (None, None, sVH) ('rfactor').
    sqVH : 12
        Absorb sqrt(s) into VH, return (None, None, √sVH) ('rsqrt').
    """

    U_s_VH = None  # 'full'
    s = 2  # 'svals'
    Usq = -12  # 'lsqrt'
    VH = -11  # 'rorthog'
    Us = -10  # 'lfactor'
    Us_VH = -1  # 'left'
    Usq_sqVH = 0  # 'both'
    U_sVH = 1  # 'right'
    U = 10  # 'lorthog'
    sVH = 11  # 'rfactor'
    sqVH = 12  # 'rsqrt'

    _map = {
        U_s_VH: U_s_VH,
        "U,s,VH": U_s_VH,
        s: s,
        "s": s,
        Usq: Usq,
        "lsqrt": Usq,
        VH: VH,
        "VH": VH,
        "rorthog": VH,
        Us: Us,
        "Us": Us,
        "lfactor": Us,
        Us_VH: Us_VH,
        "Us,VH": Us_VH,
        "left": Us_VH,
        Usq_sqVH: Usq_sqVH,
        "Usq,sqVH": Usq_sqVH,
        "both": Usq_sqVH,
        U_sVH: U_sVH,
        "U,sVH": U_sVH,
        "right": U_sVH,
        U: U,
        "U": U,
        "lorthog": U,
        sVH: sVH,
        "sVH": sVH,
        "rfactor": sVH,
        sqVH: sqVH,
        "sqVH": sqVH,
        "rsqrt": sqVH,
    }

    _transpose_map = {
        Us_VH: U_sVH,
        U_sVH: Us_VH,
        Us: sVH,
        sVH: Us,
        VH: U,
        U: VH,
        Usq: sqVH,
        sqVH: Usq,
        U_s_VH: U_s_VH,
        s: s,
        Usq_sqVH: Usq_sqVH,
    }

    @classmethod
    def parse(cls, absorb):
        """Normalize a user-facing absorb value to a canonical mode code.

        Parameters
        ----------
        absorb : int, str, or None
            Any valid absorb specification: a canonical integer code,
            a string alias (e.g. ``'left'``, ``'both'``, ``'rfactor'``),
            or ``None`` for full (no absorption).

        Returns
        -------
        int or None
            The canonical absorb mode code.

        Raises
        ------
        KeyError
            If ``absorb`` is not a recognized mode or alias.
        """
        return cls._map[absorb]

    @classmethod
    def explain(cls, absorb):
        """Get a human-readable explanation of an absorb mode.

        Parameters
        ----------
        absorb : int or None
            A canonical absorb mode code (already parsed).

        Returns
        -------
        str
            A human-readable explanation of the mode.
        """
        explanations = {
            cls.U_s_VH: "return U, s, VH",
            cls.s: "return s only",
            cls.Usq: "absorb sqrt(s) into U, return U√s only",
            cls.VH: "return VH only",
            cls.Us: "absorb s into U, return Us only",
            cls.Us_VH: "absorb s into U, return Us and VH",
            cls.Usq_sqVH: "absorb sqrt(s) into both, return U√s and √sVH",
            cls.U_sVH: "absorb s into VH, return U and sVH",
            cls.U: "return U only",
            cls.sVH: "absorb s into VH, return sVH only",
            cls.sqVH: "absorb sqrt(s) into VH, return √sVH only",
        }
        return explanations[absorb]

    @classmethod
    def transpose(cls, absorb):
        """Map an absorb mode to its transposed equivalent.

        Swaps left/right roles: e.g. ``U_sVH`` ('right') becomes
        ``Us_VH`` ('left'), ``sVH`` ('rfactor') becomes ``Us``
        ('lfactor'), etc. Symmetric modes (``both``, ``full``,
        ``svals``) are unchanged.

        Parameters
        ----------
        absorb : int or None
            A canonical absorb mode code (already parsed).

        Returns
        -------
        int or None
            The transposed absorb mode code.
        """
        return cls._transpose_map[absorb]

    @staticmethod
    @functools.cache
    def choose_charge_side(absorb, charge_side="auto"):

        absorb = Absorb.parse(absorb)

        if charge_side not in ("auto", "left", "right"):
            raise ValueError(f"Invalid charge_side: {charge_side}")

        if charge_side == "auto":
            # we prefer keeping the charge on the isometric factor, as the
            # reduced factor is more likely to be passed around as a gauge
            if absorb in (Absorb.Us_VH, Absorb.Us, Absorb.VH):
                charge_side = "right"
            else:
                # for left isometric and everything else, keep charge on left
                charge_side = "left"

        return charge_side


def absorb_svd_result(U, s, VH, absorb):
    """Apply absorption of singular values into U and/or VH.

    Works on any symmray array objects that support ``multiply_diagonal`` and
    vectors that support ``.sqrt()``.

    Parameters
    ----------
    U : SymmrayCommon
        Left singular vectors.
    s : VectorCommon
        Singular values.
    VH : SymmrayCommon
        Right singular vectors.
    absorb : int or None
        Absorption mode code (should already be parsed via ``Absorb.parse``).

    Returns
    -------
    U : SymmrayCommon or None
    s : VectorCommon or None
    VH : SymmrayCommon or None
    """
    if absorb is Absorb.U_s_VH:  # None - 'full'
        return U, s, VH
    if absorb == Absorb.s:
        return None, s, None
    if absorb == Absorb.U:
        return U, None, None
    if absorb == Absorb.VH:
        return None, None, VH
    if absorb == Absorb.Us_VH:
        U.multiply_diagonal(s, axis=1, inplace=True)
        return U, None, VH
    if absorb == Absorb.U_sVH:
        VH.multiply_diagonal(s, axis=0, inplace=True)
        return U, None, VH
    if absorb == Absorb.Usq_sqVH:
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        return U, None, VH
    if absorb == Absorb.Us:
        U.multiply_diagonal(s, axis=1, inplace=True)
        return U, None, None
    if absorb == Absorb.sVH:
        VH.multiply_diagonal(s, axis=0, inplace=True)
        return None, None, VH
    if absorb == Absorb.Usq:
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        return U, None, None
    if absorb == Absorb.sqVH:
        s_sqrt = s.sqrt()
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        return None, None, VH
    raise ValueError(f"Invalid absorb mode: {absorb}")


def blocklevel_svd_via_eig(
    x,
    absorb=Absorb.U_s_VH,
    max_bond=-1,
    descending=True,
    right=None,
):
    """SVD of ``(..., da, db)`` blocks via eigendecomposition of the
    Gram matrix, with static truncation and all absorb mode shortcuts.

    Supports arbitrary leading batch dimensions (including none).

    Parameters
    ----------
    x : array_like
        Input array with shape ``(..., da, db)``.
    absorb : int or None, optional
        Absorption mode code controlling what to compute / return.
    max_bond : int, optional
        Maximum bond dimension per block, use -1 for no truncation.
    descending : bool, optional
        Whether to return singular values in descending order.
    right : bool, optional
        Whether to eigendecompose ``xdag @ x`` (True) or ``x @ xdag``
        (False). If None, choose based on shape and absorb mode.

    Returns
    -------
    U : array_like or None
    s : array_like or None
    VH : array_like or None
    """
    xp = ar.get_namespace(x)

    shape = xp.shape(x)
    da, db = shape[-2], shape[-1]
    xdag = xp.conj(xp.swapaxes(x, -2, -1))

    if right is None:
        if da > db:
            right = True
        elif da < db:
            right = False
        else:
            # avoid division if possible
            right = absorb in (
                Absorb.VH,
                Absorb.sVH,
                Absorb.sqVH,
                Absorb.Us_VH,
            )

    if right:
        # tall: eigendecompose xdag @ x
        s2, V = xp.linalg.eigh(xdag @ x)
        if 0 < max_bond < min(da, db):
            s2 = s2[..., -max_bond:]
            V = V[..., :, -max_bond:]
        if descending:
            s2 = xp.flip(s2, axis=-1)
            V = xp.flip(V, axis=-1)
        s2 = xp.maximum(s2, 0.0)

        if absorb == Absorb.s:  # 'svals'
            s = xp.sqrt(s2)
            return None, s, None
        if absorb == Absorb.VH:  # 'rorthog'
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            return None, None, VH
        if absorb == Absorb.sVH:  # 'rfactor'
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            s = xp.sqrt(s2)
            sVH = s[..., :, None] * VH
            return None, None, sVH
        if absorb == Absorb.sqVH:  # 'rsqrt'
            sq = xp.sqrt(xp.sqrt(s2))
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            sqVH = sq[..., :, None] * VH
            return None, None, sqVH

        Us = x @ V
        if absorb == Absorb.Us:  # 'lfactor'
            return Us, None, None
        if absorb == Absorb.Us_VH:  # 'left'
            VH = xp.conj(xp.swapaxes(V, -2, -1))
            return Us, None, VH

        # for all other options we need U
        s = xp.sqrt(s2)
        eps = xp.finfo(s.dtype).eps
        cutoff = xp.max(s) * eps * max(da, db)
        sinv = s / (s**2 + cutoff**2)
        U = Us * sinv[..., None, :]

        if absorb == Absorb.U:  # 'lorthog'
            return U, None, None
        if absorb == Absorb.Usq:  # 'lsqrt'
            sq = xp.sqrt(s)
            Usq = U * sq[..., None, :]
            return Usq, None, None

        # need U and VH for all remaining options
        VH = xp.conj(xp.swapaxes(V, -2, -1))
        if absorb == Absorb.U_s_VH:  # 'full'
            return U, s, VH
        if absorb == Absorb.U_sVH:  # 'right'
            sVH = s[..., :, None] * VH
            return U, None, sVH
        if absorb == Absorb.Usq_sqVH:  # 'both'
            sq = xp.sqrt(s)
            Usq = U * sq[..., None, :]
            sqVH = sq[..., :, None] * VH
            return Usq, None, sqVH

    else:
        # wide: eigendecompose x @ xdag
        s2, U = xp.linalg.eigh(x @ xdag)
        if 0 < max_bond < min(da, db):
            s2 = s2[..., -max_bond:]
            U = U[..., :, -max_bond:]
        if descending:
            s2 = xp.flip(s2, axis=-1)
            U = xp.flip(U, axis=-1)
        s2 = xp.maximum(s2, 0.0)

        if absorb == Absorb.s:  # 'svals'
            s = xp.sqrt(s2)
            return None, s, None
        if absorb == Absorb.U:  # 'lorthog'
            return U, None, None
        if absorb == Absorb.Us:  # 'lfactor'
            s = xp.sqrt(s2)
            Us = U * s[..., None, :]
            return Us, None, None
        if absorb == Absorb.Usq:  # 'lsqrt'
            sq = xp.sqrt(xp.sqrt(s2))
            Usq = U * sq[..., None, :]
            return Usq, None, None

        sVH = xp.conj(xp.swapaxes(U, -2, -1)) @ x
        if absorb == Absorb.sVH:  # 'rfactor'
            return None, None, sVH
        if absorb == Absorb.U_sVH:  # 'right'
            return U, None, sVH

        # for all other options we need VH
        s = xp.sqrt(s2)
        eps = xp.finfo(s.dtype).eps
        cutoff = xp.max(s) * eps * max(da, db)
        sinv = s / (s**2 + cutoff**2)
        VH = sinv[..., :, None] * sVH

        if absorb == Absorb.VH:  # 'rorthog'
            return None, None, VH
        if absorb == Absorb.U_s_VH:  # 'full'
            return U, s, VH
        if absorb == Absorb.Us_VH:  # 'left'
            Us = U * s[..., None, :]
            return Us, None, VH
        sq = xp.sqrt(s)
        sqVH = sq[..., :, None] * VH
        if absorb == Absorb.Usq_sqVH:  # 'both'
            Usq = U * sq[..., None, :]
            return Usq, None, sqVH
        if absorb == Absorb.sqVH:  # 'rsqrt'
            return None, None, sqVH

    raise ValueError(f"Invalid absorb mode: {absorb}")


def blocklevel_cholesky_regularized(blocks, upper=False, shift=True):
    """Cholesky decomposition of symmetric positive-definite blocks
    with optional diagonal regularization.

    Works for a single 2D block ``(d, d)`` or batched blocks with
    arbitrary leading dimensions ``(..., d, d)``.

    Parameters
    ----------
    blocks : array_like
        Blocks of shape ``(..., d, d)``.
    upper : bool, optional
        Whether to return the upper triangular Cholesky factor.
        Default is False, returning the lower triangular factor.
    shift : float, optional
        Diagonal regularization shift. If True or negative, auto-compute
        from dtype machine epsilon. The shift is always applied as a
        relative shift scaled by the trace of each block. Default is
        True.

    Returns
    -------
    L_or_R : array_like
        The Cholesky factor, shape ``(..., d, d)``.
    """
    xp = ar.get_namespace(blocks)

    if shift is True:
        shift = -1.0

    if shift < 0.0:
        shift = xp.finfo(blocks.dtype).eps

    if shift > 0.0:
        trace = xp.linalg.trace(blocks)[..., None, None]
        try:
            trace = xp.stop_gradient(trace)
        except (ImportError, AttributeError):
            pass
        I = xp.eye(blocks.shape[-1])
        blocks = blocks + shift * trace * I

    return xp.linalg.cholesky(blocks, upper=upper)


def blocklevel_lq_via_cholesky(
    x,
    absorb=Absorb.Us_VH,
    shift=True,
    solve_triangular=True,
):
    """LQ decomposition of 2D blocks ``x`` via Cholesky factorization
    of the Gram matrix ``x @ x^H``.

    Computes ``x = L @ Q`` where ``L`` is lower triangular and ``Q``
    is isometric. Works for a single 2D block ``(da, db)`` or batched
    blocks with arbitrary leading dimensions ``(..., da, db)``.

    Parameters
    ----------
    x : array_like
        Blocks of shape ``(..., da, db)``.
    absorb : int or None, optional
        Absorption mode code controlling what to return:

        - ``Absorb.Us_VH`` (-1, 'left'): return ``(L, None, Q)``.
        - ``Absorb.Us`` (-10, 'lfactor'): return ``(L, None, None)``.
        - ``Absorb.VH`` (-11, 'rorthog'): return ``(None, None, Q)``.

    shift : float, optional
        Diagonal regularization shift. If True or negative, auto-compute
        from dtype machine epsilon. The shift is always applied as a
        relative shift scaled by the trace of each block. Default is
        True.
    solve_triangular : bool, optional
        Whether to use triangular solve (faster) or general solve
        to compute Q. Default is True.

    Returns
    -------
    L : array_like or None
        The lower triangular factor, shape ``(..., da, da)``.
    s : None
        Always None.
    Q : array_like or None
        The isometric factor, shape ``(..., da, db)``.
    """
    xp = ar.get_namespace(x)

    xdag = xp.conj(xp.swapaxes(x, -2, -1))
    xx = x @ xdag

    L = blocklevel_cholesky_regularized(xx, upper=False, shift=shift)

    if absorb == Absorb.Us:  # 'lfactor'
        return L, None, None

    if solve_triangular:
        Q = xp.scipy.linalg.solve_triangular(L, x, lower=True)
    else:
        Q = xp.linalg.solve(L, x)

    if absorb == Absorb.VH:  # 'rorthog'
        return None, None, Q

    # absorb == Absorb.Us_VH ('left') or fallback
    return L, None, Q


def blocklevel_qr_via_cholesky(
    x,
    absorb=Absorb.U_sVH,
    shift=True,
    solve_triangular=True,
):
    """QR decomposition of 2D blocks ``x`` via Cholesky factorization.
    Implemented by transposing to LQ at the block level.

    Computes ``x = Q @ R`` where ``Q`` is isometric and ``R`` is
    upper triangular. Works for a single 2D block ``(da, db)`` or
    batched blocks with arbitrary leading dimensions ``(..., da, db)``.

    Parameters
    ----------
    x : array_like
        Blocks of shape ``(..., da, db)``.
    absorb : int or None, optional
        Absorption mode code controlling what to return:

        - ``Absorb.U_sVH`` (1, 'right'): return ``(Q, None, R)``.
        - ``Absorb.sVH`` (11, 'rfactor'): return ``(None, None, R)``.
        - ``Absorb.U`` (10, 'lorthog'): return ``(Q, None, None)``.

    shift : float, optional
        Diagonal regularization shift. If True or negative, auto-compute
        from dtype machine epsilon. The shift is always applied as a
        relative shift scaled by the trace of each block. Default is
        True.
    solve_triangular : bool, optional
        Whether to use triangular solve (faster) or general solve
        to compute Q. Default is True.

    Returns
    -------
    Q : array_like or None
        The isometric factor, shape ``(..., da, db)``.
    s : None
        Always None.
    R : array_like or None
        The upper triangular factor, shape ``(..., db, db)``.
    """
    # map QR absorb to LQ absorb
    absorb_t = Absorb.transpose(absorb)

    xp = ar.get_namespace(x)
    xT = xp.swapaxes(x, -2, -1)

    lt, _, qt = blocklevel_lq_via_cholesky(
        xT,
        absorb=absorb_t,
        shift=shift,
        solve_triangular=solve_triangular,
    )

    # transpose results back
    R = xp.swapaxes(lt, -2, -1) if lt is not None else None
    Q = xp.swapaxes(qt, -2, -1) if qt is not None else None

    return Q, None, R


def blocklevel_svd_rand_truncated(
    x,
    max_bond,
    absorb=Absorb.U_s_VH,
    oversample=10,
    num_iterations=2,
    seed=None,
    right=None,
):
    """Randomized SVD of 2D blocks ``x``, with static truncation
    and all absorb mode shortcuts.

    Uses a random sketch to approximate the column (or row) space,
    followed by eigendecomposition of the reduced matrix via
    ``blocklevel_svd_via_eig``.

    Works for a single 2D block ``(da, db)`` or batched blocks with
    arbitrary leading dimensions ``(..., da, db)``.

    Parameters
    ----------
    x : array_like
        Blocks of shape ``(..., da, db)``.
    max_bond : int
        Maximum bond dimension per block (target rank).
    absorb : Absorb, optional
        Absorption mode code controlling what to compute / return.
    oversample : int, optional
        Extra dimensions added to the sketch for accuracy.
        Default is 10.
    num_iterations : int, optional
        Number of power iterations for accuracy. Default is 2.
    seed : int, Generator or None, optional
        Random seed or generator for reproducibility.
    right : bool, optional
        Whether to sketch from the right (True) or left (False).
        If None, choose based on shape and absorb mode.

    Returns
    -------
    U : array_like or None
        Left singular vectors, shape ``(..., da, k)``.
    s : array_like or None
        Singular values, shape ``(..., k)``.
    VH : array_like or None
        Right singular vectors, shape ``(..., k, db)``.
    """
    xp = ar.get_namespace(x)
    shape = xp.shape(x)
    da, db = shape[-2], shape[-1]
    batch = shape[:-2]

    # determine target rank and sketch size
    k = min(da, db, max_bond)
    k_sketch = min(da, db, k + oversample)

    if right is None:
        # avoid svd on reduced factor if possible
        if absorb in (Absorb.U_sVH, Absorb.U, Absorb.sVH):
            right = True
        elif absorb in (Absorb.Us_VH, Absorb.Us, Absorb.VH):
            right = False
        else:
            right = da > db

    rng = xp.random.default_rng(seed, like=x)
    xdag = xp.conj(xp.swapaxes(x, -2, -1))

    if right:
        # tall: sketch from the right
        omega = rng.normal(size=batch + (db, k_sketch))
        y = x @ omega
        if num_iterations:
            for _ in range(num_iterations):
                y = xdag @ y
                y = x @ y

        # form orthonormal basis of column space / 'U'
        Q, _ = xp.linalg.qr(y)
        Qdag = xp.conj(xp.swapaxes(Q, -2, -1))

        # X ~ Q @ B
        if k >= k_sketch:
            if absorb == Absorb.U_sVH:  # 'right'
                return Q, None, Qdag @ x
            if absorb == Absorb.sVH:  # 'rfactor'
                return None, None, Qdag @ x
            if absorb == Absorb.U:  # 'lorthog'
                return Q, None, None

        # form reduced factor
        B = Qdag @ x
    else:
        # wide: sketch from the left
        omega = rng.normal(size=batch + (k_sketch, da))
        y = omega @ x
        if num_iterations:
            for _ in range(num_iterations):
                y = y @ xdag
                y = y @ x
        y = xp.conj(xp.swapaxes(y, -2, -1))

        # form orthonormal basis of row space / 'V'
        Q, _ = xp.linalg.qr(y)
        Qdag = xp.conj(xp.swapaxes(Q, -2, -1))

        # X ≈ B @ Qdag, maybe shortcut for some absorb if no truncation needed
        if k >= k_sketch:
            if absorb == Absorb.Us_VH:  # 'left'
                return x @ Q, None, Qdag
            if absorb == Absorb.Us:  # 'lfactor'
                return x @ Q, None, None
            if absorb == Absorb.VH:  # 'rorthog'
                return None, None, Qdag

        # form reduced factor
        B = x @ Q

    # decompose and maybe further truncate reduced matrix
    U, s, VH = blocklevel_svd_via_eig(
        B,
        absorb=absorb,
        max_bond=k,
        descending=True,
    )

    # expand back out from reduced space
    if (U is not None) and right:
        U = Q @ U
    if (VH is not None) and not right:
        VH = VH @ Qdag

    return U, s, VH
