import functools

from autoray import do, get_lib_fn, register_function

from .core import BlockArray, BlockIndex, symmetry


def _get_qr_fn(backend, stabilized=False):
    _qr = get_lib_fn(backend, "linalg.qr")

    if not stabilized:
        return _qr

    try:
        _qr_stab = get_lib_fn(backend, "qr_stabilized")

        def _qr(x):
            q, _, r = _qr_stab(x)
            return q, r

    except ImportError:
        _qr_ubstab = _qr
        _diag = get_lib_fn(backend, "diag")
        _reshape = get_lib_fn(backend, "reshape")
        _sign = get_lib_fn(backend, "sign")

        def _qr(x):
            q, r = _qr_ubstab(x)
            s = _sign(_diag(r))
            q = q * _reshape(s, (1, -1))
            r = r * _reshape(s, (-1, 1))
            return q, r

    return _qr


def qr(x, stabilized=False):
    if x.ndim != 2:
        raise NotImplementedError(
            "qr only implemented for 2D BlockArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )

    # get the 'lower' qr function that acts on the blocks
    _qr = _get_qr_fn(x.backend, stabilized=stabilized)

    q_blocks = {}
    r_blocks = {}
    new_chargemap = {}

    for sector, array in x.blocks.items():
        q, r = _qr(array)
        q_blocks[sector] = q
        r_blocks[sector] = r
        new_chargemap[sector[1]] = q.shape[1]

    bond_index = BlockIndex(chargemap=new_chargemap, flow=x.indices[1].flow)
    q = BlockArray(
        indices=(x.indices[0], bond_index),
        charge_total=x.charge_total,
        blocks=q_blocks,
    )
    r = BlockArray(
        indices=(bond_index.conj(), x.indices[1]),
        charge_total=symmetry(),
        blocks=r_blocks,
    )
    return q, r


def qr_stabilized(x):
    q, r = qr(x, stabilized=True)
    return q, None, r


def svd(x):
    if x.ndim != 2:
        raise NotImplementedError(
            "svd only implemented for 2D BlockArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )

    _svd = get_lib_fn(x.backend, "linalg.svd")

    u_blocks = {}
    s_store = {}
    v_blocks = {}
    new_chargemap = {}

    for sector, array in x.blocks.items():
        u, s, v = _svd(array)
        u_blocks[sector] = u
        s_store[sector] = s
        v_blocks[sector] = v
        new_chargemap[sector[1]] = u.shape[1]

    bond_index = BlockIndex(chargemap=new_chargemap, flow=x.indices[1].flow)
    u = BlockArray(
        indices=(x.indices[0], bond_index),
        charge_total=x.charge_total,
        blocks=u_blocks,
    )
    v = BlockArray(
        indices=(bond_index.conj(), x.indices[1]),
        charge_total=symmetry(),
        blocks=v_blocks,
    )
    return u, s_store, v


def svd_truncated(
    x, cutoff=-1.0, cutoff_mode=3, max_bond=-1, absorb=0, renorm=0
):
    """Truncated svd or raw array ``x``.

    Parameters
    ----------
    cutoff : float
        Singular value cutoff threshold.
    cutoff_mode : {1, 2, 3, 4, 5, 6}
        How to perform the trim:

            - 1: ['abs'], trim values below ``cutoff``
            - 2: ['rel'], trim values below ``s[0] * cutoff``
            - 3: ['sum2'], trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4: ['rsum2'], trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5: ['sum1'], trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6: ['rsum1'], trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

    max_bond : int
        An explicit maximum bond dimension, use -1 for none.
    absorb : {-1, 0, 1, None}
        How to absorb the singular values. -1: left, 0: both, 1: right and
        None: don't absorb (return).
    renorm : {0, 1}
        Whether to renormalize the singular values (depends on `cutoff_mode`).
    """
    backend = x.backend

    # x.check()

    # first perform untruncated svd
    U, s, VH = svd(x)

    if cutoff > 0.0:
        raise NotImplementedError("cutoff > 0.0 not implemented yet.")

    if max_bond > 0:

        # first combine all singular values into a single, sorted array
        sall = do("concatenate", tuple(s.values()), like=backend)
        sall = do("sort", sall, like=backend)[::-1]

        # now find the absolute sigular value at the cutoff
        absolute_cutoff = sall[max_bond]

        # U.check()
        # VH.check()

        for sector in s:
            # check how many singular values from this sector are validi
            n_chi = do(
                "count_nonzero", s[sector] > absolute_cutoff, like=backend
            )

            if n_chi == 0:
                # TODO: drop the block?
                raise NotImplementedError

            # slice the values and left and right vectors
            s[sector] = s[sector][:n_chi]
            U.blocks[sector] = U.blocks[sector][:, :n_chi]
            VH.blocks[sector] = VH.blocks[sector][:n_chi, :]

            # make sure the index chargemaps are updated too
            U.indices[-1].chargemap[sector[-1]] = n_chi
            VH.indices[0].chargemap[sector[0]] = n_chi

    if absorb is None:
        return U, s, VH

    # absorb the singular values block by block
    for sector in s:
        if absorb == -1:
            U.blocks[sector] *= s[sector].reshape((1, -1))
        elif absorb == 1:
            VH.blocks[sector] *= s[sector].reshape((-1, 1))
        elif absorb == 0:
            s_sqrt = do("sqrt", s[sector], like=backend)
            U.blocks[sector] *= s_sqrt.reshape((1, -1))
            VH.blocks[sector] *= s_sqrt.reshape((-1, 1))

    # U.check()
    # VH.check()

    return U, None, VH


# these are used by quimb for compressed contraction and gauging
register_function("symmray", "qr_stabilized", qr_stabilized)
register_function("symmray", "svd_truncated", svd_truncated)
