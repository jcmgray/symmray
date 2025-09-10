import autoray as ar

from .flat_core import AbelianArrayFlat, FlatIndex, FlatVector
from .linalg import qr, svd, svd_truncated, eigh, eigh_truncated
from .utils import DEBUG


@qr.register(AbelianArrayFlat)
def qr_flat(
    x: AbelianArrayFlat,
    stabilized=False,
):
    if x.ndim != 2:
        raise ValueError("QR is only defined for 2D AbelianArrayFlat objects.")

    qb, rb = ar.do("linalg.qr", x._blocks, like=x.backend)

    if stabilized:
        # make each r-factor have positive diagonal
        rbd = ar.do("diagonal", rb, axis1=1, axis2=2, like=x.backend)
        r0 = rbd == 0
        s = (rbd + r0) / (ar.do("abs", rbd, like=x.backend) + r0)

        qb = qb * s[:, None, :]
        rb = rb * s[:, :, None]

    ixl, ixr = x.indices

    # drop fusing info from bond
    bond_ind = FlatIndex(
        num_charges=ixr.num_charges,
        charge_size=min(ixl.charge_size, ixr.charge_size),
        dual=ixr.dual,
    )

    q = x.copy_with(
        blocks=qb,
        indices=(ixl, bond_ind),
    )

    # R is always charge 0 and thus block diagonal
    r = x.copy_with(
        sectors=x.sectors[:, (1, 1)],
        blocks=rb,
        indices=(bond_ind.conj(), ixr),
    )

    return q, r


@svd.register(AbelianArrayFlat)
def svd_flat(
    x: AbelianArrayFlat,
) -> tuple[AbelianArrayFlat, FlatVector, AbelianArrayFlat]:
    if x.ndim != 2:
        raise ValueError(
            "SVD is only defined for 2D AbelianArrayFlat objects."
        )

    ub, sb, vb = ar.do(
        "linalg.svd",
        x._blocks,
        like=x.backend,
    )

    ixl, ixr = x.indices

    # drop fusing info from bond
    bond_ind = FlatIndex(
        num_charges=ixr.num_charges,
        charge_size=min(ixl.charge_size, ixr.charge_size),
        dual=ixr.dual,
    )

    u = x.copy_with(
        blocks=ub,
        indices=(ixl, bond_ind),
    )

    s = FlatVector(
        sectors=x.sectors[:, -1],
        blocks=sb,
    )

    # VH is always charge 0 and thus block diagonal
    vh = x.copy_with(
        sectors=x.sectors[:, (1, 1)],
        blocks=vb,
        indices=(bond_ind.conj(), ixr),
    )

    if DEBUG:
        # u and vh checked in copy_with
        s.check()

    return u, s, vh


_ABSORB_MAP = {
    -1: -1,
    "left": -1,
    0: 0,
    "both": 0,
    1: 1,
    "right": 1,
    None: None,
}


def _truncate_svd_result(
    U: AbelianArrayFlat,
    s: FlatVector,
    VH: AbelianArrayFlat,
    cutoff: float,
    cutoff_mode: int,
    max_bond: int,
    absorb: int,
    renorm: int,
):
    absorb = _ABSORB_MAP[absorb]

    if cutoff > 0.0:
        raise NotImplementedError(
            "Cutoff is not implemented for flat SVD yet."
        )

    if renorm:
        raise NotImplementedError(
            "Renormalization is not implemented for flat SVD yet."
        )

    if max_bond > 0:
        # we must evenly distribute the bond dimension across charges
        bond = U.indices[1]
        # can't make bond larger
        charge_size = min(
            bond.charge_size,
            max_bond // bond.num_charges,
        )

        # we make sure to drop fusing subinfo from truncated bond
        U.modify(
            blocks=U._blocks[:, :, :charge_size],
            indices=(
                U.indices[0],
                U.indices[1].copy_with(
                    charge_size=charge_size,
                    subinfo=None,
                ),
            ),
        )
        s._blocks = s._blocks[:, :charge_size]
        VH.modify(
            blocks=VH._blocks[:, :charge_size, :],
            indices=(
                VH.indices[0].copy_with(
                    charge_size=charge_size,
                    subinfo=None,
                ),
                VH.indices[1],
            ),
        )

    if absorb is None:
        if DEBUG:
            s.check()
    elif absorb == 0:
        # absorb sqrt(s) into both U and VH
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        s = None
    elif absorb == -1:
        # absorb s left into U
        U.multiply_diagonal(s, axis=1, inplace=True)
        s = None
    elif absorb == 1:
        # absorb s right into VH
        VH.multiply_diagonal(s, axis=0, inplace=True)
        s = None
    else:
        raise ValueError(
            f"absorb must be 0, -1, 1, or None. Got {absorb} instead."
        )

    if DEBUG:
        U.check()
        VH.check()

    return U, s, VH


@svd_truncated.register(AbelianArrayFlat)
def svd_truncated(
    x: AbelianArrayFlat,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
):
    U, s, VH = svd_flat(x)
    return _truncate_svd_result(
        U, s, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
    )


@eigh.register(AbelianArrayFlat)
def eigh_flat(
    a: AbelianArrayFlat,
) -> tuple[AbelianArrayFlat, FlatVector]:
    if a.ndim != 2:
        raise NotImplementedError(
            "Eigendecomposition is only defined for 2D AbelianArrayFlat objects."
        )

    eval_blocks, evec_blocks = ar.do(
        "linalg.eigh",
        a._blocks,
        like=a.backend,
    )

    eigenvectors = a.copy_with(blocks=evec_blocks)
    eigenvalues = FlatVector(sectors=a.sectors[:, -1], blocks=eval_blocks)

    if DEBUG:
        eigenvectors.check()
        eigenvalues.check()

    return eigenvalues, eigenvectors


@eigh_truncated.register(AbelianArrayFlat)
def eigh_truncated(
    a: AbelianArrayFlat,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
    positive=0,
    **kwargs,
):
    if kwargs:
        import warnings

        warnings.warn(
            f"Got unexpected kwargs {kwargs} in eigh_truncated "
            "for AbelianArrayFlat. Ignoring them.",
            UserWarning,
        )

    s, U = eigh_flat(a)

    # make sure to sort by descending absolute value
    if not positive:
        idx = ar.do(
            "argsort", -ar.do("abs", s._blocks, like=a.backend), axis=1
        )
        s.modify(
            blocks=ar.do("take_along_axis", s._blocks, idx, axis=1),
        )
        U.modify(
            blocks=ar.do("take_along_axis", U._blocks, idx[:, None, :], axis=2)
        )
    else:
        # assume all positive, just need to flip
        s.modify(blocks=s._blocks[:, ::-1])
        U.modify(blocks=U._blocks[:, :, ::-1])

    if DEBUG:
        s.check()
        U.check()

    # then we can truncate as if svd
    return _truncate_svd_result(
        U, s, U.H, cutoff, cutoff_mode, max_bond, absorb, renorm
    )
