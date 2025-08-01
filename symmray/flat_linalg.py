import autoray as ar

from .flat_core import AbelianArrayFlat, FlatIndex, FlatVector
from .linalg import qr, svd, svd_truncated
from .utils import DEBUG


@qr.register(AbelianArrayFlat)
def qr_flat(x, stabilized=False):
    if x.ndim != 2:
        raise ValueError("QR is only defined for 2D AbelianArrayFlat objects.")

    qb, rb = ar.do("linalg.qr", x._blocks, like=x.backend)

    if stabilized:
        # make each r-factor have positive diagonal
        rbd = ar.do("diagonal", rb, axis1=1, axis2=2)
        r0 = rbd == 0
        s = (rbd + r0) / (ar.do("abs", rbd) + r0)

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

    if DEBUG:
        q.check()
        r.check()

    return q, r


@svd.register(AbelianArrayFlat)
def svd_flat(
    x: AbelianArrayFlat,
) -> tuple[AbelianArrayFlat, FlatVector, AbelianArrayFlat]:
    if x.ndim != 2:
        raise ValueError(
            "SVD is only defined for 2D AbelianArrayFlat objects."
        )

    ub, sb, vb = ar.do("linalg.svd", x._blocks)

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


@svd_truncated.register(AbelianArrayFlat)
def svd_truncated(
    x: AbelianArrayFlat,
    cutoff=-1.0,
    cutoff_mode=4,
    max_bond=-1,
    absorb=0,
    renorm=0,
):
    if cutoff > 0.0:
        raise NotImplementedError(
            "Cutoff is not implemented for flat SVD yet."
        )

    if renorm:
        raise NotImplementedError(
            "Renormalization is not implemented for flat SVD yet."
        )

    U, s, VH = svd_flat(x)

    if max_bond > 0:
        # we must evenly distribute the bond dimension across charges
        charge_size = max_bond // x.order

        U.modify(
            blocks=U._blocks[:, :, :charge_size],
            indices=(
                U.indices[0],
                U.indices[1].copy_with(charge_size=charge_size),
            ),
        )
        s._blocks = s._blocks[:, :charge_size]
        VH.modify(
            blocks=VH._blocks[:, :charge_size, :],
            indices=(
                VH.indices[0].copy_with(charge_size=charge_size),
                VH.indices[1],
            ),
        )

    if absorb is None:
        if DEBUG:
            s.check()
    elif absorb == 0:
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        s = None
    elif absorb == -1:
        U.multiply_diagonal(s, axis=1, inplace=True)
        s = None
    elif absorb == 1:
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
