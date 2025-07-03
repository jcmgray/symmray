import autoray as ar

from .flat_core import AbelianArrayFlat, FlatIndex, FlatVector
from .linalg import qr, svd


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

    # drop fusing info from bond
    bond_ind = FlatIndex(dual=x.indices[1].dual)

    q = x.__class__(
        sectors=x.sectors,
        blocks=qb,
        indices=(
            x.indices[0],
            bond_ind,
        ),
    )

    r = x.__class__(
        sectors=x.sectors[:, (1, 1)],
        blocks=rb,
        indices=(
            bond_ind.conj(),
            x.indices[1],
        ),
    )

    return q, r


@svd.register(AbelianArrayFlat)
def svd_flat(x):
    if x.ndim != 2:
        raise ValueError(
            "SVD is only defined for 2D AbelianArrayFlat objects."
        )

    ub, sb, vb = ar.do("linalg.svd", x._blocks)

    # drop fusing info from bond
    bond_ind = FlatIndex(dual=x.indices[1].dual)

    u = x.__class__(
        sectors=x.sectors,
        blocks=ub,
        indices=(
            x.indices[0],
            bond_ind,
        ),
    )

    s = FlatVector(
        sectors=x.sectors[:, -1],
        blocks=sb,
    )

    vh = x.__class__(
        sectors=x.sectors[:, (1, 1)],
        blocks=vb,
        indices=(
            bond_ind.conj(),
            x.indices[1],
        ),
    )

    return u, s, vh
