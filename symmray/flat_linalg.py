import autoray as ar

from .flat_core import FlatIndex, AbelianArrayFlat
from .linalg import qr


@qr.register(AbelianArrayFlat)
def qr_flat(x, stabilized=False):
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
