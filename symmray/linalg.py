from autoray import get_lib_fn

from .core import BlockArray, BlockIndex, symmetry


def qr(x):
    if x.ndim != 2:
        raise NotImplementedError(
            "qr only implemented for 2D BlockArrays,"
            f" got {x.ndim}D. Consider fusing first."
        )

    _qr = get_lib_fn(x.backend, "linalg.qr")

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
