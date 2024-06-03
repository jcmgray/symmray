def get_rand_z2array(
    shape,
    flows=None,
    charge_total=0,
    seed=None,
    dist="normal",
):
    """Generate a random Z2Array with the given shape, with charge sectors and
    flows automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    flows : list of bool, optional
        The flow of each dimension. If None, the flow is set to True for the
        first half of the dimensions and False for the second half.
    charge_total : int, optional
        The total charge of the array.
    seed : int, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".

    Returns
    -------
    Z2Array
    """
    import symmray as sr

    ndim = len(shape)

    if flows is None:
        flows = [i < ndim // 2 for i in range(ndim)]

    return sr.Z2Array.random(
        indices=[
            sr.BlockIndex(
                {0: d // 2 + d % 2, 1: d // 2},
                flow=f,
            )
            for d, f in zip(shape, flows)
        ],
        charge_total=charge_total,
        seed=seed,
        dist=dist,
    )


def get_rand_u1array(
    shape, flows=None, charge_total=0, seed=None, dist="normal"
):
    """Generate a random U1Array with the given shape, with charge sectors and
    flows automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    flows : list of bool, optional
        The flow of each dimension. If None, the flow is set to True for the
        first half of the dimensions and False for the second half.
    charge_total : int, optional
        The total charge of the array.
    seed : int, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".

    Returns
    -------
    U1Array
    """
    import symmray as sr

    ndim = len(shape)

    if flows is None:
        flows = [i < ndim // 2 for i in range(ndim)]

    return sr.U1Array.random(
        indices=[
            sr.BlockIndex(
                {c: 1 for c in range(-d // 2, d // 2 + 1)},
                flow=f,
            )
            for d, f in zip(shape, flows)
        ],
        charge_total=charge_total,
        seed=seed,
        dist=dist,
    )


def get_rand_symmetric(
    symmetry,
    shape,
    flows=None,
    charge_total=0,
    seed=None,
    dist="normal",
):
    if symmetry == "Z2":
        return get_rand_z2array(
            shape,
            flows=flows,
            charge_total=charge_total,
            seed=seed,
            dist=dist,
        )
    elif symmetry == "U1":
        return get_rand_u1array(
            shape,
            flows=flows,
            charge_total=charge_total,
            seed=seed,
            dist=dist,
        )
    else:
        raise ValueError(f"Symmetry unknown or not supported: {symmetry}.")


def get_rand_blockvector(
    size,
    block_size=0.25,
    seed=None,
    dist="normal",
):
    import numpy as np
    import symmray as sr

    rng = np.random.default_rng(seed)
    blocks = {}
    d = 0
    i = 0
    while d < size:
        if block_size < 1:
            # take as typical fraction
            block_size = rng.poisson(block_size * size)
        else:
            # take as absolute size
            block_size = int(block_size)

        block_size = min(max(block_size, 1), size - d)
        block = getattr(rng, dist)(size=block_size)
        blocks[i] = block
        d += block_size
        i += 1

    return sr.BlockVector(blocks)
