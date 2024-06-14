# a simple flag for enabling rigorous checks in many places
DEBUG = False


def get_rng(seed=None):
    import numpy as np

    return np.random.default_rng(seed)


def rand_z2_index(
    d,
    flow=None,
    subsizes=None,
    seed=None,
):
    """Generate a random Z2 index with the given dimension.

    Parameters
    ----------
    d : int
        The total size of the index.
    flow : bool, optional
        The flow of the index. If None, it is randomly chosen.
    subsizes : None, "equal", "maximal", or tuple of int, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). For
        Z2 indices, "maximal" is equivalent to "equal".
    seed : None, int, or numpy.random.Generator, optional
        The seed for the random number generator.

    Returns
    -------
    BlockIndex
    """
    import symmray as sr

    rng = get_rng(seed)

    if flow is None:
        flow = rng.choice([False, True])

    if d == 1:
        charge = int(rng.choice([0, 1]))
        return sr.BlockIndex(chargemap={charge: 1}, flow=flow)

    if subsizes is None:
        d0 = int(rng.integers(1, d))
        d1 = d - d0
    elif subsizes in ("equal", "maximal"):
        d0 = d // 2
        d1 = d - d0
    else:
        d0, d1 = subsizes

    return sr.BlockIndex(chargemap={0: d0, 1: d1}, flow=flow)


def rand_partition(d, n, seed=None):
    """Randomly partition `d` into `n` sizes each of size at least 1."""
    if d == n:
        return [1] * n

    rng = get_rng(seed)
    splits = (
        0,
        *sorted(rng.choice(range(1, d - 1), size=n - 1, replace=False)),
        d,
    )
    return [int(splits[i + 1] - splits[i]) for i in range(n)]


def rand_u1_index(
    d,
    flow=None,
    subsizes=None,
    seed=None,
):
    """Generate a random U1 index with the given dimension.

    Parameters
    ----------
    d : int
        The total size of the index.
    flow : bool, optional
        The flow of the index. If None, it is randomly chosen.
    subsizes : None, "equal", or tuple of int, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", there will be `d` charges of size 1.
    seed : None, int, or numpy.random.Generator, optional
        The seed for the random number generator.

    Returns
    -------
    BlockIndex
    """
    import symmray as sr

    rng = get_rng(seed)

    if flow is None:
        flow = rng.choice([False, True])

    if subsizes is None:
        ncharge = rng.integers(1, d + 1)
        subsizes = rand_partition(d, ncharge)
    elif subsizes == "equal":
        ncharge = d // 2
        subsizes = [2 for _ in range(ncharge)]
        if d % 2:
            ncharge += 1
            subsizes.append(1)
    elif subsizes == "maximal":
        ncharge = d
        subsizes = [1 for _ in range(ncharge)]
    else:
        ncharge = len(subsizes)

    charges = range(-ncharge // 2 + 1, ncharge // 2 + 1)

    return sr.BlockIndex(chargemap=dict(zip(charges, subsizes)), flow=flow)


def choose_flows(flows, ndim):
    if flows == "equal":
        return [i >= ndim // 2 for i in range(ndim)]
    elif flows is None:
        return [None] * ndim
    else:
        if len(flows) != ndim:
            raise ValueError(
                f"Length of flows ({len(flows)}) does not match ndim ({ndim})."
            )
        return flows


def get_rand_z2array(
    shape,
    flows=None,
    charge_total=0,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
):
    """Generate a random Z2Array with the given shape, with charge sectors and
    flows automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    flows : list of bool, optional
        The flow of each dimension. If None, the flow is set to False for the
        first half of the dimensions and True for the second half.
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

    rng = get_rng(seed)

    flows = choose_flows(flows, len(shape))

    if fermionic:
        cls = sr.Z2FermionicArray
    else:
        cls = sr.Z2Array

    return cls.random(
        indices=[
            rand_z2_index(d, flow=f, subsizes=subsizes, seed=rng)
            for d, f in zip(shape, flows)
        ],
        charge_total=charge_total,
        seed=seed,
        dist=dist,
    )


def get_rand_u1array(
    shape,
    flows=None,
    charge_total=0,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
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
    subsizes : None, "equals", or tuple of int, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal.

    Returns
    -------
    U1Array
    """
    import symmray as sr

    rng = get_rng(seed)

    flows = choose_flows(flows, len(shape))

    if fermionic:
        cls = sr.U1FermionicArray
    else:
        cls = sr.U1Array

    return cls.random(
        indices=[
            rand_u1_index(d, f, subsizes=subsizes, seed=rng)
            for d, f in zip(shape, flows)
        ],
        charge_total=charge_total,
        seed=seed,
        dist=dist,
    )


def get_rand(
    symmetry,
    shape,
    flows=None,
    charge_total=0,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
    **kwargs,
):
    """Get a random symmray array.

    Parameters
    ----------
    symmetry : str
        The symmetry of the array.
    shape : tuple of int
        The desired overall effective shape of the array.
    flows : None, "equals", or Sequence[bool], optional
        The flow of each index. If None, the flows are chosen randomly. If
        "equal", the flows are chosen so the first half of the indices have
        flow False and the second half have flow True.
    charge_total : int, optional
        The total charge of the array.
    seed : None, int, or numpy.random.Generator, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".
    fermionic : bool, optional
        Whether to generate a fermionic array.
    subsizes : None, "equal", or tuple of int, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders).

    Returns
    -------
    SymmetricArray or FermionicArray
    """
    if symmetry == "Z2":
        fn = get_rand_z2array
    elif symmetry == "U1":
        fn = get_rand_u1array
    else:
        raise ValueError(f"Symmetry unknown or not supported: {symmetry}.")

    return fn(
        shape,
        flows=flows,
        charge_total=charge_total,
        seed=seed,
        dist=dist,
        fermionic=fermionic,
        subsizes=subsizes,
        **kwargs,
    )


def get_rand_blockvector(
    size,
    block_size=0.25,
    seed=None,
    dist="normal",
):
    import symmray as sr

    rng = get_rng(seed)
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
