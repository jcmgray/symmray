# a simple flag for enabling rigorous checks in many places
DEBUG = False


def set_debug(debug):
    global DEBUG
    DEBUG = debug


def get_rng(seed=None):
    import numpy as np

    return np.random.default_rng(seed)


def rand_z2_index(
    d,
    dual=None,
    subsizes=None,
    seed=None,
):
    """Generate a random Z2 index with the given dimension.

    Parameters
    ----------
    d : int or dict
        The total size of the index. If a dict, an explicit chargemap.
    dual : bool, optional
        The dualness of the index. If None, it is randomly chosen.
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

    if dual is None:
        dual = rng.choice([False, True])

    if isinstance(d, dict):
        return sr.BlockIndex(chargemap=d, dual=dual)

    if d == 1:
        charge = int(rng.choice([0, 1]))
        return sr.BlockIndex(chargemap={charge: 1}, dual=dual)

    if subsizes is None:
        d0 = int(rng.integers(1, d))
        d1 = d - d0
    elif subsizes in ("equal", "maximal"):
        d0 = d // 2
        d1 = d - d0
    else:
        d0, d1 = subsizes

    return sr.BlockIndex(chargemap={0: d0, 1: d1}, dual=dual)


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


def rand_z2z2_index(
    d,
    dual=None,
    subsizes=None,
    seed=None,
):
    import symmray as sr

    rng = get_rng(seed)

    if dual is None:
        dual = rng.choice([False, True])

    possible = [(0, 0), (0, 1), (1, 0), (1, 1)]

    if subsizes is None:
        # randomly distributed
        if d < 4:
            charges = rng.choice(possible, size=d, replace=False)
            chargemap = {tuple(c): 1 for c in charges}
        else:
            subsizes = rand_partition(d, 4)
            chargemap = dict(zip(possible, subsizes))

    elif subsizes in ("equal", "maximal"):
        # round-robin distribution
        ncharge = min(d, 4)
        charges = possible[:ncharge]
        chargemap = {c: d // 4 + (i < d % 4) for i, c in enumerate(charges)}

    return sr.BlockIndex(chargemap=chargemap, dual=dual)


def rand_u1_index(
    d,
    dual=None,
    subsizes=None,
    seed=None,
):
    """Generate a random U1 index with the given dimension.

    Parameters
    ----------
    d : int
        The total size of the index.
    dual : bool, optional
        The dualness of the index. If None, it is randomly chosen.
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

    if dual is None:
        dual = rng.choice([False, True])

    if isinstance(d, dict):
        return sr.BlockIndex(chargemap=d, dual=dual)

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
    chargemap = dict(zip(charges, subsizes))

    return sr.BlockIndex(chargemap=chargemap, dual=dual)


def get_u1u1_charges(ncharge):
    """Get a list of ``ncharge`` distinct U1U1 charges that are as close to
    the origin as possible.
    """
    import itertools

    k = int(ncharge**0.5)
    krange = tuple(range(-k + 1, k + 1))
    charges = []
    for i, j in itertools.product(krange, repeat=2):
        charges.append((i, j))

    charges.sort(
        # choose spherical distribution with positive tie breaking bias
        key=lambda xy: (xy[0] ** 2 + xy[1] ** 2, -xy[0] - xy[1])
        # # diamond distribution:
        # key=lambda xy: (abs(xy[0]) + abs(xy[1]), -xy[0], -xy[1])
    )

    return tuple(charges[:ncharge])


def rand_u1u1_index(
    d,
    dual=None,
    subsizes=None,
    seed=None,
):
    import symmray as sr

    rng = get_rng(seed)

    if dual is None:
        dual = rng.choice([False, True])

    if isinstance(d, dict):
        return sr.BlockIndex(chargemap=d, dual=dual)

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

    charges = get_u1u1_charges(ncharge)
    chargemap = dict(zip(charges, subsizes))

    return sr.BlockIndex(chargemap=chargemap, dual=dual)


def choose_duals(duals, ndim):
    if duals == "equal":
        return [i >= ndim // 2 for i in range(ndim)]
    elif (duals is None) or (duals is False) or (duals is True):
        return [duals] * ndim
    else:
        if len(duals) != ndim:
            raise ValueError(
                f"Length of duals ({len(duals)}) does not match ndim ({ndim})."
            )
        return duals


def get_rand_z2array(
    shape,
    duals=None,
    charge=0,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
):
    """Generate a random Z2Array with the given shape, with charge sectors and
    duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    duals : list of bool, optional
        The dualness of each dimension. If None, the dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : int, optional
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

    duals = choose_duals(duals, len(shape))

    if fermionic:
        cls = sr.Z2FermionicArray
    else:
        cls = sr.Z2Array

    return cls.random(
        indices=[
            rand_z2_index(d, dual=f, subsizes=subsizes, seed=rng)
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
    )


def get_rand_z2z2array(
    shape,
    duals=None,
    charge=None,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
):
    """Generate a random Z2Z2Array with the given shape, with charge sectors
    and duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    duals : list of bool, optional
        The dualness of each dimension. If None, the dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : tuple of int, optional
        The total charge of the array.
    seed : int, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".

    Returns
    -------
    Z2Z2Array
    """
    import symmray as sr

    rng = get_rng(seed)

    duals = choose_duals(duals, len(shape))

    if fermionic:
        cls = sr.Z2Z2FermionicArray
    else:
        cls = sr.Z2Z2Array

    return cls.random(
        indices=[
            rand_z2z2_index(d, dual=f, subsizes=subsizes, seed=rng)
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
    )


def get_rand_u1array(
    shape,
    duals=None,
    charge=0,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
):
    """Generate a random U1Array with the given shape, with charge sectors and
    duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    duals : list of bool, optional
        The dualness of each dimension. If None, then dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : int, optional
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

    duals = choose_duals(duals, len(shape))

    if fermionic:
        cls = sr.U1FermionicArray
    else:
        cls = sr.U1Array

    return cls.random(
        indices=[
            rand_u1_index(d, f, subsizes=subsizes, seed=rng)
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
    )


def get_rand_u1u1array(
    shape,
    duals=None,
    charge=(0, 0),
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
):
    """Generate a random U1U1Array with the given shape, with charge sectors
    and duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array.
    duals : list of bool, optional
        The dualness of each dimension. If None, then dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : tuple of int, optional
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
    U1U1Array
    """
    import symmray as sr

    rng = get_rng(seed)

    duals = choose_duals(duals, len(shape))

    if fermionic:
        cls = sr.U1U1FermionicArray
    else:
        cls = sr.U1U1Array

    return cls.random(
        indices=[
            rand_u1u1_index(d, f, subsizes=subsizes, seed=rng)
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
    )


def get_rand(
    symmetry,
    shape,
    duals=None,
    charge=None,
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
    duals : None, "equals", or Sequence[bool], optional
        The dualness of each index. If None, the dualnesses are chosen
        randomly. If "equal", they are chosen so the first half of the
        indices have `dual=False` and the second half have `dual=True`.
    charge : int, optional
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
    AbelianArray or FermionicArray
    """
    if symmetry == "Z2":
        fn = get_rand_z2array
    elif symmetry == "Z2Z2":
        fn = get_rand_z2z2array
    elif symmetry == "U1":
        fn = get_rand_u1array
    elif symmetry == "U1U1":
        fn = get_rand_u1u1array
    else:
        raise ValueError(f"Symmetry unknown or not supported: {symmetry}.")

    return fn(
        shape,
        duals=duals,
        charge=charge,
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


def from_dense(
    array,
    symmetry,
    index_maps,
    duals=None,
    fermionic=False,
    charge=None,
):
    from .abelian_core import (
        Z2Array,
        Z2Z2Array,
        U1Array,
        U1U1Array,
    )
    from .fermionic_core import (
        Z2FermionicArray,
        U1FermionicArray,
        Z2Z2FermionicArray,
        U1U1FermionicArray,
    )

    cls = {
        ("Z2", False): Z2Array,
        ("Z2Z2", False): Z2Z2Array,
        ("U1", False): U1Array,
        ("U1U1", False): U1U1Array,
        ("Z2", True): Z2FermionicArray,
        ("Z2Z2", True): Z2Z2FermionicArray,
        ("U1", True): U1FermionicArray,
        ("U1U1", True): U1U1FermionicArray,
    }[symmetry, fermionic]

    return cls.from_dense(
        array,
        index_maps,
        duals=duals,
        charge=charge,
    )


def parse_edges_to_site_info(
    edges,
    bond_dim,
    phys_dim=2,
    site_ind_id="k{}",
    bond_ind_id="b{}-{}",
    site_tag_id="I{}",
):
    """Given a list of edges, return a dictionary of site information, each
    specifying the local shape, index identifiers, index dualnesses, and tags.
    The dualnesses of the bonds are set in a canonical order corresponding to
    sorting all the sites and the edges.

    Parameters
    ----------
    edges : Sequence[Tuple[hashable, hashable]]
        The edges of the graph.
    bond_dim : int
        The internal bond dimension.
    phys_dim : int, optional
        The physical dimension of the sites.
    site_ind_id : str, optional
        The identifier for the site indices.
    bond_ind_id : str, optional
        The identifier for the bond indices.
    site_tag_id : str, optional
        The identifier for the site tags.

    Returns
    -------
    Dict[hashable, Dict[str, Any]]
    """
    sites = {}

    starmap_ind = site_ind_id.count("{}") > 1
    starmap_tag = site_tag_id.count("{}") > 1

    # create bonds
    for sitea, siteb in sorted(edges):
        if sitea > siteb:
            sitea, siteb = siteb, sitea

        ind = bond_ind_id.format(sitea, siteb)
        infoa = sites.setdefault(sitea, {})
        infob = sites.setdefault(siteb, {})

        infoa.setdefault("inds", []).append(ind)
        infob.setdefault("inds", []).append(ind)

        infoa.setdefault("duals", []).append(0)
        infob.setdefault("duals", []).append(1)

        infoa.setdefault("shape", []).append(bond_dim)
        infob.setdefault("shape", []).append(bond_dim)

    # create physical inds
    for site in sites:
        sites[site]["coordination"] = len(sites[site]["inds"])

        if starmap_ind:
            site_ind = site_ind_id.format(*site)
        else:
            site_ind = site_ind_id.format(site)

        sites[site]["inds"].append(site_ind)
        sites[site]["duals"].append(0)
        sites[site]["shape"].append(phys_dim)

        if starmap_tag:
            site_tag = site_tag_id.format(*site)
        else:
            site_tag = site_tag_id.format(site)

        sites[site]["tags"] = (site_tag,)

    return sites
