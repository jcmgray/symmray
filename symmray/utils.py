import functools
import os

# a simple flag for enabling rigorous checks in many places
DEBUG = bool(os.environ.get("SYMMRAY_DEBUG", "0").upper() in ("1", "TRUE"))


def set_debug(debug):
    global DEBUG
    DEBUG = debug


def get_rng(seed=None):
    import numpy as np

    return np.random.default_rng(seed)


def get_random_fill_fn(
    seed=None,
    dist="normal",
    dtype="float64",
    scale=1.0,
    loc=0.0,
):
    """Get a function that produces numpy arrays of random numbers with the
    specified distribution, dtype, loc, and scale.

    Parameters
    ----------
    seed : None, int, or numpy.random.Generator, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform" or
        any other distribution supported by numpy.
    dtype : str, optional
        The data type of the random numbers. If "complex", the real and
        imaginary parts are generated separately and added.
    scale : float, optional
        A multiplicative factor to the distribution.
    loc : float, optional
        An additive offset to the distribution.

    Returns
    -------
    callable
        A function with signature `fill_fn(shape) -> numpy.ndarray`.
    """
    rng = get_rng(seed)

    def fill_fn(shape):
        x = getattr(rng, dist)(size=shape)
        if "complex" in dtype:
            x = x + 1j * getattr(rng, dist)(size=shape)
        if scale != 1.0:
            x *= scale
        if loc != 0.0:
            x += loc
        if x.dtype != dtype:
            x = x.astype(dtype)

        return x

    return fill_fn


def rand_partition(d, n, seed=None):
    """Randomly partition `d` into `n` sizes each of size at least 1."""
    if d == n:
        return [1] * n

    rng = get_rng(seed)

    if n == 2:
        # cut in two
        s = int(rng.integers(1, d))
        return [s, d - s]

    # cut into 3 or more
    splits = (
        0,
        *sorted(rng.choice(range(1, d - 1), size=n - 1, replace=False)),
        d,
    )
    return [int(splits[i + 1] - splits[i]) for i in range(n)]


@functools.cache
def get_zn_charges(ncharge, order):
    """Get a list of ``ncharge`` distinct ZN charges that are as close to
    0 or |G| as possible, with a slight bias towards positive charges.

    Parameters
    ----------
    ncharge : int
        The number of distinct charges to get.
    order : int
        The order (i.e. size, N) of the cyclic group ZN.

    Returns
    -------
    charges : list[int]
        A list of `ncharge <= order` distinct ZN charges.
    """
    if ncharge >= order:
        return list(range(order))

    # get charges 'close to' zero / |G|
    return sorted(
        c % order for c in range(-ncharge // 2 + 1, +ncharge // 2 + 1)
    )


def rand_zn_index(
    d,
    order,
    dual=None,
    subsizes=None,
    seed=None,
):
    """Generate a random ZN index with the given dimension and order.

    Parameters
    ----------
    d : int or dict
        The total size of the index. If a dict, an explicit chargemap.
    order : int
        The order (i.e. size, N) of the cyclic group ZN.
    dual : bool, optional
        The dualness of the index. If None, it is randomly chosen.
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - None: the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.

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
        # charges and sizes given explicitly
        return sr.BlockIndex(chargemap=d, dual=dual)

    # convert from possible numpy.int etc.
    d = int(d)

    if d == 1:
        # can only have one charge sector
        if subsizes is None:
            charge = int(rng.integers(order))
        else:
            charge = 0
        return sr.BlockIndex(chargemap={charge: 1}, dual=dual)

    if subsizes is None:
        # randomly distributed over all
        ncharge = min(d, order)
        subsizes = rand_partition(d, ncharge, seed=rng)

    elif subsizes in ("equal", "maximal"):
        # round-robin == maximal spread
        ncharge = min(d, order)
        subsizes = [
            d // ncharge + int(i < d % ncharge) for i in range(ncharge)
        ]

    elif subsizes == "minimal":
        # all in zero charge sector
        ncharge = 1
        subsizes = [d]

    else:
        # sizes given explicitly
        ncharge = len(subsizes)

    charges = get_zn_charges(ncharge, order)
    chargemap = dict(zip(charges, subsizes))

    return sr.BlockIndex(chargemap=chargemap, dual=dual)


rand_z2_index = functools.partial(rand_zn_index, order=2)


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
            subsizes = rand_partition(d, 4, seed=rng)
            chargemap = dict(zip(possible, subsizes))

    elif subsizes in ("equal", "maximal"):
        # round-robin distribution
        ncharge = min(d, 4)
        charges = possible[:ncharge]
        chargemap = {c: d // 4 + (i < d % 4) for i, c in enumerate(charges)}

    elif subsizes == "minimal":
        # all in zero charge sector
        chargemap = {(0, 0): d}

    else:
        # sizes given explicitly
        ncharge = len(subsizes)
        chargemap = dict(zip(possible, subsizes))

    return sr.BlockIndex(chargemap=chargemap, dual=dual)


def get_u1_charges(ncharge):
    """Get a list of ``ncharge`` distinct U1 charges that are as close to the
    origin as possible, with a slight bias towards positive charges.
    """
    charges = list(range(-ncharge // 2 + 1, +ncharge // 2 + 1))
    charges.sort(key=lambda x: (abs(x), -x))
    return tuple(charges[:ncharge])


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
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - None: the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.

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
        # charges and sizes given explicitly
        return sr.BlockIndex(chargemap=d, dual=dual)

    if subsizes is None:
        # a random number and distribution of charges
        ncharge = rng.integers(1, d + 1)
        subsizes = rand_partition(d, ncharge, seed=rng)

    elif subsizes == "equal":
        # 3 approx equal charges around the origin
        ncharge = min(d, 3)
        subsizes = [
            d // ncharge + int(i < d % ncharge) for i in range(ncharge)
        ]

    elif subsizes == "maximal":
        # maximal spread of charges each with size 1
        ncharge = d
        subsizes = [1 for _ in range(ncharge)]

    elif subsizes == "minimal":
        # minimal spread of charges each with size 1
        ncharge = 1
        subsizes = [d]

    else:
        # only sizes given explicitly
        ncharge = len(subsizes)

    charges = get_u1_charges(ncharge)
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
        # charges and sizes given explicitly
        return sr.BlockIndex(chargemap=d, dual=dual)

    if subsizes is None:
        # a random number and distribution of charges
        ncharge = rng.integers(1, d + 1)
        subsizes = rand_partition(d, ncharge, seed=rng)

    elif subsizes == "equal":
        # 9 approx equal charges around the origin
        ncharge = min(d, 9)
        subsizes = [
            d // ncharge + int(i < d % ncharge) for i in range(ncharge)
        ]

    elif subsizes == "maximal":
        # maximal spread of charges each with size 1
        ncharge = d
        subsizes = [1 for _ in range(ncharge)]

    elif subsizes == "minimal":
        # all in zero charge sector
        ncharge = 1
        subsizes = [d]

    else:
        # only sizes given explicitly
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


def get_rand_znarray(
    shape,
    duals=None,
    charge=0,
    order=2,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
    **kwargs,
):
    """Generate a random Z2Array with the given shape, with charge sectors and
    duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array. Each element can be an int or an
        explicit dict of charge sizes.
    duals : list of bool, optional
        The dualness of each dimension. If None, the dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : int, optional
        The total charge of the array.
    order : int, optional
        The order (i.e. size, N) of the cyclic group ZN.
    seed : int, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".
    fermionic : bool, optional
        Whether to generate a fermionic array.
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        How to choose the sizes of the charge sectors, see `rand_zn_index`.

    Returns
    -------
    Z2Array
    """
    import symmray as sr

    rng = get_rng(seed)

    duals = choose_duals(duals, len(shape))

    if order == 2:
        if fermionic:
            cls = sr.Z2FermionicArray
        else:
            cls = sr.Z2Array
    else:
        if fermionic:
            cls = sr.get_zn_fermionic_array_cls(order)
        else:
            cls = sr.get_zn_array_cls(order)

    return cls.random(
        indices=[
            (
                d
                if isinstance(d, sr.BlockIndex)
                else rand_zn_index(
                    d,
                    order=order,
                    dual=f,
                    subsizes=subsizes,
                    seed=rng,
                )
            )
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
        **kwargs,
    )


get_rand_z2array = functools.partial(get_rand_znarray, order=2)


def get_rand_z2z2array(
    shape,
    duals=None,
    charge=None,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
    **kwargs,
):
    """Generate a random Z2Z2Array with the given shape, with charge sectors
    and duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array. Each element can be an int or an
        explicit dict of charge sizes.
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
            (
                d
                if isinstance(d, sr.BlockIndex)
                else sr.BlockIndex(d, dual=f)
                if isinstance(d, dict)
                else rand_z2z2_index(d, dual=f, subsizes=subsizes, seed=rng)
            )
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
        **kwargs,
    )


def get_rand_u1array(
    shape,
    duals=None,
    charge=0,
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
    **kwargs,
):
    """Generate a random U1Array with the given shape, with charge sectors and
    duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array. Each element can be an int or an
        explicit dict of charge sizes.
    duals : list of bool, optional
        The dualness of each dimension. If None, then dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : int, optional
        The total charge of the array.
    seed : int, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - None: the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.

    kwargs
        Additional keyword arguments are passed to the random array
        generation function.

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
            (
                d
                if isinstance(d, sr.BlockIndex)
                else sr.BlockIndex(d, dual=f)
                if isinstance(d, dict)
                else rand_u1_index(d, f, subsizes=subsizes, seed=rng)
            )
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
        **kwargs,
    )


def get_rand_u1u1array(
    shape,
    duals=None,
    charge=(0, 0),
    seed=None,
    dist="normal",
    fermionic=False,
    subsizes=None,
    **kwargs,
):
    """Generate a random U1U1Array with the given shape, with charge sectors
    and duals automatically determined.

    Parameters
    ----------
    shape : tuple of int
        The overall shape of the array. Each element can be an int or an
        explicit dict of charge sizes.
    duals : list of bool, optional
        The dualness of each dimension. If None, then dual is set to False for
        the first half of the dimensions and True for the second half.
    charge : tuple of int, optional
        The total charge of the array.
    seed : int, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - None: the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.


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
            (
                d
                if isinstance(d, sr.BlockIndex)
                else sr.BlockIndex(d, dual=f)
                if isinstance(d, dict)
                else rand_u1u1_index(d, f, subsizes=subsizes, seed=rng)
            )
            for d, f in zip(shape, duals)
        ],
        charge=charge,
        seed=seed,
        dist=dist,
        **kwargs,
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
    shape : tuple[int | dict | BlockIndex, ...]
        The desired overall effective shape of the array. Each element can be
        an int, in which case the charge sizes will be generated automatically,
        or an explicit dict of charge sizes, or a BlockIndex.
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
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - None: the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.

    kwargs
        Additional keyword arguments are passed to the random array
        generation function.

    Returns
    -------
    AbelianArray or FermionicArray
    """
    import symmray as sr

    symmetry = sr.get_symmetry(symmetry)

    if isinstance(symmetry, sr.ZN):
        fn = get_rand_znarray
        kwargs["order"] = symmetry.N
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
        U1Array,
        U1U1Array,
        Z2Array,
        Z2Z2Array,
    )
    from .fermionic_core import (
        U1FermionicArray,
        U1U1FermionicArray,
        Z2FermionicArray,
        Z2Z2FermionicArray,
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


def rand_index(
    symmetry,
    d,
    dual=None,
    subsizes=None,
    seed=None,
):
    """Get a random index with the given symmetry.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the index.
    d : int or dict
        The total size of the index. If a dict, an explicit chargemap.
    dual : bool, optional
        The dualness of the index. If None, it is randomly chosen.
    subsizes : None, "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - None: the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.

    seed : None, int, or numpy.random.Generator, optional
        The seed for the random number generator.

    Returns
    -------
    BlockIndex
    """
    import symmray as sr

    symmetry = sr.get_symmetry(symmetry)

    if isinstance(symmetry, sr.ZN):
        return rand_zn_index(
            d, order=symmetry.N, dual=dual, subsizes=subsizes, seed=seed
        )
    elif symmetry == "Z2Z2":
        return rand_z2z2_index(d, dual=dual, subsizes=subsizes, seed=seed)
    elif symmetry == "U1":
        return rand_u1_index(d, dual=dual, subsizes=subsizes, seed=seed)
    elif symmetry == "U1U1":
        return rand_u1u1_index(d, dual=dual, subsizes=subsizes, seed=seed)
    else:
        raise ValueError(f"Symmetry unknown or not supported: {symmetry}.")
