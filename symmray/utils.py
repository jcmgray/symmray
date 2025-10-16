import functools
import hashlib
import numbers
import os
import pickle

# a simple flag for enabling rigorous checks in many places
DEBUG = bool(os.environ.get("SYMMRAY_DEBUG", "0").upper() in ("1", "TRUE"))


def set_debug(debug):
    global DEBUG
    DEBUG = debug


def lazyabstractmethod(method):
    """Mark a method as one that must be implemented in a subclass, but only
    enforce this when the method is called. This can be used as a decorator (if
    you want to demonstrate the call signature) or by directly assigning the
    result to a method name.
    """

    if callable(method):
        name = method.__name__
    else:
        name = str(method)

    def raising_method(self, *args, **kwargs):
        raise NotImplementedError(
            f"`{name}` must be implemented in "
            f"subclass `{self.__class__.__name__}`"
        )

    return raising_method


def hasher(k):
    return hashlib.sha1(pickle.dumps(k)).hexdigest()


class RandomStateTranslated:
    """Simple wrapper to make `numpy.random.RandomState` have the same
    interface as `numpy.random.Generator`."""

    def __init__(self, rng):
        self.rng = rng

    def integers(self, *args, **kwargs):
        return self.rng.randint(*args, **kwargs)

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            x = getattr(self.rng, name)
            super().__getattribute__("__dict__")[name] = x
            return x


def get_rng(seed=None):
    import numpy as np

    # RandomStateTranslated is useful for determinism across numpy versions
    if isinstance(seed, RandomStateTranslated):
        return seed
    elif isinstance(seed, np.random.RandomState):
        return RandomStateTranslated(seed)
    else:
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
    dual="random",
    subsizes="random",
    seed=None,
):
    """Generate a random ZN index with the given dimension and order.

    Parameters
    ----------
    d : int, dict[hashable, int] or sequence[(hashable, int), ...]
        The total size of the index. If a dict, an explicit chargemap. If a
        sequence, a linearmap.
    order : int
        The order (i.e. size, N) of the cyclic group ZN.
    dual : bool or "random", optional
        The dualness of the index. If "random", it is randomly chosen.
    subsizes : "random", "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - "random": the charges and sizes are randomly determined.
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

    if (dual is None) or (dual == "random"):
        dual = rng.choice([False, True])

    # convert from possible numpy.int etc.
    d = int(d)

    if d == 1:
        # can only have one charge sector
        if (subsizes is None) or (subsizes == "random"):
            charge = int(rng.integers(order))
        else:
            charge = 0
        return sr.BlockIndex(chargemap={charge: 1}, dual=dual)

    if (subsizes is None) or (subsizes == "random"):
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
    dual="random",
    subsizes="random",
    seed=None,
):
    import symmray as sr

    rng = get_rng(seed)

    if (dual is None) or (dual == "random"):
        dual = rng.choice([False, True])

    possible = [(0, 0), (0, 1), (1, 0), (1, 1)]

    if (subsizes is None) or (subsizes == "random"):
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
    dual="random",
    subsizes="random",
    seed=None,
):
    """Generate a random U1 index with the given dimension.

    Parameters
    ----------
    d : int
        The total size of the index.
    dual : bool or "random", optional
        The dualness of the index. If "random", it is randomly chosen.
    subsizes : "random", "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - "random": the charges and sizes are randomly determined.
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

    if (dual is None) or (dual == "random"):
        dual = rng.choice([False, True])

    if (subsizes is None) or (subsizes == "random"):
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
    dual="random",
    subsizes="random",
    seed=None,
):
    import symmray as sr

    rng = get_rng(seed)

    if (dual is None) or (dual == "random"):
        dual = rng.choice([False, True])

    if (subsizes is None) or (subsizes == "random"):
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


def rand_index(
    symmetry,
    d,
    dual="random",
    subsizes="random",
    seed=None,
):
    """Get a random index with the given symmetry.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the index.
    d : int, dict, sequence, or BlockIndex
        The total size of the index. If a dict, an explicit chargemap. If a
        sequence an explicit linearmap. If a BlockIndex, it is returned
        unchanged.
    dual : bool or "random", optional
        The dualness of the index. If "random", it is randomly chosen.
    subsizes : "random", "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - "random": the charges and sizes are randomly determined.
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

    if isinstance(d, sr.BlockIndex):
        # already a BlockIndex, nothing more to generate
        return d

    symmetry = sr.get_symmetry(symmetry)

    rng = get_rng(seed)

    if (dual is None) or (dual == "random"):
        dual = rng.choice([False, True])

    if isinstance(d, dict):
        # charges and sizes given explicitly, nothing more to generate
        return sr.BlockIndex(chargemap=d, dual=dual)

    if not isinstance(d, numbers.Integral):
        # charges and sizes given as a linearmap, nothing more to generate
        return sr.BlockIndex(linearmap=d, dual=dual)

    kwargs = {
        "dual": dual,
        "subsizes": subsizes,
        "seed": rng,
    }

    if isinstance(symmetry, sr.ZN):
        return rand_zn_index(d, order=symmetry.N, **kwargs)
    elif symmetry == "Z2Z2":
        return rand_z2z2_index(d, **kwargs)
    elif symmetry == "U1":
        return rand_u1_index(d, **kwargs)
    elif symmetry == "U1U1":
        return rand_u1u1_index(d, **kwargs)
    else:
        raise ValueError(f"Symmetry unknown or not supported: {symmetry}.")


def get_array_cls(symmetry, fermionic=False, flat=False) -> type:
    """Get the array class for the given symmetry, fermionic flag, and flat
    flag.

    Parameters
    ----------

    """
    import symmray as sr

    if symmetry in ("Z2", "U1", "Z2Z2", "U1U1"):
        # statically defined array classes
        return {
            # blocksparse abelian arrays
            ("Z2", 0, 0): sr.Z2Array,
            ("U1", 0, 0): sr.U1Array,
            ("Z2Z2", 0, 0): sr.Z2Z2Array,
            ("U1U1", 0, 0): sr.U1U1Array,
            # blocksparse fermionic arrays
            ("Z2", 1, 0): sr.Z2FermionicArray,
            ("U1", 1, 0): sr.U1FermionicArray,
            ("Z2Z2", 1, 0): sr.Z2Z2FermionicArray,
            ("U1U1", 1, 0): sr.U1U1FermionicArray,
            # flat abelian arrays
            ("Z2", 0, 1): sr.Z2ArrayFlat,
            # flat fermionic arrays
            ("Z2", 1, 1): sr.Z2FermionicArrayFlat,
        }[str(symmetry), fermionic, flat]
    else:
        # symmetry is defined dynamically, and should be supplied as kwarg
        return {
            (0, 0): sr.AbelianArray,
            (1, 0): sr.FermionicArray,
            (0, 1): sr.AbelianArrayFlat,
            (1, 1): sr.FermionicArrayFlat,
        }[fermionic, flat]


def choose_duals(duals, ndim):
    if duals == "equal":
        # split ~half and ~half
        return [i >= ndim // 2 for i in range(ndim)]
    elif (
        (duals == "random")
        or (duals is None)
        or (duals is False)
        or (duals is True)
    ):
        # repeat for all axes
        return [duals] * ndim
    else:
        # assume given explicit sequence
        if len(duals) != ndim:
            raise ValueError(
                f"Length of duals ({len(duals)}) does not match ndim ({ndim})."
            )
        return duals


def get_rand(
    symmetry,
    shape,
    duals="random",
    charge=None,
    seed=None,
    dist="normal",
    fermionic=False,
    label=None,
    flat=False,
    subsizes="random",
    **kwargs,
):
    """Get a random symmray array, with the given symmetry and shape. The
    duals, charge, and sub charge sizes can be specified or automatically
    or randomly generated.

    Parameters
    ----------
    symmetry : str
        The symmetry of the array.
    shape : tuple[int | dict | sequence | BlockIndex, ...]
        The desired overall effective shape of the array. Each element can be
        an int, in which case the charge sizes will be generated automatically
        according to `subsizes`, or an explicit dict of charge sizes, or an
        explicit linearmap, or a `BlockIndex`.
    duals : "random", "equals", or Sequence[bool], optional
        The dualness of each index. If "random", the dualnesses are chosen
        randomly. If "equal", they are chosen so the first half of the
        indices have `dual=False` and the second half have `dual=True`.
        If `shape` contains `BlockIndex` objects, the dualness of that index
        takes precedence over any specification here.
    charge : int or tuple[int], optional
        The total charge of the array. If not given the 'zero' charge is used,
        which depends on the symmetry.
    seed : None, int, or numpy.random.Generator, optional
        The seed for the random number generator.
    dist : str, optional
        The distribution of the random numbers. Can be "normal" or "uniform".
    fermionic : bool, optional
        Whether to generate a fermionic array.
    label : hashable, optional
        An optional label for the array, potentially needed for ordering dummy
        odd fermionic modes.
    flat : bool, optional
        Whether to generate a 'flat' backend array (True) or the default
        block-sparse backend array (False).
    subsizes : "random", "equal", "maximal", "minimal", or tuple[int], optional
        The sizes of the charge sectors. The choices are as follows:

        - "random": the charges and sizes are randomly determined.
        - "equal": a fixed number of charges 'close to' zero charge are chosen,
          all with equal size (up to remainders).
        - "maximal": as many charges as possible are chosen, each with size 1
          (or more if the total number of charges is less than the total size).
        - "minimal": only the zero charge sector is chosen, with full size.
        - tuple: the sizes of the charge sectors, a matching number of charges
          are chosen automatically, in sequence 'closest to zero'.

        If `shape` contains `dict` or `BlockIndex` objects, the subsizes of
        those indices take precedence over any specification here.
    kwargs
        Additional keyword arguments are passed to the random array
        generation function.

    Returns
    -------
    AbelianArray or FermionicArray
    """
    import symmray as sr

    symmetry = sr.get_symmetry(symmetry)
    rng = get_rng(seed)
    duals = choose_duals(duals, len(shape))
    cls = get_array_cls(symmetry, fermionic=fermionic, flat=False)

    indices = [
        rand_index(symmetry, d, dual=dual, subsizes=subsizes, seed=rng)
        for d, dual in zip(shape, duals)
    ]

    if not fermionic:
        kwargs.pop("oddpos", None)

    x = cls.random(
        indices=indices,
        charge=charge,
        seed=rng,
        dist=dist,
        symmetry=symmetry,
        label=label,
        **kwargs,
    )

    if flat:
        x = x.to_flat()

    return x


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
    cls = get_array_cls(symmetry, fermionic=fermionic, flat=False)
    return cls.from_dense(
        array,
        index_maps,
        duals=duals,
        charge=charge,
        symmetry=symmetry,
    )
