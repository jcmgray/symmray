"""Utility functions specifically for testing."""

import functools


def hash_kwargs_to_int(**kwargs) -> int:
    """Hash a set of keyword arguments to a deterministic integer, which can be
    used for seeding random number generators in tests, for example, allowing a
    varying but reproducible seed.
    """
    import hashlib

    kwargs_str = str(sorted(kwargs.items()))
    kwargs_hash = hashlib.md5(kwargs_str.encode()).hexdigest()
    return int(kwargs_hash, 16) % (2**32 - 1)


def rand_valid_tensordot(
    symmetry,
    ndim_a=None,
    ndim_b=None,
    ncon=None,
    charge_a=None,
    charge_b=None,
    min_ndim=1,
    max_ndim=5,
    min_d=1,
    max_d=3,
    dimension_multiplier=1,
    subsizes=None,
    seed=None,
    **kwargs,
):
    """Generate two random symmetric arrays and valid contraction ``axes``.
    For testing purposes.

    Parameters
    ----------
    symmetry : str or Symmetry
        Symmetry to use.
    ndim_a : int or None
        Number of dimensions for the first array. If None, a random number is
        chosen between `min_ndim` and `max_ndim` (inclusive).
    ndim_b : int or None
        Number of dimensions for the second array. If None, a random number is
        chosen between `min_ndim` and `max_ndim` (inclusive).
    ncon : int or None
        Number of axes to contract. If None, a random number is chosen between
        0 and min(ndim_a, ndim_b) (inclusive).
    charge_a : charge_like or None
        Charge for the first array. If None, a random charge is chosen.
    charge_b : charge_like or None
        Charge for the second array. If None, a random charge is chosen.
    min_ndim : int
        Minimum number of dimensions for each array.
    max_ndim : int
        Maximum number of dimensions for each array.
    min_d : int
        Minimum dimension for each index. This is multiplied by
        `dimension_multiplier` if that is not None.
    max_d : int
        Maximum dimension for each index. This is multiplied by
        `dimension_multiplier` if that is not None.
    dimension_multiplier : int, optional
        Each index will have dimension `d * dimension_multiplier` where `d`
        is a random integer in [min_d, max_d]. Useful for generating flat
        arrays with exactly equally distributed charge sizes.
    subsizes : str or None
        Passed to `rand_index`. If "equal", all sectors will have equal size,
        required for flat arrays.
    seed : int or np.random.Generator or None
        Random seed or generator. If None, a new generator is created.
    **kwargs
        Additional arguments passed to `get_rand`.

    Returns
    -------
    a : AbelianArray
        First random array.
    b : AbelianArray
        Second random array.
    axes : tuple[tuple[int, ...], tuple[int, ...]]
        Axes to contract.
    """
    import numpy as np

    from .symmetries import get_symmetry
    from .utils import get_rand, get_rng, rand_index

    symmetry = get_symmetry(symmetry)
    rng = get_rng(seed)

    # dimensionality of each array
    if ndim_a is None:
        ndim_a = rng.integers(min_ndim, max_ndim + 1)
    if ndim_b is None:
        ndim_b = rng.integers(min_ndim, max_ndim + 1)

    # number of axes to contract
    if ncon is None:
        ncon = rng.integers(min(ndim_a, ndim_b) + 1)

    # which axes to contract
    axes_a = rng.choice(np.arange(ndim_a), size=ncon, replace=False)
    axes_b = rng.choice(np.arange(ndim_b), size=ncon, replace=False)
    axes = (tuple(map(int, axes_a)), tuple(map(int, axes_b)))

    # now construct compatible indices
    indices_a = [None] * ndim_a
    indices_b = [None] * ndim_b

    for axa, axb in zip(axes[0], axes[1]):
        d = int(dimension_multiplier * rng.integers(min_d, max_d + 1))
        ix = rand_index(symmetry, d, subsizes=subsizes, seed=rng)
        indices_a[axa] = ix.conj()
        indices_b[axb] = ix

    for i in range(ndim_a):
        if indices_a[i] is None:
            d = int(dimension_multiplier * rng.integers(min_d, max_d + 1))
            indices_a[i] = rand_index(symmetry, d, subsizes=subsizes, seed=rng)
    for i in range(ndim_b):
        if indices_b[i] is None:
            d = int(dimension_multiplier * rng.integers(min_d, max_d + 1))
            indices_b[i] = rand_index(symmetry, d, subsizes=subsizes, seed=rng)

    if charge_a is None:
        charge_a = symmetry.random_charge(rng)
    if charge_b is None:
        charge_b = symmetry.random_charge(rng)

    a = get_rand(
        symmetry,
        indices_a,
        charge=charge_a,
        seed=rng,
        label="a",
        **kwargs,
    )
    b = get_rand(
        symmetry,
        indices_b,
        charge=charge_b,
        seed=rng,
        label="b",
        **kwargs,
    )

    return a, b, axes


def rand_matrix(
    symmetry,
    d,
    seed=None,
    subsizes="equal",
    dtype="float64",
    fermionic=False,
    flat=False,
    matrix_type="general",
):
    """Create a general, hermitian, or positive definite matrix (with conjugate
    indices) with the given symmetry and shape (d, d). For testing purposes.
    """
    from .symmetries import get_symmetry
    from .utils import get_rand, get_rng, rand_index

    symm = get_symmetry(symmetry)

    rng = get_rng(
        hash_kwargs_to_int(
            symmetry=symmetry,
            d=d,
            seed=seed,
            subsizes=subsizes,
            dtype=dtype,
            fermionic=fermionic,
            flat=flat,
            matrix_type=matrix_type,
        )
    )

    i = rand_index(symm, d, subsizes=subsizes, seed=rng)

    sx = get_rand(
        symm,
        shape=(i, i.conj()),
        dtype=dtype,
        seed=rng,
        fermionic=fermionic,
    )

    xp = sx.get_namespace()

    if matrix_type == "hermitian":

        def fn_blocks(block):
            return block + xp.conj(xp.transpose(block))

    elif matrix_type == "posdef":

        def fn_blocks(block):
            bh = block @ xp.conj(xp.transpose(block))
            bh = bh + 1e-4 * xp.eye(bh.shape[0], dtype=dtype)
            return bh

    elif matrix_type != "general":
        raise ValueError(f"Invalid matrix_type: {matrix_type}")

    sx._map_blocks(fn_blocks)
    if flat:
        return sx.to_flat()
    else:
        return sx


rand_herm = functools.partial(rand_matrix, matrix_type="hermitian")
rand_posdef = functools.partial(rand_matrix, matrix_type="posdef")
