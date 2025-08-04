import random

import pytest

import symmray as sr


def get_zn_blocksparse_flat_compat(
    symmetry,
    shape,
    charge,
    seed=42,
    shape_relative_to_z2=True,
    **kwargs,
):
    rng = random.Random(seed)

    N = int(symmetry[1:])

    if shape_relative_to_z2:
        shape = [N * d // 2 for d in shape]

    if charge:
        charge = rng.choice(range(1, N))

    return sr.utils.get_rand(
        symmetry,
        shape=shape,
        subsizes="equal",
        charge=charge,
        seed=seed,
        **kwargs,
    )


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize(
    "shape,perm",
    [
        ([2], [0]),
        ([4, 2], [0, 1]),
        ([2, 4], [1, 0]),
        ([2, 4, 2, 4], (2, 1, 3, 0)),
        ([2, 2, 2, 2, 2, 2, 2], (2, 0, 1, 3, 5, 4, 6)),
    ],
)
@pytest.mark.parametrize("charge", [0, 1])
def test_transpose_roundtrip(symmetry, shape, perm, charge):
    sx = get_zn_blocksparse_flat_compat(symmetry, shape, charge, seed=42)
    sy = sx.transpose(perm)
    perm_inv = tuple(perm.index(i) for i in range(len(perm)))

    fx = sx.to_flat()
    fx.check()
    assert fx.to_blocksparse().allclose(sx)
    fy = fx.transpose(perm)
    fy.check()
    assert fy.to_blocksparse().allclose(sy)
    fz = fy.transpose(perm_inv)
    fz.check()
    assert fz.to_blocksparse().allclose(sx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize(
    "shape,axes_groups",
    [
        ([4], [(0,)]),
        ([2, 2], [(0, 1)]),
        ([2, 2], [(0,), (1,)]),
        ([2, 2], [(1,), (0,)]),
        ([4] * 6, [(1, 3), (4, 2)]),
        ([6, 2, 4, 8], [(3, 2, 1)]),
        ([6, 2, 4, 8], [(0, 1)]),
        ([6, 2, 4, 8], [(2, 3)]),
        ([6, 2, 4, 8], [(0, 1, 2, 3)]),
        ([6, 2, 4, 8], [(2, 3, 1, 0)]),
        ([6, 2, 4, 8], [(0, 1), (2, 3)]),
        ([2, 2, 2, 2, 2], [(0, 1), (2, 3, 4)]),
        ([2, 2, 2, 2, 2, 2], [(0, 1), (2, 3), (4, 5)]),
        ([2, 2, 2, 2, 2, 2], [(0, 1), (4, 5), (2, 3)]),
        ([4, 2, 6, 2], [(0, 3)]),
        ([2, 2, 2, 2, 2, 2, 2, 2], [(5,), (7, 2, 3), (1, 4)]),
    ],
)
@pytest.mark.parametrize("charge", [0, 1])
def test_fuse_roundtrip(symmetry, shape, axes_groups, charge):
    sx = get_zn_blocksparse_flat_compat(symmetry, shape, charge, seed=42)
    sy = sx.fuse(*axes_groups)
    fx = sx.to_flat()
    fx.check()
    assert fx.to_blocksparse().allclose(sx)
    fy = fx.fuse(*axes_groups)
    fy.check()
    assert fy.to_blocksparse().allclose(sy)
    xu = fy.unfuse_all()
    xu.check()
    # fuse + unfuse is identity up to permutation of axes
    axes_grouped = [i for g in axes_groups for i in g]
    axes_rem = [i for i in range(fx.ndim) if i not in axes_grouped]
    ax_g0 = min(axes_grouped)
    new_axes = axes_rem[:ax_g0] + axes_grouped + axes_rem[ax_g0:]
    sxt = sx.transpose(new_axes)
    fxus = xu.to_blocksparse()
    assert fxus.allclose(sxt)


@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize(
    "symmetry,shape,newshape",
    [
        ["Z2", (2, 2, 2, 2), (4, 2, 2)],
        ["Z2", (2, 2, 2, 2), (2, 4, 2)],
        ["Z2", (2, 2, 2, 2), (2, 2, 4)],
        ["Z2", (2, 2, 2, 2), (4, 4)],
        ["Z3", (3, 3, 3, 3), (9, 3, 3)],
        ["Z3", (3, 3, 3, 3), (3, 9, 3)],
        ["Z3", (3, 3, 3, 3), (3, 3, 9)],
        ["Z3", (3, 3, 3, 3), (9, 9)],
        ["Z4", (4, 4, 4, 4), (16, 4, 4)],
        ["Z4", (4, 4, 4, 4), (4, 16, 4)],
        ["Z4", (4, 4, 4, 4), (4, 4, 16)],
        ["Z4", (4, 4, 4, 4), (16, 16)],
        ["Z4", (4, 4, 4, 4, 4), (64, 4, 4)],
        ["Z4", (4, 4, 4, 4, 4), (4, 64, 4)],
        ["Z4", (4, 4, 4, 4, 4), (4, 4, 64)],
    ],
)
def test_reshape_roundtrip(symmetry, charge, shape, newshape):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape,
        charge,
        seed=42,
        shape_relative_to_z2=False,
    )
    fx = sx.to_flat()
    sy = sx.reshape(newshape)
    fy = fx.reshape(newshape)
    fy.check()
    assert fy.to_blocksparse().allclose(sy)
    fz = fy.reshape(shape)
    fz.check()
    assert fz.to_blocksparse().allclose(sx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("charge_x", [0, 1])
@pytest.mark.parametrize("charge_y", [0, 1])
def test_matmul(symmetry, seed, charge_x, charge_y):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)
    dc = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)
    c_ind = sr.utils.rand_index(symmetry, N * dc, subsizes="equal", seed=rng)

    if charge_x:
        charge_x = rng.integers(1, N)
    if charge_y:
        charge_y = rng.integers(1, N)

    sx = sr.utils.get_rand(
        symmetry, (a_ind, b_ind.conj()), charge=charge_x, seed=rng
    )
    sy = sr.utils.get_rand(symmetry, (b_ind, c_ind), charge=charge_y, seed=rng)
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("charge_x", [0, 1])
@pytest.mark.parametrize("charge_y", [0, 1])
def test_matvec(symmetry, seed, charge_x, charge_y):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)

    if charge_x:
        charge_x = rng.integers(1, N)
    if charge_y:
        charge_y = rng.integers(1, N)

    sx = sr.utils.get_rand(
        symmetry, (a_ind, b_ind.conj()), charge=charge_x, seed=rng
    )
    sy = sr.utils.get_rand(symmetry, (b_ind,), charge=charge_y, seed=rng)
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("charge_x", [0, 1])
@pytest.mark.parametrize("charge_y", [0, 1])
def test_vecmat(symmetry, seed, charge_x, charge_y):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)

    if charge_x:
        charge_x = rng.integers(1, N)
    if charge_y:
        charge_y = rng.integers(1, N)

    sx = sr.utils.get_rand(symmetry, (a_ind,), charge=charge_x, seed=rng)
    sy = sr.utils.get_rand(
        symmetry, (a_ind.conj(), b_ind), charge=charge_y, seed=rng
    )
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_build_cyclic_keys_conserve(ndim, order, seed):
    import numpy as np

    from symmray.flat_core import (
        build_cyclic_keys_conserve,
        lexsort_sectors,
        zn_combine,
    )

    rng = random.Random(seed)
    charge = rng.randint(0, order - 1)
    duals = [rng.choice([True, False]) for _ in range(ndim)]

    sectors = build_cyclic_keys_conserve(
        ndim,
        order=order,
        charge=charge,
        duals=duals,
        flat=True,
    )
    scharges = zn_combine(order, sectors, duals=duals)

    assert set(map(int, scharges)) == {charge}
    assert np.all(lexsort_sectors(sectors) == np.arange(order ** (ndim - 1)))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_build_cyclic_keys_by_charge(ndim, order, seed):
    import numpy as np

    from symmray.flat_core import (
        build_cyclic_keys_by_charge,
        lexsort_sectors,
        zn_combine,
    )

    rng = random.Random(seed)
    duals = [rng.choice([True, False]) for _ in range(ndim)]

    sectors = build_cyclic_keys_by_charge(
        ndim,
        order=order,
        duals=duals,
    )
    scharges = zn_combine(order, sectors, duals=duals)

    for i in range(order):
        # all have matching charge
        assert np.all(scharges[i] == i)
        # and are sorted within that charge
        assert np.all(
            lexsort_sectors(sectors[i]) == np.arange(order ** (ndim - 1))
        )


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("seed", range(50))
def test_tensordot(symmetry, seed):
    from symmray.utils_test import rand_valid_tensordot

    N = int(symmetry[1:])

    a, b, axes = rand_valid_tensordot(
        symmetry,
        seed=seed,
        dimension_multiplier=N,
        subsizes="equal",
    )
    c = sr.tensordot(a, b, axes, preserve_array=True)

    fa = a.to_flat()
    fb = b.to_flat()

    fc = sr.tensordot(
        fa,
        fb,
        axes,
        preserve_array=True,
    )
    fc.check()

    if c.is_zero() and fc.is_zero():
        # both are zero, other tests might break
        return

    assert (
        len(
            set(map(float, fc.blocks.round(9).flat))
            - set(map(float, c.to_flat().blocks.round(9).flat))
        )
        == 0
    )

    assert fc.to_blocksparse().allclose(c)
