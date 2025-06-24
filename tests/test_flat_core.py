import random

import pytest
import symmray as sr


def get_zn_blocksparse_flat_compat(symmetry, shape, charge, seed=42):
    rng = random.Random(seed)

    if symmetry == "Z3":
        shape = [3 * d // 2 for d in shape]
        if charge:
            charge = rng.choice([1, 2])
    elif symmetry == "Z4":
        shape = [4 * d // 2 for d in shape]
        if charge:
            charge = rng.choice([1, 2, 3])

    return sr.utils.get_rand(
        symmetry, shape=shape, subsizes="equal", charge=charge, seed=seed
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


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(10))
def test_matmul(symmetry, seed):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)
    dc = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)
    c_ind = sr.utils.rand_index(symmetry, N * dc, subsizes="equal", seed=rng)
    sx = sr.utils.get_rand(symmetry, (a_ind, b_ind.conj()))
    sy = sr.utils.get_rand(symmetry, (b_ind, c_ind))
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)
