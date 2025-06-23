import pytest
import symmray as sr


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
    if symmetry == "Z3":
        shape = [3 * d // 2 for d in shape]
    elif symmetry == "Z4":
        shape = [4 * d // 2 for d in shape]

    xs = sr.utils.get_rand(
        symmetry, shape=shape, subsizes="equal", charge=charge, seed=42
    )
    ys = xs.fuse(*axes_groups)
    x = xs.to_flat()
    x.check()
    assert x.to_blocksparse().allclose(xs)
    y = x.fuse(*axes_groups)
    y.check()
    assert y.to_blocksparse().allclose(ys)
    xu = y.unfuse_all()
    xu.check()
    # fuse + unfuse is identity up to permutation of axes
    axes_grouped = [i for g in axes_groups for i in g]
    axes_rem = [i for i in range(x.ndim) if i not in axes_grouped]
    ax_g0 = min(axes_grouped)
    new_axes = axes_rem[:ax_g0] + axes_grouped + axes_rem[ax_g0:]
    xst = xs.transpose(new_axes)
    xus = xu.to_blocksparse()
    assert xus.allclose(xst)
