import autoray as ar
import pytest
from numpy.testing import assert_allclose

import symmray as sr


@pytest.mark.parametrize("dual", (False, True))
def test_block_index_basics(dual):
    ix = sr.BlockIndex(
        chargemap={
            -2: 3,
            0: 1,
            1: 2,
        },
        dual=dual,
    )
    ix.check()
    assert ix.size_total == 6
    assert ix.num_charges == 3
    assert ix.size_of(-2) == 3
    assert ix.matches(ix.conj())
    assert repr(ix)


def test_z2symmetric_array_basics():
    x = sr.utils.get_rand_z2array((3, 4, 5, 6))
    x.check()
    assert x.shape == (3, 4, 5, 6)
    assert x.ndim == 4
    assert x.num_blocks == 8
    assert x.get_sparsity() == 1
    assert x.allclose(x.copy())


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
def test_AbelianArray_to_dense(symmetry):
    x = sr.utils.get_rand(symmetry, (3, 4, 5, 6))
    assert ar.do("linalg.norm", x) == pytest.approx(
        ar.do("linalg.norm", x.to_dense())
    )
    assert ar.do("linalg.norm", 2 * x) == pytest.approx(
        ar.do("linalg.norm", 2 * x.to_dense())
    )
    assert_allclose(
        x.transpose((3, 1, 2, 0)).to_dense(),
        x.to_dense().transpose((3, 1, 2, 0)),
    )


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("missing", (False, True))
@pytest.mark.parametrize("mode", ["insert", "concat"])
def test_AbelianArray_fuse(symmetry, missing, mode):
    x = sr.utils.get_rand(symmetry, (3, 4, 5, 6))

    if missing:
        sector_to_drop = list(x.sectors)[len(x.sectors) // 2]
        x.blocks.pop(sector_to_drop)

    xf = x.fuse((0, 2), (1, 3), mode=mode)
    if symmetry == "Z2":
        assert xf.shape == (15, 24)
        assert xf.num_blocks == 2
    xu = xf.unfuse_all().transpose((0, 2, 1, 3))
    assert x.allclose(xu)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize(
    "shape0, shape1",
    [
        ((3, 4, 5, 6), (3, 4, 5, 6)),
        ((3, 4, 5, 6), (12, 30)),
        ((3, 4, 5, 6), (3, 20, 6)),
        ((2, 2, 2, 2, 2), (4, 2, 4)),
        ((2, 2, 2, 1, 2, 2), (1, 4, 2, 4, 1)),
        ((1, 1, 1, 1), (1, 1)),
        ((1, 1, 1), (1, 1, 1)),
        ((1, 1), (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("seed", range(5))
def test_AbelianArray_reshape(symmetry, shape0, shape1, seed):
    x = sr.utils.get_rand(symmetry, shape0, seed=seed, subsizes="maximal")
    y = ar.do("reshape", x, shape1)
    assert all(da <= db for da, db in zip(y.shape, shape1))
    y.check()
    z = ar.do("reshape", y, shape0)
    z.check()
    assert x.allclose(z)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("mode", ["insert", "concat"])
def test_abelian_reshape_unfuse(symmetry, seed, mode):
    a = sr.utils.get_rand(
        symmetry, (3, 4, 5, 6), seed=seed, subsizes="maximal"
    )
    b = a.fuse((0, 1), (2, 3), mode=mode)
    c = b.reshape((12, 5, 6))
    assert c.allclose(a.fuse((0, 1), mode=mode))


def test_calc_reshape_args_edgecase():
    from symmray.abelian_core import calc_reshape_args

    axs_unfuse, axs_fuse_groupings, axs_expand = calc_reshape_args(
        shape=(4, 4, 4), newshape=(16, 4), subsizes=(None, None, (4, 4))
    )
    assert axs_unfuse == ()
    assert axs_expand == ()
    assert axs_fuse_groupings == (((0, 1),),)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("subsizes", ("maximal", "equal"))
@pytest.mark.parametrize("mode", ("fused", "blockwise"))
@pytest.mark.parametrize(
    "shape1,shape2,axes",
    [
        ((10,), (10,), 1),
        ((4, 5), (4, 5), 2),
        ((4,), (5,), 0),
    ],
)
def test_tensordot(symmetry, shape1, shape2, axes, subsizes, mode):
    if symmetry in ("Z2", "U1"):
        charge = 1
    else:
        charge = (1, 1)

    a = sr.utils.get_rand(
        symmetry,
        shape1,
        duals=[False] * len(shape1),
        charge=charge,
        subsizes=subsizes,
    )
    b = sr.utils.get_rand(
        symmetry,
        shape2,
        duals=[True] * len(shape2),
        charge=charge,
        subsizes=subsizes,
    )

    with sr.default_tensordot_mode(mode):
        c = ar.do("tensordot", a, b, axes=axes)
        assert sr.get_default_tensordot_mode() == mode
    assert sr.get_default_tensordot_mode() == "auto"

    d = ar.do("tensordot", a.to_dense(), b.to_dense(), axes=axes)

    if isinstance(c, sr.AbelianArray):
        if symmetry == "Z2":
            assert c.charge == 0
        elif symmetry == "U1":
            assert c.charge == 2
        elif symmetry == "Z2Z2":
            assert c.charge == (0, 0)
        elif symmetry == "U1U1":
            assert c.charge == (2, 2)
        if c.shape == d.shape:
            # XXX: can only check when no empty charges have been dropped
            assert_allclose(c.to_dense(), d)
        elif c.blocks:
            # misaligned charges are all zero entries -> check sum
            assert_allclose(d.sum(), c.sum())
    else:
        assert_allclose(c, d)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
def test_AbelianArray_reductions(symmetry):
    x = sr.utils.get_rand(symmetry, (3, 4, 5, 6), dist="uniform", seed=42)
    assert ar.do("min", x) < ar.do("max", x) < ar.do("sum", x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
def test_block_multiply_diagonal(symmetry):
    import numpy as np

    rng = np.random.default_rng(42)
    x = sr.utils.get_rand(symmetry, (3, 4, 5, 6))
    axis = 2
    v = sr.BlockVector(
        {c: rng.normal(size=d) for c, d in x.indices[axis].chargemap.items()}
    )
    y = ar.do("multiply_diagonal", x, v, axis=axis)
    yd = y.to_dense()

    assert_allclose(
        yd,
        np.einsum("abcd,c->abcd", x.to_dense(), v.to_dense()),
    )


@pytest.mark.parametrize(
    "eq",
    (
        "ab->ab",
        "aa->",
        "aabb->",
        "abab->",
        "cba->abc",
        "cbabc->a",
    ),
)
@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
def test_einsum_single_term(eq, symmetry):
    lhs, *_ = eq.split("->")

    indices = {
        "a": sr.utils.rand_index(symmetry, 3),
        "b": sr.utils.rand_index(symmetry, 4),
        "c": sr.utils.rand_index(symmetry, 5),
        "d": sr.utils.rand_index(symmetry, 6),
    }

    shape = []
    seen = set()
    for q in lhs:
        if q in seen:
            shape.append(indices[q].conj())
        else:
            shape.append(indices[q])
            seen.add(q)

    x = sr.utils.get_rand(symmetry, shape)
    x.check()
    dx = x.to_dense()
    y = ar.do("einsum", eq, x)
    y.check()
    dy = y.to_dense()
    assert_allclose(dy, ar.do("einsum", eq, dx))
