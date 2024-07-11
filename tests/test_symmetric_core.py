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
    assert ix.copy().chargemap is not ix.chargemap
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


@pytest.mark.parametrize("symmetry", ("Z2",))
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


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2"))
@pytest.mark.parametrize("missing", (False, True))
def test_AbelianArray_fuse(symmetry, missing):
    x = sr.utils.get_rand(symmetry, (3, 4, 5, 6))

    if missing:
        sector_to_drop = list(x.sectors)[len(x.sectors) // 2]
        x.blocks.pop(sector_to_drop)

    xf = x.fuse((0, 2), (1, 3))
    if symmetry == "Z2":
        assert xf.shape == (15, 24)
        assert xf.num_blocks == 2
    xu = xf.unfuse_all().transpose((0, 2, 1, 3))
    assert x.allclose(xu)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize(
    "shape0, shape1",
    [
        ((3, 4, 5, 6), (3, 4, 5, 6)),
        ((3, 4, 5, 6), (12, 30)),
        ((3, 4, 5, 6), (3, 20, 6)),
        ((1, 1, 1, 1), (1, 1)),
    ],
)
def test_AbelianArray_reshape(symmetry, shape0, shape1):
    x = sr.utils.get_rand(symmetry, shape0)
    y = ar.do("reshape", x, shape1)
    assert all(da <= db for da, db in zip(y.shape, shape1))
    y.check()
    z = ar.do("reshape", y, shape0)
    z.check()
    assert x.allclose(z)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("subsizes", ("maximal", "equal"))
@pytest.mark.parametrize(
    "shape1,shape2,axes",
    [
        ((10,), (10,), 1),
        ((4, 5), (4, 5), 2),
        ((4,), (5,), 0),
    ],
)
def test_tensordot(symmetry, shape1, shape2, axes, subsizes):
    a = sr.utils.get_rand(
        symmetry,
        shape1,
        duals=[False] * len(shape1),
        charge=1,
        subsizes=subsizes,
    )
    b = sr.utils.get_rand(
        symmetry,
        shape2,
        duals=[True] * len(shape2),
        charge=1,
        subsizes=subsizes,
    )
    c = ar.do("tensordot", a, b, axes=axes)
    d = ar.do("tensordot", a.to_dense(), b.to_dense(), axes=axes)

    if isinstance(c, sr.AbelianArray):
        if symmetry == "U1":
            assert c.charge == 2
        else:
            assert c.charge == 0
        assert_allclose(c.to_dense(), d)
    else:
        assert_allclose(c, d)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
def test_AbelianArray_reductions(symmetry):
    x = sr.utils.get_rand(symmetry, (3, 4, 5, 6), dist="uniform")
    assert ar.do("min", x) < ar.do("max", x) < ar.do("sum", x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
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
