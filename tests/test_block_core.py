import autoray as ar
import pytest
from numpy.testing import assert_allclose

import symmray as sr


@pytest.mark.parametrize("flow", (False, True))
def test_block_index_basics(flow):
    ix = sr.BlockIndex(
        chargemap={
            -2: 3,
            0: 1,
            1: 2,
        },
        flow=flow,
    )
    ix.check()
    assert ix.size_total == 6
    assert ix.num_charges == 3
    assert ix.copy().chargemap is not ix.chargemap
    assert ix.size_of(-2) == 3
    assert ix.matches(ix.conj())
    assert repr(ix)


def test_block_array_to_dense():
    x = sr.utils.get_rand_z2array((3, 4, 5, 6))
    x.check()
    assert x.shape == (3, 4, 5, 6)
    assert x.ndim == 4
    assert x.num_blocks == 8
    assert x.get_sparsity() == 1
    assert x.allclose(x.copy())


def test_blockarray_to_dense():
    x = sr.utils.get_rand_z2array((3, 4, 5, 6))
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


def test_blockarray_fuse():
    x = sr.utils.get_rand_z2array((3, 4, 5, 6))
    xf = x.fuse((0, 2), (1, 3))
    assert xf.shape == (15, 24)
    assert xf.num_blocks == 2
    xu = xf.unfuse_all().transpose((0, 2, 1, 3))
    assert x.allclose(xu)


@pytest.mark.parametrize(
    "shape1,shape2,axes",
    [
        ((10,), (10,), 1),
        ((4, 5), (4, 5), 2),
        ((4,), (5,), 0),
    ],
)
def test_tensordot(shape1, shape2, axes):
    a = sr.utils.get_rand_z2array(
        shape1, flows=[False] * len(shape1), charge_total=1
    )
    b = sr.utils.get_rand_z2array(
        shape2, flows=[True] * len(shape2), charge_total=1
    )
    c = ar.do("tensordot", a, b, axes=axes)
    assert c.charge_total == 0
    d = ar.do("tensordot", a.to_dense(), b.to_dense(), axes=axes)
    assert_allclose(c.to_dense(), d)


def test_blockarray_reductions():
    x = sr.utils.get_rand_z2array((3, 4, 5, 6), dist="uniform")
    assert ar.do("min", x) < ar.do("max", x) < ar.do("sum", x)
