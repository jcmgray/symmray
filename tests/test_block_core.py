import operator

import pytest
from numpy.testing import assert_allclose

import symmray as sr


@pytest.mark.parametrize("op", ["mul", "truediv"])
def test_block_scalar_operations(op):
    fn = getattr(operator, op)
    x = sr.utils.get_rand_symmetric(
        "Z2", (7, 5, 6), flows=(1, 0, 1), charge_total=0
    )
    y = 3.14
    z = fn(x, y)
    assert_allclose(
        z.to_dense(),
        fn(x.to_dense(), y),
    )


@pytest.mark.parametrize("op", ["mul", "sub", "add", "truediv", "pow"])
def test_block_binary_operations(op):
    fn = getattr(operator, op)
    x = sr.utils.get_rand_blockvector(10, block_size=3)
    y = sr.utils.get_rand_blockvector(10, block_size=3)
    z = fn(x, y)
    assert_allclose(
        z.to_dense(),
        fn(x.to_dense(), y.to_dense()),
    )
