import operator

import pytest
from numpy.testing import assert_allclose

import symmray as sr


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("op", ["mul", "sub", "add", "truediv", "pow"])
@pytest.mark.parametrize("rhs", [False, True])
def test_flatvector_scalar_operations(symmetry, op, rhs):
    fn = getattr(operator, op)

    x = sr.FlatVector.rand(symmetry, 5, seed=42)

    xd = x.to_dense()
    y = yd = 3.14
    if rhs:
        x, y = y, x
        xd, yd = yd, xd
    z = fn(x, y)
    assert_allclose(
        z.to_dense(),
        fn(xd, yd),
    )


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("op", ["mul", "sub", "add", "truediv", "pow"])
def test_flatvector_binary_operations(symmetry, op):
    fn = getattr(operator, op)
    x = sr.FlatVector.rand(symmetry, 5, seed=42)
    y = sr.FlatVector.rand(symmetry, 5, seed=42)
    z = fn(x, y)
    assert_allclose(
        z.to_dense(),
        fn(x.to_dense(), y.to_dense()),
    )
