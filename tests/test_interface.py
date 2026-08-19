import autoray as ar
import numpy as np
import pytest

import symmray as sr


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "complex64", "complex128"],
)
@pytest.mark.parametrize("flat", [False, True])
def test_finfo(dtype, flat):
    x = sr.utils.get_rand(
        "Z2",
        (4, 4),
        seed=42,
        dtype=dtype,
        flat=flat,
        subsizes="equal",
    )

    expected = np.finfo(dtype)
    assert sr.finfo(x.dtype).eps == expected.eps
    assert ar.get_namespace(x).finfo(x.dtype).eps == expected.eps
