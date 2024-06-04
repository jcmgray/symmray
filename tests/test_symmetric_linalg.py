import autoray as ar
import pytest
from numpy.testing import assert_allclose

import symmray as sr


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_qr_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand_symmetric(
        symmetry, (d0, d1), flows=[f0, f1], charge_total=c
    )
    x.check()
    q, r = sr.linalg.qr(x)
    q.check()
    r.check()
    assert sr.tensordot(q, r, 1).allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_svd_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand_symmetric(
        symmetry, (d0, d1), flows=[f0, f1], charge_total=c
    )
    x.check()
    u, s, vh = sr.linalg.svd(x)
    u.check()
    s.check()
    vh.check()
    us = ar.do("multiply_diagonal", u, s, axis=1)
    usvh = sr.tensordot(us, vh, 1)
    assert usvh.allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
def test_expm_with_reshape(symmetry, d):
    x = sr.utils.get_rand_symmetric(
        symmetry, (d, d, d, d), flows=[0, 0, 1, 1],
    )
    x_matrix = ar.do("reshape", x, (d**2, d**2))
    # == x_matrix = x.fuse((0, 1), (2, 3))
    xe_matrix = ar.do("scipy.linalg.expm", x_matrix)
    xe = ar.do("reshape", xe_matrix, (d, d, d, d))
    #  == xe = xe_matrix.unfuse_all()
    xe_dense = ar.do(
        "scipy.linalg.expm",
        x.to_dense().reshape((d**2, d**2))
    ).reshape((d, d, d, d))
    assert_allclose(xe.to_dense(), xe_dense)
