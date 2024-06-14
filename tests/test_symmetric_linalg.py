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
    x = sr.utils.get_rand(symmetry, (d0, d1), duals=[f0, f1], charge_total=c)
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
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        duals=[f0, f1],
        charge_total=c,
        subsizes="maximal",
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
def test_eigh(symmetry, d):
    x = sr.utils.get_rand(
        symmetry,
        (d, d),
        duals=[0, 1],
        subsizes="equal",
    )
    # need to make sure x is hermitian
    x.apply_to_arrays(lambda x: (x + x.T) / 2)
    el, ev = ar.do("linalg.eigh", x)
    el.check()
    ev.check()
    xr = ev @ ar.do("multiply_diagonal", ev.H, el, axis=0)
    xr.check()
    assert x.allclose(xr)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
def test_expm_with_reshape(symmetry, d):
    pytest.importorskip("scipy")

    x = sr.utils.get_rand(
        symmetry,
        (d, d, d, d),
        duals=[0, 0, 1, 1],
        subsizes="equal",
    )
    x_matrix = ar.do("reshape", x, (d**2, d**2))
    # == x_matrix = x.fuse((0, 1), (2, 3))
    xe_matrix = ar.do("scipy.linalg.expm", x_matrix)
    xe_matrix.check()
    xe = ar.do("reshape", xe_matrix, (d, d, d, d))
    xe.check()
    #  == xe = xe_matrix.unfuse_all()
    xe_dense = ar.do(
        "scipy.linalg.expm", x.to_dense().reshape((d**2, d**2))
    ).reshape((d, d, d, d))
    assert_allclose(xe.to_dense(), xe_dense)
