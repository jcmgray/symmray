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
    x = sr.utils.get_rand(symmetry, (d0, d1), duals=[f0, f1], charge=c)
    x.check()
    q, r = sr.linalg.qr(x)
    q.check()
    r.check()
    assert (q @ r).allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1", "Z2Z2"))
@pytest.mark.parametrize("seed", range(5))
def test_qr_with_expand_dims(symmetry, seed):
    x = sr.utils.get_rand(
        symmetry,
        [4, 5, 6],
        subsizes="maximal",
        seed=seed,
    )
    y = x.reshape(
        (
            1,
            4 * 5 * 6,
        )
    )
    q, r = sr.linalg.qr(y)
    z = (q @ r).reshape((4, 5, 6))
    assert z.allclose(x)


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
        charge=c,
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


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
@pytest.mark.parametrize("seed", range(1))
def test_eigh(symmetry, d, seed):
    x = sr.utils_test.rand_herm(symmetry, d, seed=seed)
    el, ev = ar.do("linalg.eigh", x)
    el.check()
    ev.check()
    xr = ev @ ar.do("multiply_diagonal", ev.H, el, axis=0)
    xr.check()
    assert x.allclose(xr)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky(symmetry, d, seed, dtype):
    x = sr.utils_test.rand_posdef(symmetry, d, seed=seed, dtype=dtype)

    left = sr.linalg.cholesky(x, upper=False)
    left.check()
    assert left.ndim == 2
    assert left.dtype == dtype
    # roundtrip: L @ L^H should equal A
    y = left @ left.H
    y.check()
    y.test_allclose(x)

    right = sr.linalg.cholesky(x, upper=True)
    right.check()
    assert right.ndim == 2
    assert right.dtype == dtype
    # roundtrip: R^H @ R should equal A
    y = right.H @ right
    y.check()
    y.test_allclose(x)

    # check left/lower and right/upper are consistent with each other
    y = left @ right
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 5))
@pytest.mark.parametrize("absorb", [-12, 0, 12])
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_regularized(symmetry, d, absorb, seed, dtype):
    x = sr.utils_test.rand_posdef(symmetry, d, seed=seed, dtype=dtype)

    left, s, right = sr.linalg.cholesky_regularized(x, absorb=absorb)
    assert s is None

    if absorb == -12:
        assert right is None
        left.check()
        # reconstruct
        y = left @ left.H
        y.check()
        y.test_allclose(x)
    elif absorb == 12:
        assert left is None
        right.check()
        # reconstruct
        y = right.H @ right
        y.check()
        y.test_allclose(x)
    else:
        left.check()
        right.check()
        # roundtrip: L @ L^H == A
        y = left @ right
        y.check()
        y.test_allclose(x)


def test_cholesky_regularized_ar_dispatch():
    """Check that autoray dispatch works for cholesky_regularized."""
    x = sr.utils_test.rand_posdef("Z2", 2, seed=0, dtype="complex128")
    left, s, right = ar.do("cholesky_regularized", x)
    assert s is None
    left.check()
    right.check()
    y = left @ right
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 7, 31))
def test_solve(symmetry, d):
    ind = sr.utils.rand_index(symmetry, d)

    a = sr.utils.get_rand(
        symmetry,
        shape=(ind, ind.conj()),
        dtype="complex128",
    )
    b = sr.utils.get_rand(
        symmetry,
        shape=[ind],
        dtype="complex128",
    )
    x = sr.linalg.solve(a, b)
    x.check()
    assert (a @ x).allclose(b)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d0", (2, 7, 31))
@pytest.mark.parametrize("d1", (2, 3, 4))
def test_solve_2d(symmetry, d0, d1):
    ind = sr.utils.rand_index(symmetry, d0)
    ind1 = sr.utils.rand_index(symmetry, d1)

    a = sr.utils.get_rand(
        symmetry,
        shape=(ind, ind.conj()),
        dtype="complex128",
    )
    b = sr.utils.get_rand(
        symmetry,
        shape=(ind, ind1),
        dtype="complex128",
    )
    x = sr.linalg.solve(a, b)
    x.check()
    assert (a @ x).allclose(b)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
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
