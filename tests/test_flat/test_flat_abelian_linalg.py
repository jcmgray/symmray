import autoray as ar
import pytest

import symmray as sr

from .test_flat_abelian_array import get_zn_blocksparse_flat_compat


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d1", [6, 8])
@pytest.mark.parametrize("d2", [6, 8])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 34])
def test_flat_qr(symmetry, d1, d2, charge, seed):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d1, d2],
        charge=charge,
        seed=seed,
    )
    sQ, _, sR = sr.linalg.qr_stabilized(sx)
    fx = sx.to_flat()
    fx.check()
    fQ, _, fR = sr.linalg.qr_stabilized(fx)
    fQ.check()
    fR.check()

    assert fQ.charge == fx.charge

    assert fQ.to_blocksparse().allclose(sQ)
    assert fR.to_blocksparse().allclose(sR)

    fy = fQ @ fR
    fy.check()

    assert fy.charge == fx.charge

    assert fy.allclose(fx)
    assert fy.to_blocksparse().allclose(sx)


@pytest.mark.parametrize("symmetry", ("Z2",))
@pytest.mark.parametrize("seed", range(5))
def test_qr_with_expand_dims(symmetry, seed):
    pytest.xfail("Unfusing single dimensions not implemented yet")

    x = sr.utils.get_rand(
        symmetry,
        [4, 6, 6],
        seed=seed,
        flat=True,
        subsizes="maximal",
    )
    y = x.reshape((1, 4 * 6 * 6))
    q, r = sr.linalg.qr(y)
    xqr = q @ r
    z = xqr.reshape((4, 6, 6))
    z.test_allclose(x)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d1", [6, 8])
@pytest.mark.parametrize("d2", [6, 8])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 34])
def test_flat_svd(symmetry, d1, d2, charge, seed):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d1, d2],
        charge=charge,
        seed=seed,
    )
    sU, sS, sVh = sr.linalg.svd(sx)
    fx = sx.to_flat()
    fx.check()
    fU, fS, fVh = sr.linalg.svd(fx)
    fU.check()
    fVh.check()

    assert fU.charge == fx.charge
    assert fS.to_blockvector().allclose(sS)

    assert fU.to_blocksparse().allclose(sU)
    assert fVh.to_blocksparse().allclose(sVh)

    fy = fU @ fVh.multiply_diagonal(fS, 0)
    fy.check()
    assert fy.charge == fx.charge
    assert fy.allclose(fx)
    assert fy.to_blocksparse().allclose(sx)

    fy = fU.multiply_diagonal(fS, 1) @ fVh
    fy.check()
    assert fy.charge == fx.charge
    assert fy.allclose(fx)
    assert fy.to_blocksparse().allclose(sx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d", [4, 8])
@pytest.mark.parametrize("seed", [42, 34])
@pytest.mark.parametrize("duals", [(0, 1), (1, 0)])
def test_eigh_flat(symmetry, d, seed, duals):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d, d],
        duals=duals,
        charge=0,
        seed=seed,
    )
    # need to make sure x is hermitian
    fx = sx.to_flat()

    sx = sx + sx.H
    fx = fx + fx.H

    fx.check()
    fevals, fvecs = sr.linalg.eigh(fx)
    fvecs.check()
    fevals.check()
    fy = fvecs.multiply_diagonal(fevals, 1) @ fvecs.H
    fy.check()
    assert fy.allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d1", [6, 8])
@pytest.mark.parametrize("d2", [6, 8])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 34])
def test_flat_svd_via_eig(symmetry, d1, d2, charge, seed):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d1, d2],
        charge=charge,
        seed=seed,
    )
    fx = sx.to_flat()
    fx.check()

    fU, fS, fVh = fx.svd_via_eig()
    fU.check()
    fVh.check()

    assert fU.charge == fx.charge

    # roundtrip via right multiply
    fy = fU @ fVh.multiply_diagonal(fS, 0)
    fy.check()
    assert fy.charge == fx.charge
    assert fy.allclose(fx)

    # roundtrip via left multiply
    fy = fU.multiply_diagonal(fS, 1) @ fVh
    fy.check()
    assert fy.charge == fx.charge
    assert fy.allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d1", [6, 8])
@pytest.mark.parametrize("d2", [6, 8])
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize("seed", [42, 34])
def test_flat_svd_via_eig_truncated(symmetry, d1, d2, absorb, seed):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d1, d2],
        seed=seed,
    )
    fx = sx.to_flat()
    fx.check()

    u, s, vh = fx.svd_via_eig_truncated(absorb=absorb)
    u.check()
    vh.check()

    if absorb is None:
        s.check()
        xr = u @ vh.multiply_diagonal(s, 0)
    else:
        assert s is None
        xr = u @ vh

    xr.check()
    assert xr.allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d1", [6, 8])
@pytest.mark.parametrize("d2", [6, 8])
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize("seed", [42, 34])
def test_flat_svd_via_eig_truncated_max_bond(symmetry, d1, d2, absorb, seed):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d1, d2],
        seed=seed,
    )
    fx = sx.to_flat()
    fx.check()

    u, s, vh = fx.svd_via_eig_truncated(max_bond=4, absorb=absorb)
    u.check()
    vh.check()

    if absorb is None:
        s.check()
        assert s.size <= 4
    else:
        assert s is None

    bond_u = u.indices[1].charge_size
    bond_vh = vh.indices[0].charge_size
    assert bond_u == bond_vh


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d1", [6, 8])
@pytest.mark.parametrize("d2", [6, 8])
def test_flat_svd_via_eig_truncated_ar_dispatch(symmetry, d1, d2, seed=42):
    """Check that autoray dispatch works for svd_via_eig_truncated."""
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[d1, d2],
        seed=seed,
    )
    fx = sx.to_flat()

    u, s, vh = ar.do("svd_via_eig_truncated", fx, max_bond=4, absorb=None)
    u.check()
    vh.check()
    s.check()
    assert s.size <= 4


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("dtype", ["float64", "complex128", "complex64"])
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_flat(symmetry, d, seed, dtype):
    sx = sr.utils_test.rand_posdef(
        symmetry,
        d * int(symmetry[1:]),
        seed=seed,
        dtype=dtype,
        subsizes="equal",
    )
    sleft = sr.linalg.cholesky(sx, upper=False)
    sright = sr.linalg.cholesky(sx, upper=True)

    fx = sx.to_flat()
    fleft = sr.linalg.cholesky(fx)
    fleft.check()
    assert fleft.ndim == 2
    assert fleft.dtype == dtype
    fleft.to_blocksparse().test_allclose(sleft)
    # roundtrip: L @ L^H == A
    fy = fleft @ fleft.H
    fy.check()
    fy.test_allclose(fx)
    fy.to_blocksparse().test_allclose(sx)

    fright = sr.linalg.cholesky(fx, upper=True)
    fright.check()
    assert fright.ndim == 2
    assert fright.dtype == dtype
    fright.to_blocksparse().test_allclose(sright)
    # roundtrip: U^H @ U == A
    fy = fright.H @ fright
    fy.check()
    fy.test_allclose(fx)
    fy.to_blocksparse().test_allclose(sx)

    # combine lower and upper factors
    fy = fleft @ fright
    fy.check()
    fy.test_allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d", [4, 8])
@pytest.mark.parametrize("absorb", [-12, 0, 12])
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_regularized_flat(symmetry, d, seed, absorb, dtype):
    sx = sr.utils_test.rand_posdef(
        symmetry,
        d * int(symmetry[1:]),
        seed=seed,
        subsizes="equal",
        dtype=dtype,
    )
    sleft, _, sright = sr.linalg.cholesky_regularized(sx, absorb=absorb)

    fx = sx.to_flat()
    fleft, fs, fright = sr.linalg.cholesky_regularized(fx, absorb=absorb)
    assert fs is None

    if absorb == -12:
        assert fright is None
        fleft.check()
        fleft.to_blocksparse().test_allclose(sleft)
        y = fleft @ fleft.H
        y.check()
        y.test_allclose(fx)
    elif absorb == 12:
        assert fleft is None
        fright.check()
        fright.to_blocksparse().test_allclose(sright)
        y = fright.H @ fright
        y.check()
        y.test_allclose(fx)
    else:
        fleft.check()
        fright.check()
        # roundtrip: L @ R == A
        fy = fleft @ fright
        fy.check()
        fy.test_allclose(fx)
        fleft.to_blocksparse().test_allclose(sleft)
        fright.to_blocksparse().test_allclose(sright)


def test_cholesky_regularized_flat_ar_dispatch():
    """Check that autoray dispatch works for cholesky_regularized."""
    sx = sr.utils_test.rand_posdef(
        "Z4",
        8,
        seed=42,
        subsizes="equal",
    )
    fx = sx.to_flat()

    left, s, right = ar.do("cholesky_regularized", fx, absorb=0)
    assert s is None
    left.check()
    right.check()
    fy = left @ right
    fy.check()
    fy.test_allclose(fx)
