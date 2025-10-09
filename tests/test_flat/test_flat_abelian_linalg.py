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
