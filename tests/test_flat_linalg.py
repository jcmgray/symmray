import pytest
import symmray as sr

from test_flat_core import get_zn_blocksparse_flat_compat


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
