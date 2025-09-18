import pytest

from .test_flat_abelian_core import get_zn_blocksparse_flat_compat


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("sync", [False, True])
def test_to_and_from_blocksparse_with_phase_sync(
    symmetry,
    charge,
    seed,
    sync,
):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    x = get_zn_blocksparse_flat_compat(
        symmetry,
        (2, 4, 6),
        charge=charge,
        fermionic=True,
        seed=seed,
    )
    x.transpose((2, 0, 1), inplace=True)
    assert x.phases
    fx = x.to_flat()
    assert fx.fermionic
    if sync:
        fx.phase_sync(inplace=True)
    fx.check()
    y = fx.to_blocksparse()
    if sync:
        assert not y.phases
    else:
        assert y.phases
    y.check()
    assert y.allclose(x)
