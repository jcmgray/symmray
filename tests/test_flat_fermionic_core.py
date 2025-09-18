import pytest

from .test_flat_abelian_core import get_zn_blocksparse_flat_compat


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_to_and_from_blocksparse(symmetry, charge, seed):
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
    fx.check()
    y = fx.to_blocksparse()
    y.check()
    assert y.allclose(x)
