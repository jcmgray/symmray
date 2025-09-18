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
    # add some non-trivial phases
    x.transpose((2, 0, 1), inplace=True)
    assert x.phases
    fx = x.to_flat()
    assert fx.fermionic
    if sync:
        fx.phase_sync(inplace=True)
    fx.check()
    y = fx.to_blocksparse()
    assert sync == (not y.phases)
    y.check()
    assert y.allclose(x)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("axs", [(), (0,), (1,), (0, 2), (0, 2, 1)])
def test_phase_flip(
    symmetry,
    charge,
    seed,
    sync,
    axs,
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
    # add some non-trivial phases
    x.transpose((2, 0, 1), inplace=True)
    assert x.phases
    fx = x.to_flat()
    xflipped = x.phase_flip(*axs)
    fxflipped = fx.phase_flip(*axs)
    if sync:
        fxflipped.phase_sync(inplace=True)
    fxflipped.check()
    y = fxflipped.to_blocksparse()
    y.check()
    if sync:
        # phases should have been absorbed into blocks
        assert not y.phases
    assert y.allclose(xflipped)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize(
    "perm",
    [
        None,
        (2, 1, 0),
        (2, 0, 1),
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
    ],
)
def test_phase_transpose(
    symmetry,
    charge,
    seed,
    sync,
    perm,
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
    # add some non-trivial phases
    x.transpose((2, 0, 1), inplace=True)
    assert x.phases
    fx = x.to_flat()

    x_phase_transposed = x.phase_transpose(perm)
    fx_phase_transposed = fx.phase_transpose(perm)
    if sync:
        fx_phase_transposed.phase_sync(inplace=True)
    fx_phase_transposed.check()
    if perm != (0, 1, 2):
        # the phases should have changed
        assert (fx_phase_transposed.phases != fx.phases).sum()
    y = fx_phase_transposed.to_blocksparse()
    y.check()
    if sync:
        # phases should have been absorbed into blocks
        assert not y.phases
    assert y.allclose(x_phase_transposed)
