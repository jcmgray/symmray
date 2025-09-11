import pytest

import symmray as sr


@pytest.mark.parametrize(
    "symm,fermionic,Lx,Ly,D,subsizes,seed,zex,zap",
    [
        ("Z2", 0, 4, 5, 3, None, 42, 18686525784203.750, 18699063660322.18),
        ("Z2", 1, 4, 5, 3, None, 42, 22755076798682.906, 22728959655138.24),
        ("U1", 0, 4, 5, 3, None, 42, 55955059.004159900, 55961396.57935014),
        ("U1", 1, 4, 5, 3, None, 42, 64162935.019977340, 64162323.77904858),
    ],
)
def test_peps_norm_abelian(
    symm, fermionic, Lx, Ly, D, subsizes, seed, zex, zap
):
    # regression test to catch otherwise silent implementation change errors
    pytest.importorskip("quimb")

    def site_charge(coo):
        return (coo[0] + coo[1]) % 2

    psi = sr.networks.PEPS_abelian_rand(
        symm,
        Lx,
        Ly,
        D,
        seed=seed,
        subsizes=subsizes,
        site_charge=site_charge,
        fermionic=fermionic,
    )
    psi.check()
    norm = psi.make_norm()
    norm.check()

    assert norm.contract(all, optimize="random-greedy-128") == pytest.approx(
        zex
    )

    assert norm.contract_boundary(
        max_bond=40,
        layer_tags=["KET", "BRA"],
    ) == pytest.approx(zap)
