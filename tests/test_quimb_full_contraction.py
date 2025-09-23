import pytest

import symmray as sr


@pytest.mark.parametrize(
    "symm,fermionic,flat,D,subsizes,seed,zex",
    [
        ("Z2", 0, 0, 4, "equal", 42, -281.62364214437235),
        ("Z2", 0, 1, 4, "equal", 42, -281.62364214437235),
        ("Z2", 1, 0, 4, "equal", 42, -366.2700849695507),
        ("Z2", 1, 1, 4, "equal", 42, -366.2700849695507),
        ("Z3", 0, 0, 6, "equal", 42, -3370.3874923970207),
        ("Z3", 0, 1, 6, "equal", 42, -3370.3874923970207),
        ("Z3", 1, 0, 6, "equal", 42, -1676.676550645644),
        ("Z3", 1, 1, 6, "equal", 42, -1676.676550645644),
        ("Z4", 0, 0, 8, "equal", 42, -2850.1344930363116),
        ("Z4", 0, 1, 8, "equal", 42, -2850.1344930363116),
        ("Z4", 1, 0, 8, "equal", 42, 512.9367355654399),
        ("Z4", 1, 1, 8, "equal", 42, 512.9367355654399),
    ],
)
def test_tn_contract_exact_rand_reg(
    symm,
    fermionic,
    flat,
    D,
    subsizes,
    seed,
    zex,
):
    pytest.importorskip("quimb")

    # networkx.random_regular_graph(3, 10, seed=42)
    edges = (
        (0, 7),
        (0, 3),
        (0, 6),
        (1, 5),
        (1, 3),
        (1, 9),
        (2, 6),
        (2, 5),
        (2, 8),
        (3, 4),
        (4, 9),
        (4, 8),
        (5, 9),
        (6, 7),
        (7, 8),
    )
    tn = sr.networks.TN_abelian_from_edges_rand(
        symm,
        edges,
        bond_dim=D,
        seed=seed,
        subsizes=subsizes,
        fermionic=fermionic,
    )

    if flat:
        for t in tn:
            t.modify(data=t.data.to_flat())

    z = tn.contract(all)
    assert z == pytest.approx(zex)


@pytest.mark.parametrize(
    "symm,fermionic,Lx,Ly,D,subsizes,seed,zex,zap",
    [
        ("Z2", 0, 4, 5, 3, None, 42, 18686525784203.750, 18699063660322.18),
        ("Z2", 1, 4, 5, 3, None, 42, 22755076798682.906, 22728959655138.24),
        ("U1", 0, 4, 5, 3, None, 42, 55955059.004159900, 55961396.57935014),
        ("U1", 1, 4, 5, 3, None, 42, 64162935.019977340, 64162323.77904858),
    ],
)
def test_peps_approx_norm_abelian(
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
