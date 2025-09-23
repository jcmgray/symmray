import pytest

import symmray as sr


@pytest.mark.parametrize(
    "symm,fermionic,flat,D,subsizes,seed,zex",
    [
        ("Z2", 0, 0, 4, "equal", 42, 910.9953756296064),
        ("Z2", 0, 1, 4, "equal", 42, 910.9953756296064),
        ("Z2", 1, 0, 4, "equal", 42, -559.9041778658463),
        ("Z2", 1, 1, 4, "equal", 42, -559.9041778658463),
        ("Z3", 0, 0, 6, "equal", 42, -2273.9553413723775),
        ("Z3", 0, 1, 6, "equal", 42, -2273.9553413723775),
        ("Z3", 1, 0, 6, "equal", 42, 2790.4191674364483),
        ("Z3", 1, 1, 6, "equal", 42, 2790.4191674364483),
        ("Z4", 0, 0, 8, "equal", 42, -28168.346984047483),
        ("Z4", 0, 1, 8, "equal", 42, -28168.346984047483),
        ("Z4", 1, 0, 8, "equal", 42, 21259.573156456645),
        ("Z4", 1, 1, 8, "equal", 42, 21259.573156456645),
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

    import numpy

    # this guarantees state across versions
    rng = numpy.random.RandomState(42)

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
        seed=rng,
        subsizes=subsizes,
        fermionic=fermionic,
        flat=flat,
    )

    z = tn.contract(all)
    assert z == pytest.approx(zex)


@pytest.mark.parametrize(
    "symm,fermionic,flat,Lx,Ly,D,subsizes,seed,zex,zap",
    [
        ("Z2", 0, 0, 4, 5, 3, "random", 42, 932868121102350, 931801000689350),
        ("Z2", 1, 0, 4, 5, 3, "random", 42, 836370835922494, 836451499443962),
        ("U1", 0, 0, 4, 5, 3, "random", 42, 2616935568011.6, 2616935568011.69),
        ("U1", 1, 0, 4, 5, 3, "random", 42, 3347515047752.4, 3347515047752.46),
        ("Z2", 0, 0, 4, 5, 4, "equal", 42, 7.194706515e18, 6.8934743978e18),
        ("Z2", 0, 1, 4, 5, 4, "equal", 42, 7.194706515e18, 6.8947000425e18),
    ],
)
def test_peps_approx_norm_abelian(
    symm, fermionic, flat, Lx, Ly, D, subsizes, seed, zex, zap
):
    pytest.importorskip("quimb")

    import numpy

    # this guarantees state across versions
    rng = numpy.random.RandomState(seed)

    def site_charge(coo):
        return (coo[0] + coo[1]) % 2

    psi = sr.networks.PEPS_abelian_rand(
        symm,
        Lx,
        Ly,
        D,
        seed=rng,
        subsizes=subsizes,
        site_charge=site_charge,
        fermionic=fermionic,
        flat=flat,
    )

    psi.check()
    norm = psi.make_norm()
    norm.check()

    assert norm.contract(all, optimize="random-greedy-128") == pytest.approx(
        zex
    )

    if flat:
        kwargs = {"cutoff": 0.0}
    else:
        kwargs = {"cutoff": 1e-10}

    assert norm.contract_boundary(
        max_bond=40, layer_tags=["KET", "BRA"], **kwargs
    ) == pytest.approx(zap)
