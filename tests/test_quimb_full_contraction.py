import pytest

import symmray as sr


@pytest.mark.parametrize(
    "symm,fermionic,flat,D,subsizes,seed,zex",
    [
        ("Z2", 0, 0, 4, "equal", 42, -290.561185093364),
        ("Z2", 0, 1, 4, "equal", 42, -290.561185093364),
        ("Z2", 1, 0, 4, "equal", 42, 439.79982617186926),
        ("Z2", 1, 1, 4, "equal", 42, 439.79982617186926),
        ("Z3", 0, 0, 6, "equal", 42, -2412.317879145185),
        ("Z3", 0, 1, 6, "equal", 42, -2412.317879145185),
        ("Z3", 1, 0, 6, "equal", 42, 190.10308771834895),
        ("Z3", 1, 1, 6, "equal", 42, 190.10308771834895),
        ("Z4", 0, 0, 8, "equal", 42, -14188.846416304901),
        ("Z4", 0, 1, 8, "equal", 42, -14188.846416304901),
        ("Z4", 1, 0, 8, "equal", 42, 2919.783379564473),
        ("Z4", 1, 1, 8, "equal", 42, 2919.783379564473),
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
    rng = numpy.random.RandomState(seed)

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
        (
            "Z2",
            False,
            False,
            4,
            5,
            3,
            "random",
            42,
            1249314432979625.8,
            1249385300902835.5,
        ),
        (
            "Z2",
            True,
            False,
            4,
            5,
            3,
            "random",
            42,
            1462312495716125.5,
            1461925059050817.5,
        ),
        (
            "U1",
            False,
            False,
            4,
            5,
            3,
            "random",
            42,
            12420102974.57941,
            12420102974.579369,
        ),
        (
            "U1",
            True,
            False,
            4,
            5,
            3,
            "random",
            42,
            12423392649.02872,
            12423392649.028677,
        ),
        (
            "Z2",
            False,
            False,
            4,
            5,
            4,
            "equal",
            42,
            3.699916651272369e18,
            3.5217058858073923e18,
        ),
        (
            "Z2",
            False,
            True,
            4,
            5,
            4,
            "equal",
            42,
            3.699916651272369e18,
            3.5167108363602043e18,
        ),
        (
            "Z2",
            True,
            False,
            4,
            5,
            4,
            "equal",
            42,
            4.2460299722958316e18,
            4.125147271299068e18,
        ),
        (
            "Z2",
            True,
            True,
            4,
            5,
            4,
            "equal",
            42,
            4.2460299722958316e18,
            4.1306686155615053e18,
        ),
    ],
)
def test_peps_approx_norm_abelian(
    symm, fermionic, flat, Lx, Ly, D, subsizes, seed, zex, zap
):
    pytest.importorskip("quimb")

    import numpy

    # this guarantees state across versions
    rng = numpy.random.RandomState(seed)

    if fermionic and flat:

        def site_charge(coo):
            return 0

    else:

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
