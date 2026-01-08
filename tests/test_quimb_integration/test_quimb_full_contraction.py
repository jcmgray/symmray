import pytest

import symmray as sr


def test_random_state():
    import numpy

    seed = 42
    rng = sr.utils.get_rng(numpy.random.RandomState(seed))
    assert rng.normal() == pytest.approx(0.4967141530112327)


cases = [
    ("Z2", 0, 0, 4, "equal", "even", 42, -775.395970220182, 5.361656689527674),
    ("Z2", 0, 1, 4, "equal", "even", 42, -775.395970220182, 5.361656689527674),
    ("Z2", 1, 0, 4, "equal", "even", 42, -34.1714166810151, 5.36165668952767),
    ("Z2", 1, 1, 4, "equal", "even", 42, -34.1714166810151, 5.36165668952767),
    ("Z3", 0, 0, 6, "equal", "even", 42, 1268.2307872974234, 9.01095171457521),
    ("Z3", 0, 1, 6, "equal", "even", 42, 1268.2307872974234, 9.01095171457521),
    ("Z4", 0, 0, 8, "equal", "even", 42, 7323.861783581483, 11.6961500793490),
    ("Z4", 0, 1, 8, "equal", "even", 42, 7323.861783581483, 11.6961500793490),
    ("Z4", 1, 0, 8, "equal", "even", 42, -6445.69661288631, 11.69615007934904),
    ("Z4", 1, 1, 8, "equal", "even", 42, -6445.69661288631, 11.69615007934904),
    ("Z2", 1, 0, 4, "equal", "odd", 42, 1747.7169608702252, 5.36165668952767),
    ("Z2", 1, 1, 4, "equal", "odd", 42, 1747.7169608702252, 5.36165668952767),
]


@pytest.mark.parametrize(
    "symm,fermionic,flat,D,subsizes,site_charge,seed,zex,x5norm", cases
)
def test_tn_contract_exact_rand_reg(
    symm,
    fermionic,
    flat,
    D,
    subsizes,
    site_charge,
    seed,
    zex,
    x5norm,
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

    if site_charge == "even":

        def site_charge(site):
            return 0

    elif site_charge == "odd":

        def site_charge(site):
            return 1

    elif site_charge == "mix":

        def site_charge(site):
            return site % 2

    tn = sr.networks.TN_abelian_from_edges_rand(
        symm,
        edges,
        bond_dim=D,
        seed=rng,
        subsizes=subsizes,
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
    )

    assert tuple(tn.sites) == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    assert tuple(x.signature for x in tn.arrays) == (
        "+++",
        "+++",
        "+++",
        "--+",
        "-++",
        "--+",
        "--+",
        "--+",
        "---",
        "---",
    )
    assert tn.arrays[5].norm() == pytest.approx(x5norm)

    z = tn.contract(all)
    assert z == pytest.approx(zex)


cases = [
    ("Z2", False, False, 4, 5, 3, "random", 42, 1.708875e15, 1.711580e15),
    ("Z2", True, False, 4, 5, 3, "random", 42, 1.424314e15, 1.421489e15),
    ("U1", False, False, 4, 5, 3, "random", 42, 7.713475e07, 7.713475e07),
    ("U1", True, False, 4, 5, 3, "random", 42, 7.658967e07, 7.658967e07),
    ("Z2", False, False, 4, 5, 4, "equal", 42, 1.076008e19, 1.046593e19),
    ("Z2", False, True, 4, 5, 4, "equal", 42, 1.076008e19, 1.046462e19),
    ("Z2", True, False, 4, 5, 4, "equal", 42, 1.207274e19, 1.181436e19),
    ("Z2", True, True, 4, 5, 4, "equal", 42, 1.207274e19, 1.178684e19),
]


@pytest.mark.parametrize(
    "symm,fermionic,flat,Lx,Ly,D,subsizes,seed,zex,zap", cases
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
