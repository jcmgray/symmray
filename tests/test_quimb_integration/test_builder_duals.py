import inspect

import pytest

import symmray as sr

EDGES = ((0, 1), (1, 2), (2, 3))


def _bond_duals(tn):
    """Return the dualnesses attached to each network index."""
    index_duals = {}
    for tensor in tn.tensors:
        for ind, dual in zip(tensor.inds, tensor.data.duals):
            index_duals.setdefault(ind, []).append(dual)
    return index_duals


def test_parse_edges_default_is_reversed():
    site_info_default = sr.parse_edges_to_site_info(EDGES, 2)
    site_info_reversed = sr.parse_edges_to_site_info(
        EDGES, 2, duals="reversed"
    )

    assert site_info_default == site_info_reversed


def test_parse_edges_canonical_flips_bonds_only():
    reversed_info = sr.parse_edges_to_site_info(EDGES, 2, phys_dim=2)
    canonical_info = sr.parse_edges_to_site_info(
        EDGES, 2, phys_dim=2, duals="canonical"
    )

    for site in reversed_info:
        coordination = reversed_info[site]["coordination"]
        assert canonical_info[site]["duals"][:coordination] == [
            not dual for dual in reversed_info[site]["duals"][:coordination]
        ]
        assert canonical_info[site]["duals"][coordination:] == [False]


def test_parse_edges_explicit_duals_mapping():
    default_info = sr.parse_edges_to_site_info(EDGES, 2, phys_dim=None)
    mapped_info = sr.parse_edges_to_site_info(
        EDGES,
        2,
        phys_dim=None,
        duals={
            "b0-1": True,
            (1, 2): True,
        },
    )

    assert mapped_info[0]["duals"] == [True]
    assert mapped_info[1]["duals"] == [False, True]
    assert mapped_info[2]["duals"] == [False, False]
    assert mapped_info[3]["duals"] == [True]

    assert (
        sr.parse_edges_to_site_info(
            EDGES,
            2,
            phys_dim=None,
            duals={ind: False for ind in ("b0-1", "b1-2", "b2-3")},
        )
        == default_info
    )


def test_parse_edges_rejects_unknown_duals():
    with pytest.raises(ValueError, match="Unrecognized duals"):
        sr.parse_edges_to_site_info(EDGES, 2, duals="unknown")


@pytest.mark.parametrize(
    "builder, args",
    [
        (sr.TN_abelian_from_edges_rand, ("Z2", EDGES, 2)),
        (sr.MPS_abelian_rand, ("Z2", 4, 2)),
        (sr.TN2D_abelian_rand, ("Z2", 2, 2, 2)),
        (sr.TN3D_abelian_rand, ("Z2", 2, 2, 2, 2)),
    ],
)
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_duals_are_consistent(builder, args, fermionic, seed):
    pytest.importorskip("quimb.tensor")

    tn = builder(
        *args,
        phys_dim=2,
        fermionic=fermionic,
        duals="random",
        seed=seed,
    )

    for ind, duals in _bond_duals(tn).items():
        if ind.startswith("k"):
            assert duals == [False]
        else:
            assert len(duals) == 2
            assert duals[0] != duals[1]


def test_random_duals_actually_randomize():
    pytest.importorskip("quimb.tensor")

    reversed_signatures = {
        ind: tuple(duals)
        for ind, duals in _bond_duals(
            sr.TN2D_abelian_rand("Z2", 2, 2, 2, phys_dim=2, seed=0)
        ).items()
    }
    random_signatures = [
        {
            ind: tuple(duals)
            for ind, duals in _bond_duals(
                sr.TN2D_abelian_rand(
                    "Z2", 2, 2, 2, phys_dim=2, duals="random", seed=seed
                )
            ).items()
        }
        for seed in range(5)
    ]

    assert any(
        signature != reversed_signatures for signature in random_signatures
    )


@pytest.mark.parametrize("duals", ["reversed", "canonical", "random"])
def test_fermionic_random_duals_contract_consistently(duals):
    pytest.importorskip("quimb.tensor")

    psi = sr.PEPS_fermionic_rand("Z2", 2, 2, bond_dim=2, seed=42, duals=duals)
    norm = psi.make_norm()
    exact = norm.contract(all)
    boundary = norm.contract_boundary(
        max_bond=16, layer_tags=["KET", "BRA"], cutoff=0.0
    )

    assert boundary == pytest.approx(exact)
    assert psi.H @ psi == pytest.approx(exact)


def test_random_duals_keep_local_operators_compatible():
    pytest.importorskip("quimb.tensor")

    import numpy as np
    import quimb as qu
    import quimb.operator as qop

    edges = ((0, 1), (1, 2))
    terms = sr.ham_fermi_hubbard_spinless_from_edges("Z2", edges)
    psi = sr.TN_fermionic_from_edges_rand(
        "Z2", edges, bond_dim=2, phys_dim=2, seed=42, duals="random"
    )

    energy = psi.compute_local_expectation_exact(terms)
    psi_dense = psi.contract(all).data.fuse(range(3)).phase_sync().blocks[(0,)]
    psi_dense = psi_dense / np.linalg.norm(psi_dense)
    hamiltonian = qop.fermi_hubbard_spinless_from_edges(
        edges, symmetry="Z2", sector=0
    ).build_sparse_matrix()
    assert energy == pytest.approx(
        qu.expec(psi_dense.reshape(-1, 1), hamiltonian)
    )


@pytest.mark.parametrize(
    "builder",
    [
        sr.TN_fermionic_from_edges_rand,
        sr.MPS_fermionic_rand,
        sr.TN2D_fermionic_rand,
        sr.PEPS_fermionic_rand,
        sr.TN3D_fermionic_rand,
        sr.PEPS3D_fermionic_rand,
    ],
)
def test_fermionic_builder_signatures_expose_duals(builder):
    assert "duals" in inspect.signature(builder).parameters
