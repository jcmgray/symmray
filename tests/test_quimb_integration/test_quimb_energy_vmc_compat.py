import pytest

edges_square_3x2_obc = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 1), (1, 1)),
    ((1, 0), (1, 1)),
    ((1, 0), (2, 0)),
    ((1, 1), (2, 1)),
    ((2, 0), (2, 1)),
)
edges_square_3x2_pbc = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (2, 0)),
    ((0, 1), (1, 1)),
    ((0, 1), (2, 1)),
    ((1, 0), (1, 1)),
    ((1, 0), (2, 0)),
    ((1, 1), (2, 1)),
    ((2, 0), (2, 1)),
)
edges_rrg_d3_n6_seed8 = (
    (0, 4),
    (0, 3),
    (0, 2),
    (1, 2),
    (1, 4),
    (1, 3),
    (2, 5),
    (3, 5),
    (4, 5),
)
edges_rrg_d3_n6_seed9 = (
    (0, 1),
    (0, 4),
    (0, 5),
    (1, 2),
    (1, 3),
    (2, 4),
    (2, 3),
    (3, 5),
    (4, 5),
)


@pytest.mark.parametrize(
    "edges",
    [
        edges_square_3x2_obc,
        edges_square_3x2_pbc,
        edges_rrg_d3_n6_seed8,
        edges_rrg_d3_n6_seed9,
    ],
)
@pytest.mark.parametrize("sector", [0, 1])
@pytest.mark.parametrize("flat", [False, True])
def test_spinless_su_energy_matches_z2(edges, sector, flat):
    import numpy as np
    import quimb as qu
    import quimb.operator as qop
    import quimb.tensor as qtn

    import symmray as sr

    sites = sorted(set().union(*edges))

    # define random fermi-hubbard spinless model,
    # with edge and site dependent coefficients for testing generality
    rng = np.random.default_rng(42)
    tijs = {edge: rng.uniform() for edge in edges}
    Vijs = {edge: rng.uniform() for edge in edges}
    mus = {site: rng.uniform() for site in sites}
    deltas = {edge: rng.uniform() for edge in edges}

    # build reference matrix hamiltonian
    H = qop.fermi_hubbard_spinless_from_edges(
        edges,
        t=tijs,
        V=Vijs,
        mu=mus,
        delta=deltas,
        sector=sector,
        symmetry="Z2",
    )
    H_sparse = H.build_sparse_matrix()
    ge = qu.groundenergy(H_sparse)

    # build local tensor hamiltonian
    terms = sr.ham_fermi_hubbard_spinless_from_edges(
        "Z2",
        edges,
        t=tijs,
        V=Vijs,
        mu=mus,
        delta=deltas,
        flat=flat,
    )
    ham = qtn.LocalHamGen(terms)

    # build peps with different total charges
    if sector == 0:

        def site_charge(site):
            return 0

    elif sector == 1:
        num_odd_sites = len(sites) // 2
        if num_odd_sites % 2 == 0:
            num_odd_sites += 1

        def site_charge(site):
            return 1 if site in sites[:num_odd_sites] else 0

    psi = sr.networks.TN_fermionic_from_edges_rand(
        "Z2",
        edges,
        bond_dim=4,
        phys_dim=2,
        seed=42,
        site_charge=site_charge,
        flat=flat,
    )

    # run a few steps of simple update
    su = qtn.SimpleUpdateGen(
        psi,
        ham,
        cutoff=0.0 if flat else 1e-15,
    )
    su.evolve(10, progbar=True)
    psi_su = su.state

    # compute energy via exact contraction of local tensor terms
    en_local_exact = psi_su.compute_local_expectation_exact(terms)

    if not flat:
        # compute energy via contracting the full wavefunction as a dense vector
        psi_su_dense = (
            psi_su.contract(all)
            .data.fuse(range(psi.nsites))
            .phase_sync()
            .blocks[(sector,)]
            .reshape(-1, 1)
        )
        psi_su_dense /= np.linalg.norm(psi_su_dense)
        energy_vector = qu.expec(psi_su_dense, H_sparse)
        assert energy_vector == pytest.approx(en_local_exact)

    # finally compute energy via slicing and vmc coupled configs

    def fn_amplitude(config):
        tnx = psi_su.isel(
            {psi_su.site_ind(coo): xi for coo, xi in config.items()}
        )
        return tnx.contract(all)

    energy_vmc = H.evaluate_exact_configs(fn_amplitude)
    assert energy_vmc == pytest.approx(en_local_exact)
    assert energy_vmc > ge


edges_square_2x2_obc = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 1), (1, 1)),
    ((1, 0), (1, 1)),
)
edges_complete_5 = (
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 3),
    (1, 4),
    (2, 3),
    (2, 4),
    (3, 4),
)


@pytest.mark.parametrize("edges", [edges_square_2x2_obc, edges_complete_5])
@pytest.mark.parametrize("sector", [0, 1])
@pytest.mark.parametrize("flat", [False, True])
def test_spinful_su_energy_matches_z2(edges, sector, flat):
    import numpy as np
    import quimb as qu
    import quimb.operator as qop
    import quimb.tensor as qtn

    import symmray as sr

    sites = sorted(set().union(*edges))

    # define random fermi-hubbard spinless model,
    # with edge, site and spin dependent coefficients for testing generality
    rng = np.random.default_rng(42)
    tijs = {edge: (rng.uniform(), rng.uniform()) for edge in edges}
    Uijs = {site: rng.uniform() for site in sites}
    mus = {site: (rng.uniform(), rng.uniform()) for site in sites}

    # build reference matrix hamiltonian
    H = qop.fermi_hubbard_from_edges(
        edges,
        t=tijs,
        U=Uijs,
        mu=mus,
        sector=sector,
        symmetry="Z2",
        # need to order sites by spatial coordinate first, then spin
        order=lambda coo: (coo[1], coo[0]),
    )
    H_sparse = H.build_sparse_matrix()
    ge = qu.groundenergy(H_sparse)

    # build local tensor hamiltonian
    terms = sr.ham_fermi_hubbard_from_edges(
        "Z2",
        edges,
        t=tijs,
        U=Uijs,
        mu=mus,
        flat=flat,
    )
    ham = qtn.LocalHamGen(terms)

    # build peps with different total charges
    if sector == 0:

        def site_charge(site):
            return 0

    elif sector == 1:
        num_odd_sites = len(sites) // 2
        if num_odd_sites % 2 == 0:
            num_odd_sites += 1

        def site_charge(site):
            return 1 if site in sites[:num_odd_sites] else 0

    psi = sr.networks.TN_fermionic_from_edges_rand(
        "Z2",
        edges,
        bond_dim=4,
        phys_dim=4,
        seed=42,
        site_charge=site_charge,
        flat=flat,
    )

    su = qtn.SimpleUpdateGen(
        psi,
        ham,
        cutoff=0.0 if flat else 1e-15,
    )
    su.evolve(10, progbar=True)
    psi_su = su.state

    # compute energy via exact contraction of local tensor terms
    en_local_exact = psi_su.compute_local_expectation_exact(terms)

    # NOTE: can't compute 'dense wavefunction' without reordering local
    # basis to account for spin up/down, so skip that test here

    def fn_amplitude(config):
        # map (spin-up, spin-down) to 0...3 indexing
        # into symmetry grouped: [ 00, 11 ], [ 10, 01 ]
        selector = {
            psi_su.site_ind(coo):
            # first bit set by parity
            2 * (config[("↑", coo)] != config[("↓", coo)])
            # second bit by spin down
            + config[("↓", coo)]
            for coo in sites
        }
        tnx = psi_su.isel(selector)
        return tnx.contract(all)

    energy_vmc = H.evaluate_exact_configs(fn_amplitude)
    assert energy_vmc == pytest.approx(en_local_exact)
    assert energy_vmc > ge
