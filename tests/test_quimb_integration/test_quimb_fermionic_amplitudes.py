import numpy as np
import pytest

import symmray as sr


@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("site_charge", [0, 1, "checkerboard"])
def test_2d_spinless(cyclic, flat, fermionic, site_charge):
    Lx = 3
    Ly = 3

    if site_charge == "checkerboard":

        def site_charge_fn(site):
            return sum(site) % 2

    else:

        def site_charge_fn(site):
            return site_charge

    psi = sr.networks.PEPS_abelian_rand(
        "Z2",
        Lx,
        Ly,
        bond_dim=2,
        phys_dim=2,
        seed=42,
        site_charge=site_charge_fn,
        cyclic=cyclic,
        flat=flat,
        fermionic=fermionic,
    )

    charge = sum(x.charge for x in psi.arrays) % 2

    # contract full wavefunction
    t = psi.contract(all, output_inds=psi.site_inds)
    x = t.data
    if fermionic:
        x.phase_sync(inplace=True)

    # for every bit-string, check sliced-then-contract == contract-then-slice
    N = psi.nsites
    for r in range(2 ** (N - 1)):
        # for Z2, last bit fixed by charge conservation
        prefix = list(map(int, np.binary_repr(r, width=N - 1)))
        config = (*prefix, (charge + int(np.sum(prefix) % 2)) % 2)

        # slice wavefunction and contract
        psix = psi.isel(
            {psi.site_ind(coo): v for coo, v in zip(psi.sites, config)}
        )
        amp = psix.contract(all, output_inds=())

        # retrieve relevant amplitude from full contraction
        if flat:
            x.sort_stack(inplace=True)
            famp = x.blocks[r].item()
        else:
            famp = x.blocks[config].item()

        # compare
        assert abs(1 - famp / amp) < 1e-10


edges_rrg_d3_n10_seed42 = [
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
]


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("site_charge", [0, 1, "checkerboard"])
def test_rrg_spinless(flat, fermionic, site_charge):
    if site_charge == "checkerboard":

        def site_charge_fn(site):
            return site % 2

    else:

        def site_charge_fn(site):
            return site_charge

    psi = sr.networks.TN_abelian_from_edges_rand(
        "Z2",
        edges_rrg_d3_n10_seed42,
        bond_dim=2,
        phys_dim=2,
        seed=42,
        site_charge=site_charge_fn,
        flat=flat,
        fermionic=fermionic,
    )

    charge = sum(x.charge for x in psi.arrays) % 2

    # contract full wavefunction
    t = psi.contract(all, output_inds=psi.site_inds)
    x = t.data
    if fermionic:
        x.phase_sync(inplace=True)

    # for every bit-string, check sliced-then-contract == contract-then-slice
    N = psi.nsites
    for r in range(2 ** (N - 1)):
        # for Z2, last bit fixed by charge conservation
        prefix = list(map(int, np.binary_repr(r, width=N - 1)))
        config = (*prefix, (charge + int(np.sum(prefix) % 2)) % 2)

        # slice wavefunction and contract
        psix = psi.isel(
            {psi.site_ind(coo): v for coo, v in zip(psi.sites, config)}
        )
        amp = psix.contract(all, output_inds=())

        # retrieve relevant amplitude from full contraction
        if flat:
            x.sort_stack(inplace=True)
            famp = x.blocks[r].item()
        else:
            famp = x.blocks[config].item()

        # compare
        assert abs(1 - famp / amp) < 1e-10
