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

    peps = sr.networks.PEPS_abelian_rand(
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

    charge = sum(x.charge for x in peps.arrays) % 2

    # contract full wavefunction
    t = peps.contract(all, output_inds=peps.site_inds)
    x = t.data
    if fermionic:
        x.phase_sync(inplace=True)

    # for every bit-string, check sliced-then-contract == contract-then-slice
    N = peps.nsites
    for r in range(2 ** (N - 1)):
        # for Z2, last bit fixed by charge conservation
        prefix = list(map(int, np.binary_repr(r, width=N - 1)))
        config = (*prefix, (charge + int(np.sum(prefix) % 2)) % 2)

        # slice wavefunction and contract
        psix = peps.isel(
            {peps.site_ind(coo): v for coo, v in zip(peps.sites, config)}
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
