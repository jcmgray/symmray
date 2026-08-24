import pytest

import symmray as sr


def test_tn2d_abelian_rand_scalar():
    qtn = pytest.importorskip("quimb.tensor")

    tn = sr.TN2D_abelian_rand("Z2", 2, 3, bond_dim=2, seed=42)

    assert isinstance(tn, qtn.TensorNetwork2D)
    assert (tn.Lx, tn.Ly) == (2, 3)
    assert all(not ind.startswith("k") for ind in tn.ind_map)
    assert all(tn[f"X{i}"] for i in range(2))
    assert all(tn[f"Y{j}"] for j in range(3))
    tn.check()


def test_tn2d_fermionic_rand_scalar():
    qtn = pytest.importorskip("quimb.tensor")

    tn = sr.TN2D_fermionic_rand("Z2", 2, 2, bond_dim=2, seed=42)

    assert isinstance(tn, qtn.TensorNetwork2D)
    assert all(array.fermionic for array in tn.arrays)
    tn.check()


@pytest.mark.parametrize(
    "builder", [sr.TN2D_abelian_rand, sr.PEPS_abelian_rand]
)
def test_2d_abelian_rand_with_physical_indices(builder):
    qtn = pytest.importorskip("quimb.tensor")

    tn = builder("Z2", 2, 2, bond_dim=2, phys_dim=2, seed=42)

    assert isinstance(tn, qtn.PEPS)
    assert len(tn.site_inds) == 4
    tn.check()


@pytest.mark.parametrize(
    "builder", [sr.PEPS_abelian_rand, sr.PEPS_fermionic_rand]
)
def test_peps_rand_without_physical_indices(builder):
    qtn = pytest.importorskip("quimb.tensor")

    tn = builder("Z2", 2, 2, bond_dim=2, phys_dim=None, seed=42)

    assert isinstance(tn, qtn.TensorNetwork2D)
    tn.check()
