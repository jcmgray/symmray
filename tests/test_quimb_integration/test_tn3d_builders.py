import pytest

import symmray as sr


def test_tn3d_abelian_rand_scalar():
    qtn = pytest.importorskip("quimb.tensor")

    tn = sr.TN3D_abelian_rand("Z2", 2, 2, 2, bond_dim=2, seed=42)

    assert isinstance(tn, qtn.TensorNetwork3D)
    assert (tn.Lx, tn.Ly, tn.Lz) == (2, 2, 2)
    assert all(not ind.startswith("k") for ind in tn.ind_map)
    assert all(tn[f"X{i}"] for i in range(2))
    assert all(tn[f"Y{j}"] for j in range(2))
    assert all(tn[f"Z{k}"] for k in range(2))
    tn.check()


def test_tn3d_fermionic_rand_scalar():
    qtn = pytest.importorskip("quimb.tensor")

    tn = sr.TN3D_fermionic_rand("Z2", 2, 2, 2, bond_dim=2, seed=42)

    assert isinstance(tn, qtn.TensorNetwork3D)
    assert all(array.fermionic for array in tn.arrays)
    tn.check()


@pytest.mark.parametrize(
    "builder", [sr.TN3D_abelian_rand, sr.PEPS3D_abelian_rand]
)
def test_3d_abelian_rand_with_physical_indices(builder):
    qtn = pytest.importorskip("quimb.tensor")

    tn = builder("Z2", 2, 2, 2, bond_dim=2, phys_dim=2, seed=42)

    assert isinstance(tn, qtn.PEPS3D)
    assert len(tn.site_inds) == 8
    tn.check()


@pytest.mark.parametrize(
    "builder", [sr.PEPS3D_abelian_rand, sr.PEPS3D_fermionic_rand]
)
def test_peps3d_rand_without_physical_indices(builder):
    qtn = pytest.importorskip("quimb.tensor")

    tn = builder("Z2", 2, 2, 2, bond_dim=2, phys_dim=None, seed=42)

    assert isinstance(tn, qtn.TensorNetwork3D)
    tn.check()
