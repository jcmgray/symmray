# Tensor networks and Hamiltonians

`symmray` includes constructors for
[`quimb.tensor`](https://quimb.readthedocs.io/en/latest/tensor.html) networks
whose tensors contain `symmray` arrays. Quimb is an optional dependency.

## Network constructors

The general graph constructors are:

- [`TN_abelian_from_edges_rand`](#TN_abelian_from_edges_rand)
- [`TN_fermionic_from_edges_rand`](#TN_fermionic_from_edges_rand)

Convenience wrappers create common geometries:

- [`MPS_abelian_rand`](#symmray.networks.MPS_abelian_rand) and
  [`MPS_fermionic_rand`](#symmray.networks.MPS_fermionic_rand)
- [`PEPS_abelian_rand`](#symmray.networks.PEPS_abelian_rand) and
  [`PEPS_fermionic_rand`](#symmray.networks.PEPS_fermionic_rand)
- [`PEPS3D_abelian_rand`](#symmray.networks.PEPS3D_abelian_rand) and
  [`PEPS3D_fermionic_rand`](#symmray.networks.PEPS3D_fermionic_rand)

The constructors assign consistent bond orientations and choose charge sectors
from the requested symmetry, dimensions, and total charges. Pass `flat=True`
with compatible equal sector sizes to use flat storage.

## Hamiltonian terms

Graph Hamiltonian helpers return dictionaries of local terms keyed by edges:

- [`ham_tfim_from_edges`](#ham_tfim_from_edges)
- [`ham_heisenberg_from_edges`](#ham_heisenberg_from_edges)
- [`ham_fermi_hubbard_from_edges`](#ham_fermi_hubbard_from_edges)
- [`ham_fermi_hubbard_spinless_from_edges`](#ham_fermi_hubbard_spinless_from_edges)

[`parse_edges_to_site_info`](#parse_edges_to_site_info) computes canonical bond
orientations, site dimensions, tags, and coordination numbers. The Hamiltonian
builders use those coordination numbers to distribute one-site terms across
edges without overcounting.

## Examples

The repository contains notebooks for:

- sparse and flat PEPS construction with Quimb
- spinful and spinless fermionic amplitudes
- batched GPU amplitudes with JAX and PyTorch

See the [rendered example notebooks](index_examples.md) or their
[`docs/examples` sources](https://github.com/jcmgray/symmray/tree/main/docs/examples).
