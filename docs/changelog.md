# Changelog

Release notes for `symmray`. See also the [GitHub releases page](https://github.com/jcmgray/symmray/releases).

## v0.4.0 (unreleased)

**Breaking Changes:**

- `conj_project` now takes `axes=` instead of `axis=`. It accepts either a single integer or an ordered sequence of axes to keep uncontracted.

**Enhancements:**

- Fermionic arrays expose `dummy_parity`, the combined parity of their dummy modes, while preserving traced backend scalar types for flat arrays.

## v0.3.1 (2026-08-25)

**Enhancements:**

- Added scalar 2D and 3D tensor-network builders: {func}`~symmray.networks.TN2D_abelian_rand`, {func}`~symmray.networks.TN2D_fermionic_rand`, {func}`~symmray.networks.TN3D_abelian_rand`, and {func}`~symmray.networks.TN3D_fermionic_rand`.
- Network builders accept `duals` to choose, randomize, or override bond orientations.
- {meth}`~symmray.bosonic_common.BosonicCommon.to_dense` accepts explicit index maps to restore the dense basis ordering supplied to {func}`~symmray.utils.from_dense`.

**Full Changelog**: [v0.3.0...v0.3.1](https://github.com/jcmgray/symmray/compare/v0.3.0...v0.3.1)

## v0.3.0 (2026-08-22)

**Breaking Changes:**

- The minimum `autoray` version is now 0.8.10.
- {func}`~symmray.hamiltonians.ham_heisenberg_from_edges` now takes explicit `j` and `b` arguments instead of forwarding arbitrary keyword arguments to `quimb.ham_heis`. Couplings can vary by edge and fields can vary by site.
- {func}`~symmray.spin_local_operators.tfim_local_array` moved from `symmray.hamiltonians` to `symmray.spin_local_operators` and is now exported from the top-level namespace.

**Enhancements:**

- Added `conj_project(axis=...)` for inserting an array and its conjugate as a projector with arbitrary rank, bond duality, and fermionic parity.
- Added {func}`~symmray.interface.finfo` for querying the machine limits of common NumPy floating-point and complex dtypes.
- Symmetric arrays support scalar exponentiation with `x ** p`.
- Added {meth}`~symmray.common.SymmrayCommon.to` for changing the array backend, dtype, and device, including in place. This method requires `autoray>=0.9.0`.
- Added quimb-free local spin builders: {func}`~symmray.spin_local_operators.build_local_spin_dense`, {func}`~symmray.spin_local_operators.build_local_spin_array`, {func}`~symmray.spin_local_operators.tfim_local_array`, {func}`~symmray.spin_local_operators.heisenberg_local_array`, and {func}`~symmray.spin_local_operators.spin_operator_local_array`. Passing `symmetry=None` returns dense arrays.
- {func}`~symmray.hamiltonians.ham_tfim_from_edges` and {func}`~symmray.hamiltonians.ham_heisenberg_from_edges` no longer require quimb. The Heisenberg builder supports anisotropic, per-edge couplings and per-site longitudinal fields.
- Added the local fermionic operators {func}`~symmray.fermionic_local_operators.fermi_spin_z_local_array`, {func}`~symmray.fermionic_local_operators.fermi_spin_plus_local_array`, {func}`~symmray.fermionic_local_operators.fermi_spin_minus_local_array`, {func}`~symmray.fermionic_local_operators.fermi_double_occupancy_local_array`, {func}`~symmray.fermionic_local_operators.fermi_pairing_onsite_local_array`, and {func}`~symmray.fermionic_local_operators.fermi_pairing_bond_local_array`.
- Two-input {func}`~symmray.interface.einsum` expressions dispatch pure pairwise contractions directly to {func}`~symmray.interface.tensordot` and support vector-array diagonal multiplication patterns.
- Flat fermionic dummy-mode phases are calculated with vectorized Koszul sorting. This reduces `jax.jit` and `torch.compile` graph size and supports compiled, vectorized amplitudes and gradients ({pull}`36`).
- Matmul-shaped flat contractions use the faster direct batched-matmul path.
- Flat contraction and fusion skip redundant charge columns when sorting, reducing sorting overhead.

**Bug Fixes:**

- Flat sector sorting no longer creates temporary backend arrays on the default device when packing integer keys.
- Empty {class}`~symmray.sparse.sparse_vector.BlockVector` objects now pass structural validation.
- Fermionic eigendecomposition and Cholesky accept `drop_dummy_modes` and again drop dummy modes and labels by default, avoiding unwanted phases when their factors are used as projectors.
- {func}`~symmray.linalg.eigh_truncated` clips small negative eigenvalues when `positive=True` before truncating the positive spectrum.
- Flat charge selection now constructs its selection index on the active backend and device, fixing JAX tracing and Torch execution ({pull}`42`).
- {func}`~symmray.utils.get_array_cls` raises a clear `ValueError` when flat storage is requested for an unsupported symmetry instead of exposing an internal lookup error.

**Docs:**

- Split the README into task-focused guides covering construction, storage, fermionic arrays, dummy modes, local operators, linear algebra, and tensor networks, with extensive API cross-references.
- Added a worked landing-page example and moved the existing notebooks under `docs/examples` so they are built from saved outputs without being executed.
- Added a developer guide, API reference, changelog resolver, references page, and zero-warning API docstring rendering.

**Infrastructure:**

- Added dedicated `testjax` and `testtorch` Pixi environments and CI jobs, including compiled vectorized fermionic amplitude and gradient coverage ({pull}`36`).
- Added a distribution build smoke test, refreshed package metadata and Pixi tasks, and updated Ruff formatting.
- Added contributing guidance and a code of conduct.
- Grouped monthly Dependabot updates and updated the Pixi, checkout, Codecov, setup-python, and PyPI publishing actions ({pull}`33`, {pull}`34`, {pull}`35`, {pull}`37`, and {pull}`43`).

**Full Changelog**: [v0.2.1...v0.3.0](https://github.com/jcmgray/symmray/compare/v0.2.1...v0.3.0)

## v0.2.1 (2026-04-28)

**Breaking Changes:**

- Decompositions now use the corrected convention for which side carries the charge. Results from the non-default side may differ. This affects {func}`~symmray.linalg.svd`, {func}`~symmray.linalg.qr`, {func}`~symmray.linalg.lq`, and {func}`~symmray.linalg.eigh`.
- Internal abstract hooks are now concrete methods on the appropriate common mixin. This affects external subclasses of internal mixins.
- {class}`~symmray.linalg_common.Absorb` now provides consistent absorb-mode handling across sparse and flat linear algebra.

**Enhancements:**

- {func}`~symmray.interface.einsum` supports multiple input arrays using `cotengra` contraction paths.
- Added {func}`~symmray.linalg.cholesky` and {func}`~symmray.linalg.cholesky_regularized` for sparse and flat, bosonic and fermionic arrays. Flat support was contributed in ({pull}`25`).
- Added {func}`~symmray.linalg.svd_rand_truncated`, {func}`~symmray.linalg.qr_via_cholesky`, and {func}`~symmray.linalg.lq_via_cholesky`.
- Added {func}`~symmray.linalg.svd_via_eig_truncated` for flat arrays.
- {meth}`~symmray.array_common.ArrayCommon.svd_via_eig` accepts `eps` to stabilize small singular values.
- Flat {meth}`~symmray.array_common.ArrayCommon.svd_via_eig` supports every {class}`~symmray.linalg_common.Absorb` mode and shortcut.
- Linear algebra decompositions now share the internal split interface across array types.
- Faster lexicographic sorting and phase calculations reduce `jax.jit` tracing time. Integer packing also uses narrower dtypes where possible ({pull}`31`).
- Reshape argument parsing was simplified and given broader test coverage.
- Added the API reference and Read the Docs build ({pull}`30`).

**Bug Fixes:**

- Fixed single-input {func}`~symmray.interface.einsum` expressions.
- Fixed {func}`~symmray.interface.reshape` when `newshape` contains an ambiguous `-1`.
- Fixed charge-carrying side behavior in decompositions.
- Fixed dtype handling when shifting charges.
- Fixed {func}`~symmray.linalg.cholesky` errors found during multi-backend testing.

**Internal:**

- More operations now use the shared bosonic mixin. The test suite enforces that a public class has no duplicate method definitions in its MRO.

**Infrastructure:**

- Added a debug-enabled `pixi run pytest` task, renamed the Python test environments, and simplified CI to use the Pixi task.

**Full Changelog**: [v0.2.0...v0.2.1](https://github.com/jcmgray/symmray/compare/v0.2.0...v0.2.1)

## v0.2.0 (2026-01-13)

**Breaking Changes:**

- Renamed `oddpos` to `dummy_modes` and adjusted its behavior.
- Changed the spinful local-basis ordering, with compatibility tests against tensor-network energy calculations.

**Enhancements:**

- Added `to_pytree` and `from_pytree` methods, including {meth}`~symmray.bosonic_common.BosonicCommon.to_pytree` and {meth}`~symmray.sparse.sparse_abelian_array.AbelianArray.from_pytree`.
- Preserve labels where possible when contracting arrays.
- Added `flat` to the local fermionic operator constructors, including {func}`~symmray.fermionic_local_operators.build_local_fermionic_array`.
- {func}`~symmray.utils.from_dense` accepts additional array-construction options ({pull}`19`).
- {func}`~symmray.fermionic_local_operators.fermi_hubbard_spinless_local_array` supports a pairing term.

**Bug Fixes:**

- Fixed several Torch and JAX compatibility problems.
- Use backend-agnostic {func}`~symmray.interface.take` instead of direct tensor indexing for Torch compatibility ({pull}`23`).

**Full Changelog**: [v0.1.1...v0.2.0](https://github.com/jcmgray/symmray/compare/v0.1.1...v0.2.0)

## v0.1.1 (2025-12-03)

**Enhancements:**

- Support flat fermionic arrays under `torch.vmap`.
- Added {func}`~symmray.interface.take`.
- Improved transpose handling for singleton dimensions ({pull}`1`).
- Added 2D-input support to the abelian symmetry-line solver ({pull}`14`).
- Added an MPS construction helper ({pull}`17`).

**Bug Fixes:**

- Avoided in-place multiplication in truncated SVD ({pull}`2`).
- Fixed symmetry-line location ({pull}`9`).

**Infrastructure:**

- Updated the Micromamba, uv, checkout, and Pixi setup actions ({pull}`4`, {pull}`11`, {pull}`12`, {pull}`13`, {pull}`16`, and {pull}`18`).

**Full Changelog**: [v0.1.1](https://github.com/jcmgray/symmray/commits/v0.1.1)
