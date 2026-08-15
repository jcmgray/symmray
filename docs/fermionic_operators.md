# Local operators

Local operators are typically what tensor-network calculations need to evaluate
expectation values and apply gates. `symmray` provides fermionic operators as
[`FermionicArray`](#FermionicArray) objects and spin operators as
[`AbelianArray`](#AbelianArray) objects.

## Fermionic operators

Fermionic operators are built in a fixed local basis. The
[`Z2`](#symmray.symmetries.Z2) operators can use flat storage by setting
`flat=True`.

### Built-in fermionic operators

| Function | Operator | Supported symmetries |
|---|---|---|
| [`fermi_hubbard_spinless_local_array`](#symmray.fermionic_local_operators.fermi_hubbard_spinless_local_array) | spinless two-site Hubbard, or t-V, term | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1) |
| [`fermi_hubbard_local_array`](#symmray.fermionic_local_operators.fermi_hubbard_local_array) | spinful two-site Hubbard term | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1), [`Z2Z2`](#symmray.symmetries.Z2Z2), [`U1U1`](#symmray.symmetries.U1U1) |
| [`fermi_number_operator_spinless_local_array`](#symmray.fermionic_local_operators.fermi_number_operator_spinless_local_array) | $n$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1) |
| [`fermi_number_operator_spinful_local_array`](#symmray.fermionic_local_operators.fermi_number_operator_spinful_local_array) | $n_\uparrow + n_\downarrow$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1), [`Z2Z2`](#symmray.symmetries.Z2Z2), [`U1U1`](#symmray.symmetries.U1U1) |
| [`fermi_number_up_local_array`](#symmray.fermionic_local_operators.fermi_number_up_local_array) | $n_\uparrow$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1), [`Z2Z2`](#symmray.symmetries.Z2Z2), [`U1U1`](#symmray.symmetries.U1U1) |
| [`fermi_number_down_local_array`](#symmray.fermionic_local_operators.fermi_number_down_local_array) | $n_\downarrow$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1), [`Z2Z2`](#symmray.symmetries.Z2Z2), [`U1U1`](#symmray.symmetries.U1U1) |
| [`fermi_double_occupancy_local_array`](#symmray.fermionic_local_operators.fermi_double_occupancy_local_array) | $n_\uparrow n_\downarrow$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1), [`Z2Z2`](#symmray.symmetries.Z2Z2), [`U1U1`](#symmray.symmetries.U1U1) |
| [`fermi_spin_z_local_array`](#symmray.fermionic_local_operators.fermi_spin_z_local_array) | $(n_\uparrow-n_\downarrow)/2$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1), [`Z2Z2`](#symmray.symmetries.Z2Z2), [`U1U1`](#symmray.symmetries.U1U1) |
| [`fermi_spin_plus_local_array`](#symmray.fermionic_local_operators.fermi_spin_plus_local_array) | $c_\uparrow^\dagger c_\downarrow$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1) |
| [`fermi_spin_minus_local_array`](#symmray.fermionic_local_operators.fermi_spin_minus_local_array) | $c_\downarrow^\dagger c_\uparrow$ | [`Z2`](#symmray.symmetries.Z2), [`U1`](#symmray.symmetries.U1) |
| [`fermi_pairing_onsite_local_array`](#symmray.fermionic_local_operators.fermi_pairing_onsite_local_array) | $c_\uparrow c_\downarrow$ | [`Z2`](#symmray.symmetries.Z2) |
| [`fermi_pairing_bond_local_array`](#symmray.fermionic_local_operators.fermi_pairing_bond_local_array) | $c_{a\uparrow}c_{b\downarrow}-c_{a\downarrow}c_{b\uparrow}$ | [`Z2`](#symmray.symmetries.Z2) |

[`fermi_spin_operator_local_array`](#symmray.fermionic_local_operators.fermi_spin_operator_local_array)
is an alias of
[`fermi_spin_z_local_array`](#symmray.fermionic_local_operators.fermi_spin_z_local_array).
Use [`dagger()`](#symmray.sparse.sparse_fermionic_array.FermionicArray.dagger)
on a pair-annihilation operator to obtain the corresponding pair-creation
operator.

### Fermionic local bases

The spinless basis is

$$
\{|0\rangle, c^\dagger|0\rangle\}.
$$

The spinful basis is ordered as

$$
\{|00\rangle,
  c_\downarrow^\dagger|00\rangle,
  c_\uparrow^\dagger|00\rangle,
  c_\uparrow^\dagger c_\downarrow^\dagger|00\rangle\}.
$$

For example:

```python
import symmray as sr

h = sr.fermi_hubbard_local_array(
    "U1U1",
    t=1.0,
    U=8.0,
    mu=5.0,
)
n_up = sr.fermi_number_up_local_array("U1U1")
```

For an imaginary-time Trotter step,
[`reshape`](#symmray.array_common.ArrayCommon.reshape) the two-site Hamiltonian
to a matrix, apply [`expm`](#symmray.scipy.linalg.expm), and restore its tensor
shape:

```python
tau = 0.05
h_matrix = h.reshape((16, 16))
gate = sr.scipy.linalg.expm(-tau * h_matrix).reshape(h.shape)
```

For real-time evolution, replace `-tau` with `-1j * dt`.

Two-site Hamiltonian constructors accept `coordinations=(za, zb)`. They divide
one-site terms by the corresponding coordination number so a sum over edges
does not overcount them.

### Custom fermionic operators

[`build_local_fermionic_elements`](#build_local_fermionic_elements) computes
the nonzero matrix elements from an operator string and a basis.
[`build_local_fermionic_array`](#build_local_fermionic_array) also needs a
charge map for each local basis:

```python
a, b = map(sr.FermionicOperator, "ab")

terms = (
    (8, (a.dag, a, b.dag, b)),
    (-2, (a.dag, a)),
    (-2, (b.dag, b)),
)
bases = (
    ((), (a.dag,)),
    ((), (b.dag,)),
)

operator = sr.build_local_fermionic_array(
    terms,
    bases,
    symmetry="U1",
    index_maps=((0, 1), (0, 1)),
)
```

Operator order is significant. Build terms in the same site and spin ordering
as the chosen bases.

## Bosonic and spin operators

The built-in spin operators contain no fermionic signs and return bosonic
[`AbelianArray`](#AbelianArray) objects:

- [`tfim_local_array`](#symmray.spin_local_operators.tfim_local_array) builds a
  two-site transverse-field Ising term
- [`heisenberg_local_array`](#symmray.spin_local_operators.heisenberg_local_array)
  builds a two-site Heisenberg term
- [`spin_operator_local_array`](#symmray.spin_local_operators.spin_operator_local_array)
  builds a single-site spin operator

For example:

```python
h = sr.heisenberg_local_array("U1", j=1.0)
sz = sr.spin_operator_local_array("U1", op="sz")
```

Use [`build_local_spin_array`](#symmray.spin_local_operators.build_local_spin_array)
to build a symmetric spin operator from symbolic terms, or
[`build_local_spin_dense`](#symmray.spin_local_operators.build_local_spin_dense)
to build its dense form.
