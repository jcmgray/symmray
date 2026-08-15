# Dummy fermionic modes

Dummy modes record the global ordering of odd fermionic content that is not
represented by a real array axis. They let `symmray` compute global signs using
local array operations.


```{hint}
You can always treat each dummy mode as a dimension-1, parity-odd axis
prepended to the real axes. These virtual axes make the array even overall.
They are bookkeeping only: no real axis or block dimension is added, but
a phased sort of them on the final array produces the required global sign.
```

## Why they are needed

An even-parity fermionic array composes without extra ordering information. An
odd-parity array does not: moving it past another odd object changes the global
sign.

Give each odd array a sortable [`label`](#SparseArrayCommon.label). `symmray`
then creates a matching [`FermionicOperator`](#FermionicOperator) in
[`array.dummy_modes`](#FermionicCommon.dummy_modes):

```python
import symmray as sr

x = sr.utils.get_rand(
    "Z2",
    shape=(2, 4),
    charge=1,
    fermionic=True,
    label="B",
    seed=1,
)

x.dummy_modes
# (B-,)
```

The dummy mode is metadata. It does not add an array axis or change the block
shapes.

## Contraction

When two arrays contract with [`tensordot`](#symmray.interface.tensordot),
`symmray` concatenates their dummy modes, moves them into canonical order, and
records the sign of each odd swap. Adjacent modes with the same label and
opposite orientations cancel.

```python
a = sr.utils.rand_index("Z2", 2, seed=1)
b = sr.utils.rand_index("Z2", 4, seed=2)
c = sr.utils.rand_index("Z2", 6, seed=3)
d = sr.utils.rand_index("Z2", 8, seed=4)

x = sr.utils.get_rand(
    "Z2",
    shape=(a, b, c),
    charge=1,
    fermionic=True,
    label="B",
    seed=5,
)
y = sr.utils.get_rand(
    "Z2",
    shape=(b.conj(), c.conj(), d),
    charge=1,
    fermionic=True,
    label="A",
    seed=6,
)

z = sr.tensordot(x, y, axes=((1, 2), (0, 1)))
z.dummy_modes
# (A-, B-)
```

[`Conjugation`](#symmray.sparse.sparse_fermionic_array.FermionicArray.conj)
reverses and daggers the modes:

```python
z.conj().dummy_modes
# (B+, A+)
```

Contracting an array with its conjugate cancels the matching pairs.

## Slicing

Removing an axis with odd charge moves that charge into a dummy mode. The array
must have a label so the new mode has a stable identity.

```python
x = sr.utils.get_rand(
    "Z2",
    shape=(2, 4, 6),
    charge=0,
    fermionic=True,
    label="X",
    seed=1,
)

x[:, 3, :].dummy_modes
# (('squeeze', 'X', 1)+,)
```

## Explicit modes

Pass [`dummy_modes`](#FermionicCommon.dummy_modes) to a
[`FermionicArray`](#FermionicArray) constructor to restore existing modes or
override default creation. The value must be a tuple of
[`FermionicOperator`](#FermionicOperator) objects whose combined parity makes
the effective array even.

Practical rules:

- label every odd-parity array that may join a larger computation
- preserve [`dummy_modes`](#FermionicCommon.dummy_modes) when reconstructing an
  existing array
- inspect [`dummy_modes`](#FermionicCommon.dummy_modes) first when debugging a
  global fermionic sign
- do not reuse the same label for independent odd operators
