# Getting started

`symmray` arrays behave like dense arrays where the symmetry structure permits.
The main differences are that each axis carries charges and an orientation, and
only symmetry-allowed blocks are stored.

## Create and contract arrays

The testing utilities provide a convenient way to create example arrays:

```python
import symmray as sr

shared = sr.utils.rand_index("Z2", 8, dual=True, seed=1)
x = sr.utils.get_rand(
    "Z2",
    shape=(4, 6, shared),
    duals=(False, False, True),
    seed=2,
)
y = sr.utils.get_rand(
    "Z2",
    shape=(shared.conj(), 5),
    duals=(False, True),
    seed=3,
)

z = sr.tensordot(x, y, axes=((2,), (0,)))
```

The contracted indices must have compatible charges, sizes, and opposite
orientations. The result is another `symmray` array.

Use functions from the [`symmray` namespace](symmray) directly, through
[`autoray`](https://autoray.readthedocs.io/), or through the Python Array API:

```python
import autoray as ar

z0 = sr.tensordot(x, y, axes=((2,), (0,)))
z1 = ar.do("tensordot", x, y, axes=((2,), (0,)))

xp = x.__array_namespace__()
z2 = xp.tensordot(x, y, axes=((2,), (0,)))
```

## Shape and reshaping

[`x.shape`](#SparseArrayCommon.shape) is the shape of
[`x.to_dense()`](#BosonicCommon.to_dense). [`x.size`](#SparseArrayCommon.size)
follows the same convention.
These values describe the dense array, not the amount of stored data.

`symmray` reshapes arrays by
[`fusing`](#symmray.array_common.ArrayCommon.fuse) or
[`unfusing`](#symmray.bosonic_common.BosonicCommon.unfuse) complete axes:

```python
matrix = z.fuse((0, 1), (2,))
```

See [Linear algebra](linear_algebra.md) for decompositions of matrix-like
arrays.

For sparsely occupied sector structures, such as many `U1` arrays, a fused axis
can omit charge sectors that do not occur. Its effective size can therefore
differ from a size inferred only from dense shapes.

## Change the numerical backend

[`SymmrayCommon.to`](#SymmrayCommon.to) converts the stored blocks while
preserving their symmetry metadata:

```python
x_torch = x.to("torch-float32-cuda:0")
x_jax = x.to(backend="jax", dtype="complex128")
```

This conversion is separate from choosing `symmray`'s sparse or flat [block storage](block storage.md).
