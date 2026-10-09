# Block layout

`symmray` has three block layouts: sparse, flat and packed. They represent the
same arrays and support the same high-level operations, but store and process
the blocks differently. Choose one with the `layout` keyword of constructors
such as [`get_rand`](#symmray.utils.get_rand), the network builders, local
operators and Hamiltonians, or convert an existing array.

## Sparse layout

[`AbelianArray`](#AbelianArray) and [`FermionicArray`](#FermionicArray) store a
dictionary from sector tuples to blocks. Different sectors may have different
block shapes. Blocks can come from NumPy, PyTorch, JAX, or another `autoray`
backend.

Use the sparse layout when:

- charge sectors have unequal sizes
- the set of stored sectors changes during a computation
- flexibility matters more than static array structure

The sparse layout supports the full set of built-in symmetries, and is the
default.

## Flat layout

[`AbelianArrayFlat`](#AbelianArrayFlat) and
[`FermionicArrayFlat`](#FermionicArrayFlat) stack all blocks into one array
with shape `(num_blocks, *shape_block)`. Sector keys form a second array with
shape `(num_blocks, ndim)`.

All stored blocks must therefore have the same
[`shape_block`](#symmray.flat.flat_data_common.FlatCommon.shape_block).
Equivalently, each charge sector on an axis must have the *same size*. The flat
layout also requires all possible charge sectors to be present, which in
practice means using *cyclic symmetries*. Its fixed-shape stacked arrays are
designed for vectorized GPU execution and tracing or compilation, especially
with [`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) and
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
Block shapes and sector structure must remain static across compiled calls.

For fermionic arrays, parity and phases are also handled in a branchless
manner, allowing expressions involving dynamic parities, such as wavefunction
amplitude contractions, to still be compiled and vectorized.

[`Z2ArrayFlat`](#symmray.flat.flat_abelian_array.Z2ArrayFlat) and
[`Z2FermionicArrayFlat`](#symmray.flat.flat_fermionic_array.Z2FermionicArrayFlat)
are the concrete flat classes. Other finite cyclic symmetries, such as `Z3`,
use [`AbelianArrayFlat`](#AbelianArrayFlat) or
[`FermionicArrayFlat`](#FermionicArrayFlat) with an explicit symmetry. `U1` is
not supported because the flat layout requires a finite cyclic symmetry.
Product symmetries such as `Z2Z2` and `U1U1` are also unsupported.

Create a compatible random flat array with equal sector sizes:

```python
import symmray as sr

x = sr.utils.get_rand(
    "Z2",
    shape=(8, 8, 8),
    subsizes="equal",
    layout="flat",
    seed=1,
)
```

The older `flat=True` keyword is the same as `layout="flat"`.

## Packed layout

[`AbelianArrayPacked`](#AbelianArrayPacked) and
[`FermionicArrayPacked`](#FermionicArrayPacked) store every block C contiguous,
end to end, in one 1D NumPy buffer. An integer array with shape
`(num_blocks, ndim)` gives the position of each block's charge on each axis.
Like the sparse layout, sectors can have unequal sizes and any built-in
symmetry is supported.

Most operations have two parts. A plan is built only from the block structure,
and is cached. A run then moves or multiplies the data, using compiled kernels
that loop over all blocks at once. Lazy fermionic phases are applied during
the next data copy, rather than in separate passes.

Use the packed layout when:

- there are many small blocks, e.g. `U1` and especially `U1U1` tensor
  networks, where the sparse layout spends most of its time on per block
  Python overhead
- the data is NumPy on the CPU

The compiled kernels use [`numba`](https://numba.pydata.org/) if it is
installed, and otherwise fall back to slower pure NumPy kernels.
[`set_packed_kernels`](#symmray.packed.kernels.set_packed_kernels) or the
`SYMMRAY_PACKED_KERNELS` environment variable selects them explicitly. The
first use of the numba kernels compiles them, which is cached for later runs.
Some less common operations, such as `einsum` and `trace`, have no packed
version yet. They convert to the sparse layout and back.

The concrete packed classes are
[`Z2ArrayPacked`](#symmray.packed.packed_abelian_array.Z2ArrayPacked),
[`U1ArrayPacked`](#symmray.packed.packed_abelian_array.U1ArrayPacked),
[`Z2Z2ArrayPacked`](#symmray.packed.packed_abelian_array.Z2Z2ArrayPacked),
[`U1U1ArrayPacked`](#symmray.packed.packed_abelian_array.U1U1ArrayPacked) and
their fermionic versions, such as
[`U1FermionicArrayPacked`](#symmray.packed.packed_fermionic_array.U1FermionicArrayPacked).
Singular values and eigenvalues are returned as a
[`PackedVector`](#PackedVector).

```python
x = sr.utils.get_rand(
    "U1U1",
    shape=(8, 8, 8),
    fermionic=True,
    layout="packed",
    seed=1,
)
```

## Conversion

Convert a sparse array with
[`to_flat()`](#symmray.sparse.sparse_array_common.SparseArrayCommon.to_flat) or
[`to_packed()`](#symmray.sparse.sparse_array_common.SparseArrayCommon.to_packed),
and return to the sparse layout with `to_blocksparse()`:

```python
xs = sr.utils.get_rand(
    "Z2",
    shape=(8, 8, 8),
    subsizes="equal",
    seed=2,
)
xf = xs.to_flat()
xp = xs.to_packed()
xs_again = xf.to_blocksparse()
```

Conversion changes the block layout, not the dense array represented by the
object. Sparse-to-flat conversion first removes unused charges and then stacks
the remaining blocks, so it requires compatible block shapes. Sparse-to-packed
conversion works for any array with NumPy blocks.

Use [`SymmrayCommon.to`](#SymmrayCommon.to) instead when changing the numerical
backend, dtype, or device of the stored blocks.
