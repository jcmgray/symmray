# Storage backends

`symmray` has two storage layouts. Both can hold numerical blocks from NumPy,
PyTorch, JAX, or another Autoray backend.

## Sparse storage

[`AbelianArray`](#AbelianArray) and [`FermionicArray`](#FermionicArray) store a
dictionary from sector tuples to blocks. Different sectors may have different
block shapes.

Use sparse storage when:

- charge sectors have unequal sizes
- the set of stored sectors changes during a computation
- flexibility matters more than static array structure

Sparse storage supports the full set of built-in symmetries.

## Flat storage

[`AbelianArrayFlat`](#AbelianArrayFlat) and
[`FermionicArrayFlat`](#FermionicArrayFlat) stack all blocks into one array with
shape `(num_blocks, *shape_block)`. Sector keys form a second array with shape
`(num_blocks, ndim)`.

All stored blocks must therefore have the same
[`shape_block`](#symmray.flat.flat_data_common.FlatCommon.shape_block). In
practice, each charge sector on an axis must have the same size. Flat storage
is intended for cyclic symmetries. Its fixed-shape stacked arrays are designed
for vectorized GPU execution and tracing or compilation, especially with
[`jax.jit`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html) and
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
Block shapes and sector structure must remain static across compiled calls.

[`Z2ArrayFlat`](#symmray.flat.flat_abelian_array.Z2ArrayFlat) and
[`Z2FermionicArrayFlat`](#symmray.flat.flat_fermionic_array.Z2FermionicArrayFlat)
are the concrete flat classes. Other finite cyclic symmetries, such as `Z3`,
use [`AbelianArrayFlat`](#AbelianArrayFlat) or
[`FermionicArrayFlat`](#FermionicArrayFlat) with an explicit symmetry. `U1` is
not supported because flat storage requires a finite cyclic symmetry. Product
symmetries such as `Z2Z2` and `U1U1` are also unsupported.

Create a compatible random flat array with equal sector sizes:

```python
import symmray as sr

x = sr.utils.get_rand(
    "Z2",
    shape=(8, 8, 8),
    subsizes="equal",
    flat=True,
    seed=1,
)
```

## Conversion

Convert a compatible sparse array with
[`to_flat()`](#symmray.sparse.sparse_array_common.SparseArrayCommon.to_flat)
and return to sparse storage with
[`to_blocksparse()`](#symmray.flat.flat_abelian_array.AbelianArrayFlat.to_blocksparse):

```python
xs = sr.utils.get_rand(
    "Z2",
    shape=(8, 8, 8),
    subsizes="equal",
    seed=2,
)
xf = xs.to_flat()
xs_again = xf.to_blocksparse()
```

Conversion changes the storage layout, not the dense array represented by the
object. Sparse-to-flat conversion first removes unused charges and then stacks
the remaining blocks, so it requires compatible block shapes.

Use [`SymmrayCommon.to`](#SymmrayCommon.to) instead when changing the numerical
backend, dtype, or device of the stored blocks.
