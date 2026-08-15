# Linear algebra

`symmray` applies linear algebra block by block while preserving charge and
fermionic metadata.

The decomposition routines currently use
[`quimb`](https://quimb.readthedocs.io/) and require it to be installed.

## Standard decompositions

The [`symmray.linalg`](symmray.linalg) namespace provides:

- [`norm`](#symmray.linalg.norm)
- [`solve`](#symmray.linalg.solve)
- [`svd`](#symmray.linalg.svd)
- [`qr`](#symmray.linalg.qr) and [`lq`](#symmray.linalg.lq)
- [`eigh`](#symmray.linalg.eigh)
- [`cholesky`](#symmray.linalg.cholesky)

The matrix exponential is
[`symmray.scipy.linalg.expm`](#symmray.scipy.linalg.expm). It is also registered
as `linalg.expm` for dispatch through
[`autoray`](https://autoray.readthedocs.io/).

For a matrix-like array:

```python
import symmray as sr

x = sr.utils.get_rand(
    "Z2",
    shape=(8, 10),
    duals=(False, True),
    seed=1,
)
u, s, vh = sr.linalg.svd(x)
```

SVD and eigendecomposition return their spectrum as a
[`BlockVector`](#BlockVector) or [`FlatVector`](#FlatVector). These objects map
charges to one-dimensional blocks and can be applied with
[`multiply_diagonal`](#symmray.interface.multiply_diagonal).

## Tensor-network variants

The following variants trade accuracy, stability, or setup cost for properties
useful in tensor-network calculations:

- [`svd_truncated`](#symmray.linalg.svd_truncated) and
  [`eigh_truncated`](#symmray.linalg.eigh_truncated) truncate by `max_bond`,
  `cutoff`, and `cutoff_mode`
- [`svd_rand_truncated`](#symmray.linalg.svd_rand_truncated) uses a randomized
  low-rank approximation
- [`svd_via_eig_truncated`](#symmray.linalg.svd_via_eig_truncated) computes an
  SVD through a Hermitian eigendecomposition
- [`qr_stabilized`](#symmray.linalg.qr_stabilized) and
  [`lq_stabilized`](#symmray.linalg.lq_stabilized) fix diagonal signs for
  smoother gradients
- [`qr_via_cholesky`](#symmray.linalg.qr_via_cholesky) and
  [`lq_via_cholesky`](#symmray.linalg.lq_via_cholesky) use Gram matrices and
  Cholesky factorization

Decompositions with singular or eigenvalues accept `absorb` to control where
the spectrum is placed. For example, `absorb="left"` returns an isometric right
factor.

The sparse and flat layouts share this interface. Their numerical behavior can
differ because flat algorithms operate on stacked blocks.

For fermionic arrays, [`eigh_truncated`](#symmray.linalg.eigh_truncated) and
Cholesky factorizations drop `dummy_modes` and the array label from returned
factors by default. These factors are normally used as projectors, *added* to
the network with the tensors they were decomposed from still present, as such
the dummy modes should not be duplicated. Pass `drop_dummy_modes=False` to
change this behavior.
