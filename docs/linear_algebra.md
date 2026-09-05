# Linear algebra

`symmray` applies linear algebra block by block while preserving charge and
fermionic metadata.

The decomposition routines currently use the
[`array_split`](https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/decomp/index.html#quimb.tensor.decomp.array_split)
function from [`quimb`](https://quimb.readthedocs.io/) and require it to be
installed.


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

- [`svd_truncated`](#symmray.linalg.svd_truncated) truncate by `max_bond`,
  `cutoff`, and `cutoff_mode`
- [`eigh_truncated`](#symmray.linalg.eigh_truncated) the same for an assumed
  Hermitian matrix
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
the spectrum is 'placed'. For example, `absorb="left"` returns an isometric
right factor.

The integer value or any of its listed string aliases can be supplied:

| `absorb` value | String aliases | Returned factors |
| --- | --- | --- |
| `None` | `"U,s,VH"` | `U`, `s`, and `VH` separately |
| `2` | `"s"` | `s` only |
| `-12` | `"lsqrt"` | `U * sqrt(s)` only |
| `-11` | `"VH"`, `"rorthog"` | `VH` only |
| `-10` | `"Us"`, `"lfactor"` | `U * s` only |
| `-1` | `"Us,VH"`, `"left"` | `U * s` and `VH` |
| `0` | `"Usq,sqVH"`, `"both"` | `U * sqrt(s)` and `sqrt(s) * VH` |
| `1` | `"U,sVH"`, `"right"` | `U` and `s * VH` |
| `10` | `"U"`, `"lorthog"` | `U` only |
| `11` | `"sVH"`, `"rfactor"` | `s * VH` only |
| `12` | `"sqVH"`, `"rsqrt"` | `sqrt(s) * VH` only |

In many cases, specifying the minimally required `absorb` allows various faster
shortcuts.


## Sector truncation

Sparse truncated decompositions accept `max_bond_mode`. With `"global"`, all
block spectra are computed before the largest values are selected globally.
This is the default for exact SVD and eigendecomposition. A degenerate
multiplet crossing the threshold is kept whole, so the resulting bond may
slightly exceed `max_bond`.

The default `cutoff_mode="rel"` truncates values relative to the largest value
across all retained sectors.

With `max_bond_mode="eager"`, `max_bond` is spread over charge sectors in
proportion to their current sizes *before* the block decompositions are
computed. Each sector receives at least one mode when `max_bond` allows;
otherwise the largest sectors are retained. This is the default for
[`svd_rand_truncated`](#symmray.linalg.svd_rand_truncated), since otherwise
there is no low-rank speedup. An `abs` or `rel` cutoff can further truncate the
retained spectra globally. Cumulative cutoff modes and renormalization are not
compatible with eager mode because they require the discarded spectra.

For flat arrays, both `max_bond_mode` values currently use the equivalent of
`"eager"` truncation. Their sectors have uniform sizes by construction, so
`max_bond` is divided uniformly across the stacked blocks before decomposition.


## Fermionic arrays

For fermionic arrays, [`eigh_truncated`](#symmray.linalg.eigh_truncated) and
Cholesky factorizations drop `dummy_modes` and the array label from returned
factors by default. These factors are normally used as projectors, *added* to
the network with the tensors they were decomposed from still present, as such
the dummy modes should not be duplicated. Pass `drop_dummy_modes=False` to
change this behavior.
