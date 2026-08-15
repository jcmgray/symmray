<img src="https://github.com/jcmgray/symmray/blob/HEAD/docs/_static/logo-title.png?raw=true" alt="symmray logo" width="640"/>

[![Tests](https://github.com/jcmgray/symmray/actions/workflows/tests.yml/badge.svg)](https://github.com/jcmgray/symmray/actions/workflows/tests.yml)
[![Code Coverage](https://codecov.io/gh/jcmgray/symmray/branch/main/graph/badge.svg)](https://codecov.io/gh/jcmgray/symmray)
[![Docs](https://readthedocs.org/projects/symmray/badge/?version=latest)](https://symmray.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/symmray?color=teal)](https://pypi.org/project/symmray/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/symmray/badges/version.svg)](https://anaconda.org/conda-forge/symmray)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

[`symmray`](https://symmray.readthedocs.io/) provides block-sparse arrays with
abelian symmetries and fermionic signs. Its arrays follow the NumPy interface
where possible and can store blocks using NumPy, PyTorch, JAX, or another
[`autoray`](https://autoray.readthedocs.io/) backend.

Key features include:

- sparse and flat storage layouts
- `Z2`, `U1`, `Z2Z2`, `U1U1`, and finite cyclic symmetries
- local fermionic phase handling for arbitrary tensor-network geometries
- contraction, fusion, reshaping, and blockwise linear algebra
- tensor-network and Hamiltonian constructors for
  [`quimb`](https://quimb.readthedocs.io/)

```python
import symmray as sr

# some U(1) indices
a, b, c, d, e = [
    sr.utils.rand_index("U1", size, dual=False, seed=seed)
    for seed, size in enumerate((4, 6, 8, 5, 7))
]

# some fermionic arrays
X = sr.utils.get_rand(
    "U1",
    shape=(a, b, c.conj()),
    fermionic=True,
    seed=6,
)
Y = sr.utils.get_rand(
    "U1",
    shape=(c, d.conj(), e.conj()),
    fermionic=True,
    seed=7,
)

# contract them
Z = sr.einsum("abc,cde->abde", X, Y)

# fuse into a matrix
matrix = Z.reshape((4 * 6, 5 * 7))

# decompose
U, s, VH = sr.linalg.svd(matrix)
print(U)
# U1FermionicArray(ndim=2, charge=0, indices=[
#     (24 = 2+4+6+6+4+2 : +[-2,-1,0,1,2,3])
#         -2 ; (2) : [(-1, -1)]
#         -1 ; (2+2) : [(-1, 0),(0, -1)]
#         0 ; (2+2+2) : [(-1, 1),(0, 0),(1, -1)]
#         1 ; (2+2+2) : [(0, 1),(1, 0),(2, -1)]
#         2 ; (2+2) : [(1, 1),(2, 0)]
#         3 ; (2) : [(2, 1)]
#     (23 = 2+4+5+6+4+2 : -[-2,-1,0,1,2,3])
# ], num_blocks=6, backend=numpy, dtype=float64)
```

See the [documentation](https://symmray.readthedocs.io/) for installation,
guides, examples, and the API reference. Contributions are welcome; see the
[development guide](https://symmray.readthedocs.io/en/latest/develop.html) and
[GitHub issues](https://github.com/jcmgray/symmray/issues).
