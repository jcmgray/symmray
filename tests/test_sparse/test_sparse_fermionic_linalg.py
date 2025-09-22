import autoray as ar
import pytest

import symmray as sr


def invert_permutation(p):
    pinv = [0] * len(p)
    for i, v in enumerate(p):
        pinv[v] = i
    return tuple(pinv)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
def test_qr_roundtrip(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[4, 5, 6, 7, 8],
        fermionic=True,
        dist="uniform",
        seed=rng,
    )

    axes = tuple(rng.permutation(x.ndim))
    nleft = rng.integers(1, x.ndim - 1)

    axes_left = axes[:nleft]
    axes_right = axes[nleft:]
    order = (*axes_left, *axes_right)
    perm_back = invert_permutation(order)

    # fuse into matrix
    xf = x.fuse(axes_left, axes_right)

    # perform SVD into matrix components
    q, r = ar.do("linalg.qr", xf)

    # reconstruct matrix
    xfr = q @ r

    # unfuse back into transpose tensor
    xrt = xfr.unfuse_all()

    # permute back to original order
    xr = xrt.transpose(perm_back)

    assert x.allclose(xr)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1", "Z2Z2"))
@pytest.mark.parametrize("seed", range(5))
def test_qr_with_expand_dims(symmetry, seed):
    x = sr.utils.get_rand(
        symmetry,
        [4, 5, 6],
        subsizes="maximal",
        seed=seed,
        fermionic=True,
    )
    y = x.reshape((1, 4 * 5 * 6))
    q, r = sr.linalg.qr(y)
    z = (q @ r).reshape((4, 5, 6))
    assert z.allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
def test_svd_roundtrip(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[4, 5, 6, 7, 8],
        fermionic=True,
        dist="normal",
        seed=rng,
    )

    axes = tuple(rng.permutation(x.ndim))
    nleft = rng.integers(1, x.ndim - 1)

    axes_left = axes[:nleft]
    axes_right = axes[nleft:]
    order = (*axes_left, *axes_right)
    perm_back = invert_permutation(order)

    # fuse into matrix
    xf = x.fuse(axes_left, axes_right)

    # perform SVD into matrix components
    u, s, vh = ar.do("linalg.svd", xf)

    # reconstruct matrix
    xfr = u @ ar.do("multiply_diagonal", vh, s, axis=0)

    # unfuse back into transpose tensor
    xrt = xfr.unfuse_all()

    # permute back to original order
    xr = xrt.transpose(perm_back)

    assert x.allclose(xr)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("absorb", [-1, 0, 1])
def test_svd_truncated_roundtrip(symmetry, seed, absorb):
    rng = sr.utils.get_rng(seed)

    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[4, 5, 6, 7, 8],
        fermionic=True,
        dist="normal",
        seed=rng,
    )

    axes = tuple(rng.permutation(x.ndim))
    nleft = rng.integers(1, x.ndim - 1)

    axes_left = axes[:nleft]
    axes_right = axes[nleft:]
    order = (*axes_left, *axes_right)
    perm_back = invert_permutation(order)

    # fuse into matrix
    xf = x.fuse(axes_left, axes_right)

    # perform SVD into matrix components
    u, _, vh = ar.do("svd_truncated", xf, cutoff=1e-10, absorb=absorb)

    # reconstruct matrix
    xfr = u @ vh

    # unfuse back into transpose tensor
    xrt = xfr.unfuse_all()

    # permute back to original order
    xr = xrt.transpose(perm_back)

    assert x.allclose(xr)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(20))
def test_svd_truncated_cutoff_max_bond(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[100, 100],
        fermionic=True,
        dist="uniform",
        subsizes=None,
        seed=rng,
    )

    # cutoff only
    _, s, _ = ar.do(
        "svd_truncated",
        x,
        cutoff=3e-2,
        absorb=None,
    )
    assert s.size < 80

    # max_bond only
    _, s, _ = ar.do(
        "svd_truncated",
        x,
        cutoff=0.0,
        max_bond=5,
        absorb=None,
    )
    assert s.size == 5

    # both
    _, s, _ = ar.do(
        "svd_truncated",
        x,
        cutoff=1e-2,
        max_bond=37,
        absorb=None,
    )
    assert s.size <= 37


@pytest.mark.parametrize("symm", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_eigh_fermionic(symm, seed, dtype):
    d = 10
    i = sr.utils.rand_index(
        symm,
        d,
        subsizes=None,
        seed=seed,
    )
    x = sr.utils.get_rand(
        symm,
        shape=(i, i.conj()),
        fermionic=True,
        dtype=dtype,
        seed=seed,
    )
    # needs to be hermitian for eigh
    x = x + x.H
    el, ev = sr.linalg.eigh(x)
    # reconstruct the matrix
    y = sr.multiply_diagonal(ev, el, 1) @ ev.H
    assert y.allclose(x)


@pytest.mark.parametrize("symm", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("b_charge", [False, True])
def test_solve_fermionic(symm, seed, b_charge):
    if symm in ("Z2", "U1"):
        b_charge = 1 if b_charge else 0
    elif symm in ("Z2Z2", "U1U1"):
        b_charge = (1, 0) if b_charge else (0, 1)

    i = sr.utils.rand_index(symm, 100, seed=seed)
    j = i.conj()
    A = sr.utils.get_rand(
        symm, shape=(i, j), fermionic=1, oddpos="a", seed=seed
    )
    b = sr.utils.get_rand(
        symm, shape=(i,), fermionic=1, charge=b_charge, oddpos="b", seed=seed
    )
    x = sr.linalg.solve(A, b)
    assert (A @ x).allclose(b)
