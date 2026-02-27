import autoray as ar
import pytest

import symmray as sr


def invert_permutation(p):
    pinv = [0] * len(p)
    for i, v in enumerate(p):
        pinv[v] = i
    return tuple(pinv)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_qr_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        duals=[f0, f1],
        charge=c,
        fermionic=True,
        label="x",
    )
    x.check()
    q, r = sr.linalg.qr(x)
    q.check()
    r.check()
    assert (q @ r).allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_svd_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        duals=[f0, f1],
        charge=c,
        fermionic=True,
        subsizes="maximal",
        label="x",
        seed=42,
    )
    x.check()
    u, s, vh = sr.linalg.svd(x)
    u.check()
    s.check()
    vh.check()
    assert (u @ ar.do("ldmul", s, vh)).allclose(x)


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

    # perform QR into matrix components
    q, r = ar.do("linalg.qr", xf)

    # reconstruct matrix
    xfr = q @ r

    # unfuse back into transpose tensor
    xrt = xfr.unfuse_all()

    # permute back to original order
    xr = xrt.transpose(perm_back)

    x.test_allclose(xr)


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
    z.test_allclose(x)


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

    x.test_allclose(xr)


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

    x.test_allclose(xr)


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
    x = sr.utils_test.rand_herm(
        symm, 10, seed=seed, dtype=dtype, fermionic=True
    )
    el, ev = sr.linalg.eigh(x)
    # reconstruct the matrix
    y = sr.multiply_diagonal(ev, el, 1) @ ev.H
    y.test_allclose(x)


@pytest.mark.parametrize("symm", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_eigh_truncated_fermionic(symm, seed, dtype):
    x = sr.utils_test.rand_herm(
        symm, 10, seed=seed, dtype=dtype, fermionic=True
    )
    u, s, vh = sr.linalg.eigh_truncated(x, max_bond=-1, absorb=None)
    # reconstruct the matrix
    y = sr.multiply_diagonal(u, s, 1) @ vh
    y.test_allclose(x)

    z = u @ u.dagger_project_left() @ x @ vh.dagger_project_right() @ vh
    z.test_allclose(x)


@pytest.mark.parametrize("symm", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_eigh_truncated_fermionic_fsued(symm, seed, dtype):
    import cotengra as ctg

    di = 4
    dj = 3
    i = sr.utils.rand_index(
        symm,
        di,
        subsizes=None,
        seed=seed,
    )
    j = sr.utils.rand_index(
        symm,
        dj,
        subsizes=None,
        seed=seed,
    )

    x = sr.utils.get_rand(
        symm,
        (i, j, i.conj(), j.conj()),
        fermionic=True,
        dtype=dtype,
    )
    x.randomize_phases(inplace=True)

    xf = x.fuse((0, 1), (2, 3))

    for sector, block in xf.get_sector_block_pairs():
        xf.set_block(sector, block + block.conj().T)

    # get unfused hermitian version
    x = xf.unfuse_all()

    # decompose
    u, s, vh = ar.do("eigh_truncated", xf, absorb=None)

    # reconstruct
    y = u.multiply_diagonal(s, axis=1) @ vh
    y.unfuse_all().test_allclose(x)

    # recostruct via identity resolved projectors
    udag = u.dagger_project_left().unfuse_all()
    vhdag = vh.dagger_project_right().unfuse_all()
    y = ctg.array_contract(
        arrays=[u.unfuse_all(), udag, x, vhdag, vh.unfuse_all()],
        inputs=[
            tuple("abc"),
            tuple("cde"),
            tuple("defg"),
            tuple("fgh"),
            tuple("hij"),
        ],
        output=tuple("abij"),
    )
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
@pytest.mark.parametrize("seed", range(1))
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
def test_cholesky_fermionic(symmetry, d, seed, dtype):
    x = sr.utils_test.rand_posdef(
        symmetry, d, seed=seed, dtype=dtype, fermionic=True
    )

    left = sr.linalg.cholesky(x, upper=False)
    left.check()
    assert left.ndim == 2
    assert left.dtype == dtype
    # roundtrip: L @ L^H should equal A
    y = left @ left.dagger_compose_right()
    y.check()
    y.test_allclose(x)

    right = sr.linalg.cholesky(x, upper=True)
    right.check()
    assert right.ndim == 2
    assert right.dtype == dtype
    # roundtrip: R^H @ R should equal A
    y = right.dagger_compose_left() @ right
    y.check()
    y.test_allclose(x)

    # check left/lower and right/upper are consistent with each other
    y = left @ right
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 5))
@pytest.mark.parametrize("absorb", [-12, 0, 12])
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_regularized_fermionic(symmetry, d, absorb, seed, dtype):
    x = sr.utils_test.rand_posdef(
        symmetry, d, seed=seed, dtype=dtype, fermionic=True
    )

    left, s, right = sr.linalg.cholesky_regularized(x, absorb=absorb)
    assert s is None

    if absorb == -12:
        assert right is None
        left.check()
        y = left @ left.dagger_compose_right()
        y.check()
        y.test_allclose(x)
    elif absorb == 12:
        assert left is None
        right.check()
        y = right.dagger_compose_left() @ right
        y.check()
        y.test_allclose(x)
    else:
        left.check()
        right.check()
        y = left @ right
        y.check()
        y.test_allclose(x)


def test_cholesky_regularized_fermionic_ar_dispatch():
    """Check that autoray dispatch works for cholesky_regularized."""
    x = sr.utils_test.rand_posdef(
        "U1", 2, seed=0, dtype="complex128", fermionic=True
    )
    left, s, right = ar.do("cholesky_regularized", x)
    assert s is None
    left.check()
    right.check()
    y = left @ right
    y.check()
    y.test_allclose(x)


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
        symm, shape=(i, j), fermionic=1, label="a", seed=seed
    )
    b = sr.utils.get_rand(
        symm, shape=(i,), fermionic=1, charge=b_charge, label="b", seed=seed
    )
    x = sr.linalg.solve(A, b)
    (A @ x).test_allclose(b)


@pytest.mark.parametrize("symm", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("b_charge", [False, True])
def test_solve_fermionic_2d(symm, seed, b_charge):
    if symm in ("Z2", "U1"):
        b_charge = 1 if b_charge else 0
    elif symm in ("Z2Z2", "U1U1"):
        b_charge = (1, 0) if b_charge else (0, 1)

    i = sr.utils.rand_index(symm, 100, seed=seed)
    j = i.conj()
    k = sr.utils.rand_index(symm, 20, seed=seed)
    A = sr.utils.get_rand(
        symm, shape=(i, j), fermionic=1, label="a", seed=seed
    )
    b = sr.utils.get_rand(
        symm, shape=(i, k), fermionic=1, charge=b_charge, label="b", seed=seed
    )
    x = sr.linalg.solve(A, b)
    (A @ x).test_allclose(b)


@pytest.mark.parametrize("symm", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(20))
def test_svd_projector_identity(symm, seed):
    import cotengra as ctg

    x = sr.utils.get_rand(
        symm,
        (4, 4, 5, 3),
        fermionic=True,
        seed=seed,
    )
    x.randomize_phases(inplace=True)
    xf = x.fuse((0, 1), (2, 3))

    u, s, vh = ar.do("linalg.svd", xf)
    udag = u.dagger_project_left()
    vhdag = vh.dagger_project_right()

    udag = udag.unfuse_all()
    vhdag = vhdag.unfuse_all()

    z = ctg.einsum(
        "abc,cde,defg,fgh,hij->abij",
        u.unfuse_all(),
        udag,
        x,
        vhdag,
        vh.unfuse_all(),
    )
    z.test_allclose(x)
