import autoray as ar
import pytest

import symmray as sr


def invert_permutation(p):
    pinv = [0] * len(p)
    for i, v in enumerate(p):
        pinv[v] = i
    return tuple(pinv)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([24, 36], [36, 36], [36, 24]))
@pytest.mark.parametrize(
    "duals",
    (
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ),
)
def test_qr(symmetry, shape, duals, seed=42):
    import autoray as ar

    x = sr.utils.get_rand(
        symmetry,
        shape,
        duals=duals,
        fermionic=True,
        seed=seed,
        subsizes="equal",
    )
    x.randomize_phases(seed=seed + 1, inplace=True)
    q, r = ar.do("linalg.qr", x)

    fx = x.to_flat()
    fx.check()
    fx.to_blocksparse().test_allclose(x)
    fq, fr = ar.do("linalg.qr", fx)
    fq.check()
    fq.to_blocksparse().test_allclose(q)
    fr.check()
    fr.to_blocksparse().test_allclose(r)

    fxr = fq @ fr
    fxr.check()
    fxr.to_blocksparse().test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([24, 36], [36, 36], [36, 24]))
@pytest.mark.parametrize(
    "duals",
    (
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ),
)
def test_svd(symmetry, shape, duals, seed=42):
    import autoray as ar

    x = sr.utils.get_rand(
        symmetry,
        shape,
        duals=duals,
        fermionic=True,
        seed=seed,
        subsizes="equal",
    )
    x.randomize_phases(seed=seed + 1, inplace=True)
    u, s, vh = ar.do("linalg.svd", x)

    fx = x.to_flat()
    fx.check()
    fx.to_blocksparse().test_allclose(x)
    fu, fs, fvh = ar.do("linalg.svd", fx)
    fu.check()
    fu.to_blocksparse().test_allclose(u)
    fs.check()
    fs.to_blockvector().test_allclose(s)
    fvh.check()
    fvh.to_blocksparse().test_allclose(vh)

    fxr = fu @ fvh.multiply_diagonal(fs, axis=0)
    fxr.check()
    fxr.to_blocksparse().test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2",))
@pytest.mark.parametrize("seed", range(10))
def test_qr_roundtrip(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    sx = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[4, 4, 6, 6, 8],
        fermionic=True,
        dist="uniform",
        seed=rng,
        flat=False,
        subsizes="equal",
    )
    sx.randomize_phases(seed=rng, inplace=True)
    x = sx.to_flat()

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
    x.to_blocksparse().test_allclose(sx)


@pytest.mark.parametrize("symmetry", ("Z2",))
@pytest.mark.parametrize("seed", range(5))
def test_qr_with_expand_dims(symmetry, seed):
    pytest.xfail("Unfusing single dimensions not implemented yet")
    x = sr.utils.get_rand(
        symmetry,
        [4, 6, 6],
        seed=seed,
        fermionic=True,
        flat=True,
        subsizes="maximal",
    )
    y = x.reshape((1, 4 * 6 * 6))
    q, r = sr.linalg.qr(y)
    z = (q @ r).reshape((4, 6, 6))
    z.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2",))
@pytest.mark.parametrize("seed", range(10))
def test_svd_roundtrip(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    sx = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[4, 6, 6, 8, 8],
        fermionic=True,
        dist="normal",
        seed=rng,
        subsizes="equal",
    )
    sx.randomize_phases(seed=rng, inplace=True)
    x = sx.to_flat()

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
    x.to_blocksparse().test_allclose(sx)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([84, 96], [96, 96], [96, 84]))
@pytest.mark.parametrize("seed", range(20))
def test_svd_truncated_cutoff_max_bond(symmetry, shape, seed):
    rng = sr.utils.get_rng(seed)

    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=shape,
        fermionic=True,
        dist="uniform",
        seed=rng,
        flat=True,
        subsizes="equal",
    )

    # # cutoff only
    # _, s, _ = ar.do(
    #     "svd_truncated",
    #     x,
    #     cutoff=3e-2,
    #     absorb=None,
    # )
    # assert s.size < 80

    # max_bond only
    _, s, _ = ar.do(
        "svd_truncated",
        x,
        cutoff=0.0,
        max_bond=12,
        absorb=None,
    )
    assert s.size == 12

    # # both
    # _, s, _ = ar.do(
    #     "svd_truncated",
    #     x,
    #     cutoff=1e-2,
    #     max_bond=37,
    #     absorb=None,
    # )
    # assert s.size <= 37


@pytest.mark.parametrize("symm", ("Z2",))
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_eigh_fermionic(symm, seed, dtype):
    x = sr.utils_test.rand_herm(
        symm,
        10,
        seed=seed,
        dtype=dtype,
        flat=True,
        fermionic=True,
    )
    el, ev = sr.linalg.eigh(x)
    # reconstruct the matrix
    y = sr.multiply_diagonal(ev, el, 1) @ ev.H
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([24, 36], [36, 36], [36, 24]))
@pytest.mark.parametrize(
    "duals",
    (
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ),
)
def test_lq_via_qr(symmetry, shape, duals, seed=42):
    import autoray as ar

    x = sr.utils.get_rand(
        symmetry,
        shape,
        duals=duals,
        fermionic=True,
        seed=seed,
        subsizes="equal",
    )
    x.randomize_phases(seed=seed + 1, inplace=True)

    xt = ar.do("transpose", x)
    QT, _, LT = ar.do("qr_stabilized", xt)
    Q = ar.do("transpose", QT)
    L = ar.do("transpose", LT)

    fx = x.to_flat()
    fx.check()
    fx.to_blocksparse().test_allclose(x)
    fxt = ar.do("transpose", fx)
    fxt.check()
    fxt.to_blocksparse().test_allclose(xt)
    fQT, _, fLT = ar.do("qr_stabilized", fxt)
    fQT.check()
    fQT.to_blocksparse().test_allclose(QT)
    fLT.check()
    fLT.to_blocksparse().test_allclose(LT)
    fQ = ar.do("transpose", fQT)
    fQ.check()
    fQ.to_blocksparse().test_allclose(Q)
    fL = ar.do("transpose", fLT)
    fL.check()
    fL.to_blocksparse().test_allclose(L)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([24, 36], [36, 36], [36, 24]))
@pytest.mark.parametrize(
    "duals",
    (
        [False, False],
        [True, False],
        [False, True],
        [True, True],
    ),
)
def test_svd_via_eig(symmetry, shape, duals, seed=42):
    x = sr.utils.get_rand(
        symmetry,
        shape,
        duals=duals,
        fermionic=True,
        seed=seed,
        subsizes="equal",
    )
    x.randomize_phases(seed=seed + 1, inplace=True)

    fx = x.to_flat()
    fx.check()

    fu, fs, fvh = fx.svd_via_eig()
    fu.check()
    fvh.check()
    fs.check()

    fxr = fu @ fvh.multiply_diagonal(fs, axis=0)
    fxr.check()
    fxr.to_blocksparse().test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2",))
@pytest.mark.parametrize("seed", range(10))
def test_svd_via_eig_roundtrip(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    sx = sr.utils.get_rand(
        symmetry=symmetry,
        shape=[4, 6, 6, 8, 8],
        fermionic=True,
        dist="normal",
        seed=rng,
        subsizes="equal",
    )
    sx.randomize_phases(seed=rng, inplace=True)
    x = sx.to_flat()

    axes = tuple(rng.permutation(x.ndim))
    nleft = rng.integers(1, x.ndim - 1)

    axes_left = axes[:nleft]
    axes_right = axes[nleft:]
    order = (*axes_left, *axes_right)
    perm_back = invert_permutation(order)

    # fuse into matrix
    xf = x.fuse(axes_left, axes_right)

    # perform SVD via eig into matrix components
    u, s, vh = xf.svd_via_eig()

    # reconstruct matrix
    xfr = u @ ar.do("multiply_diagonal", vh, s, axis=0)

    # unfuse back into transpose tensor
    xrt = xfr.unfuse_all()

    # permute back to original order
    xr = xrt.transpose(perm_back)

    x.test_allclose(xr)
    x.to_blocksparse().test_allclose(sx)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([24, 36], [36, 36], [36, 24]))
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize("seed", range(5))
def test_svd_via_eig_truncated(symmetry, shape, absorb, seed):
    rng = sr.utils.get_rng(seed)

    sx = sr.utils.get_rand(
        symmetry=symmetry,
        shape=shape,
        fermionic=True,
        dist="uniform",
        seed=rng,
        subsizes="equal",
    )
    sx.randomize_phases(seed=rng, inplace=True)
    x = sx.to_flat()

    u, s, vh = x.svd_via_eig_truncated(absorb=absorb)
    u.check()
    vh.check()

    if absorb is None:
        s.check()
        xr = u @ vh.multiply_diagonal(s, axis=0)
    else:
        assert s is None
        xr = u @ vh

    xr.check()
    assert xr.allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "Z3", "Z4"))
@pytest.mark.parametrize("shape", ([84, 96], [96, 96], [96, 84]))
@pytest.mark.parametrize("seed", range(5))
def test_svd_via_eig_truncated_max_bond(symmetry, shape, seed):
    rng = sr.utils.get_rng(seed)

    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=shape,
        fermionic=True,
        dist="uniform",
        seed=rng,
        flat=True,
        subsizes="equal",
    )

    # max_bond only, using ar.do dispatch
    _, s, _ = ar.do(
        "svd_via_eig_truncated",
        x,
        cutoff=0.0,
        max_bond=12,
        absorb=None,
    )
    assert s.size == 12


@pytest.mark.parametrize("symmetry", ("Z2",))
@pytest.mark.parametrize("shape", ([24, 36], [36, 24]))
@pytest.mark.parametrize("dtype", ("float64", "complex128"))
def test_svd_via_eig_complex(symmetry, shape, dtype, seed=42):
    fx = sr.utils.get_rand(
        symmetry,
        shape,
        fermionic=True,
        dtype=dtype,
        seed=seed,
        flat=True,
        subsizes="equal",
    )

    u, s, vh = fx.svd_via_eig()
    u.check()
    vh.check()
    s.check()

    xr = u @ vh.multiply_diagonal(s, axis=0)
    xr.check()
    assert xr.allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("dtype", ["float64", "complex128", "complex64"])
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_fermionic_flat(symmetry, d, seed, dtype):
    sx = sr.utils_test.rand_posdef(
        symmetry,
        d * int(symmetry[1:]),
        seed=seed,
        dtype=dtype,
        subsizes="equal",
        fermionic=True,
    )
    sleft = sr.linalg.cholesky(sx, upper=False)
    sright = sr.linalg.cholesky(sx, upper=True)

    fx = sx.to_flat()
    fleft = sr.linalg.cholesky(fx)
    fleft.check()
    assert fleft.ndim == 2
    assert fleft.dtype == dtype
    fleft.to_blocksparse().test_allclose(sleft)
    # roundtrip: L @ L^H == A
    fy = fleft @ fleft.dagger_compose_right()
    fy.check()
    fy.test_allclose(fx)
    fy.to_blocksparse().test_allclose(sx)

    fright = sr.linalg.cholesky(fx, upper=True)
    fright.check()
    assert fright.ndim == 2
    assert fright.dtype == dtype
    fright.to_blocksparse().test_allclose(sright)
    # roundtrip: U^H @ U == A
    fy = fright.dagger_compose_left() @ fright
    fy.check()
    fy.test_allclose(fx)
    fy.to_blocksparse().test_allclose(sx)

    # combine lower and upper factors
    fy = fleft @ fright
    fy.check()
    fy.test_allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("d", [4, 8])
@pytest.mark.parametrize("absorb", [-12, 0, 12])
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_regularized_fermionic_flat(symmetry, d, seed, absorb, dtype):
    sx = sr.utils_test.rand_posdef(
        symmetry,
        d * int(symmetry[1:]),
        seed=seed,
        subsizes="equal",
        dtype=dtype,
        fermionic=True,
    )
    sleft, _, sright = sr.linalg.cholesky_regularized(sx, absorb=absorb)

    fx = sx.to_flat()
    fleft, fs, fright = sr.linalg.cholesky_regularized(fx, absorb=absorb)
    assert fs is None

    if absorb == -12:
        assert fright is None
        fleft.check()
        fleft.to_blocksparse().test_allclose(sleft)
        y = fleft @ fleft.dagger_compose_right()
        y.check()
        y.test_allclose(fx)
    elif absorb == 12:
        assert fleft is None
        fright.check()
        fright.to_blocksparse().test_allclose(sright)
        y = fright.dagger_compose_left() @ fright
        y.check()
        y.test_allclose(fx)
    else:
        fleft.check()
        fright.check()
        # roundtrip: L @ R == A
        fy = fleft @ fright
        fy.check()
        fy.test_allclose(fx)
        fleft.to_blocksparse().test_allclose(sleft)
        fright.to_blocksparse().test_allclose(sright)


def test_cholesky_regularized_fermionic_flat_ar_dispatch():
    """Check that autoray dispatch works for cholesky_regularized."""
    sx = sr.utils_test.rand_posdef(
        "Z4",
        8,
        seed=42,
        subsizes="equal",
        fermionic=True,
    )
    fx = sx.to_flat()

    left, s, right = ar.do("cholesky_regularized", fx, absorb=0)
    assert s is None
    left.check()
    right.check()
    fy = left @ right
    fy.check()
    fy.test_allclose(fx)
