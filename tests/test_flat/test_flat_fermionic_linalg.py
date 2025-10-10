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
    d = 10
    i = sr.utils.rand_index(
        symm,
        d,
        seed=seed,
        subsizes="equal",
    )
    x = sr.utils.get_rand(
        symm,
        shape=(i, i.conj()),
        fermionic=True,
        dtype=dtype,
        seed=seed,
        flat=True,
    )
    # needs to be hermitian for eigh
    x = (x + x.H) / 2
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
