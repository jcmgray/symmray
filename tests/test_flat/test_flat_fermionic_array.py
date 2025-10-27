import pytest

import symmray as sr

from .test_flat_abelian_array import get_zn_blocksparse_flat_compat


@pytest.mark.parametrize("symmetry", ("Z2", "Z4"))
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("all_axes", [False, True])
def test_sort_sectors(
    symmetry,
    seed,
    ndim,
    all_axes,
):
    rng = sr.utils.get_rng(seed)
    shape = [rng.choice([2, 4]) for _ in range(ndim)]
    if ndim > 2:
        num_sort = rng.integers(1, ndim)
    else:
        num_sort = 1
    axes_sort = tuple(rng.choice(ndim, size=num_sort, replace=False))
    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=shape,
        charge=0,
        fermionic=True,
        seed=rng,
    )
    x.randomize_phases(seed + 1, inplace=True)
    fx: sr.FermionicArrayFlat = x.to_flat()
    fx.sort_stack(axes_sort, all_axes=all_axes, inplace=True)
    fx.check()
    y = fx.to_blocksparse()
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("sync", [False, True])
def test_to_and_from_blocksparse_with_phase_sync(
    symmetry,
    charge,
    seed,
    sync,
):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    x = get_zn_blocksparse_flat_compat(
        symmetry,
        (2, 4, 6),
        charge=charge,
        fermionic=True,
        seed=seed,
    )
    # add some non-trivial phases
    x.transpose((2, 0, 1), inplace=True)
    assert x.phases
    fx = x.to_flat()
    assert fx.fermionic
    if sync:
        fx.phase_sync(inplace=True)
    fx.check()
    y = fx.to_blocksparse()
    assert sync == (not y.phases)
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("axs", [(), (0,), (1,), (0, 2), (0, 2, 1)])
def test_phase_flip(
    symmetry,
    charge,
    seed,
    sync,
    axs,
):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    x = get_zn_blocksparse_flat_compat(
        symmetry,
        (2, 4, 6),
        charge=charge,
        fermionic=True,
        seed=seed,
    )
    x.randomize_phases(seed + 1, inplace=True)
    assert x.phases
    fx = x.to_flat()
    xflipped = x.phase_flip(*axs)
    fxflipped = fx.phase_flip(*axs)
    if sync:
        fxflipped.phase_sync(inplace=True)
    fxflipped.check()
    y = fxflipped.to_blocksparse()
    y.check()
    if sync:
        # phases should have been absorbed into blocks
        assert not y.phases
    y.test_allclose(xflipped)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("sync", [False, True])
def test_phase_transpose(
    symmetry,
    charge,
    seed,
    sync,
):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    rng = sr.utils.get_rng(seed)
    N = rng.integers(1, 7)
    perm = tuple(rng.permutation(N))
    shape = tuple(rng.choice([2, 4], size=N))

    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape,
        charge=charge,
        fermionic=True,
        seed=seed,
    )
    # add some non-trivial phases
    x.randomize_phases(seed + 1, inplace=True)
    fx = x.to_flat()

    x_phase_transposed = x.phase_transpose(perm)
    fx_phase_transposed = fx.phase_transpose(perm)
    if sync:
        fx_phase_transposed.phase_sync(inplace=True)
    fx_phase_transposed.check()
    y = fx_phase_transposed.to_blocksparse()
    y.check()
    if sync:
        # phases should have been absorbed into blocks
        assert not y.phases
    y.test_allclose(x_phase_transposed)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
def test_phase_global(symmetry, charge, seed):
    if charge:
        pytest.xfail("oddpos not implemented yet.")
    x = get_zn_blocksparse_flat_compat(
        symmetry,
        (2, 4, 6),
        charge=charge,
        fermionic=True,
        seed=seed,
    )
    # add some non-trivial phases
    x.randomize_phases(seed + 1, inplace=True)
    fx = x.to_flat()
    fxg = fx.phase_global()
    fxg.to_blocksparse().test_allclose(x.phase_global())


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("sync", [False, True])
def test_transpose(
    symmetry,
    charge,
    ndim,
    seed,
    sync,
):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    rng = sr.utils.get_rng(seed)

    shape = [rng.choice([2, 4]) for _ in range(ndim)]

    if rng.random() < 0.2:
        perm = None
    else:
        perm = rng.permutation(ndim)

    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape,
        charge=charge,
        fermionic=True,
        seed=rng,
    )
    x.randomize_phases(rng, inplace=True)
    xt = x.transpose(perm)
    fx = x.to_flat()
    fxt = fx.transpose(perm)
    fxt.check()
    if sync:
        fxt.phase_sync(inplace=True)
    y = fxt.to_blocksparse()
    y.check()
    y.test_allclose(xt)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("shape", [(2, 6, 4), (2, 2, 2, 2)])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("phase_permutation", [False, True])
@pytest.mark.parametrize("phase_dual", [False, True])
def test_conj(
    symmetry,
    shape,
    charge,
    seed,
    sync,
    phase_permutation,
    phase_dual,
):
    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape,
        charge=charge,
        fermionic=True,
        oddpos="x",
        seed=seed,
    )
    # add some non-trivial phases
    x.randomize_phases(seed + 1, inplace=True)
    assert x.phases
    xc = x.conj(phase_permutation=phase_permutation, phase_dual=phase_dual)

    fx = x.to_flat()
    fxc = fx.conj(phase_permutation=phase_permutation, phase_dual=phase_dual)
    fxc.check()
    if sync:
        fxc.phase_sync(inplace=True)
    y = fxc.to_blocksparse()
    y.check()
    y.test_allclose(xc)


@pytest.mark.parametrize("symmetry", ("Z2", "Z4"))
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
@pytest.mark.parametrize("seed", range(5))
def test_dagger(symmetry, charge, ndim, dtype, seed):
    rng = sr.utils.get_rng(seed)
    xs = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=[2] * ndim,
        charge=charge,
        fermionic=True,
        oddpos="x",
        seed=rng,
        dtype=dtype,
    )
    xs.randomize_phases(seed + 1, inplace=True)
    x = xs.to_flat()
    x.dagger().test_allclose(x.H)
    x.dagger().test_allclose(x.conj().transpose())
    x.dagger().to_blocksparse().test_allclose(xs.dagger())


@pytest.mark.parametrize(
    "symmetry, charge",
    [
        ("Z2", 0),
        ("Z2", 1),
        ("Z4", 0),
        ("Z4", 1),
        ("Z4", 2),
        ("Z4", 3),
    ],
)
@pytest.mark.parametrize("seed", range(3))
def test_fermi_norm_phase_dual(symmetry, charge, seed):
    sx = sr.utils.get_rand(
        symmetry,
        (4, 8, 4, 8),
        subsizes="equal",
        fermionic=True,
        charge=charge,
        oddpos="x",
        flat=False,
        seed=seed,
    )
    x = sx.to_flat()
    n2 = x.norm() ** 2

    xd = x.dagger(phase_dual=True)
    assert xd.to_blocksparse().test_allclose(sx.dagger(phase_dual=True))
    na = sr.tensordot(xd, x, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    nb = sr.tensordot(x, xd, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    assert na == pytest.approx(nb)
    assert na == pytest.approx(n2)
    assert nb == pytest.approx(n2)

    xc = x.conj(phase_dual=True)
    assert xc.to_blocksparse().test_allclose(sx.conj(phase_dual=True))
    nc = sr.tensordot(xc, x, axes=[(0, 1, 2, 3), (0, 1, 2, 3)])
    nd = sr.tensordot(x, xc, axes=[(0, 1, 2, 3), (0, 1, 2, 3)])
    assert nc == pytest.approx(nd)
    assert nc == pytest.approx(n2)
    assert nd == pytest.approx(n2)

    xct = x.conj(phase_dual=True).transpose()
    assert xct.to_blocksparse().test_allclose(
        sx.conj(phase_dual=True).transpose()
    )
    ne = sr.tensordot(xct, x, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    nf = sr.tensordot(x, xct, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    assert ne == pytest.approx(nf)
    assert ne == pytest.approx(n2)
    assert nf == pytest.approx(n2)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize(
    "shape,axes_groups",
    [
        ((2, 4, 6, 4), ((0,), (1,), (2,), (3,))),
        ((2, 4, 6, 4), ()),
        ((2, 4, 6, 4), ((0, 1),)),
        ((2, 4, 6, 4), ((1, 2),)),
        ((2, 4, 6, 4), ((1, 3),)),
        ((2, 4, 6, 4), ((2, 3),)),
        ((2, 4, 6, 4), ((0, 1), (2, 3))),
        ((2, 4, 6, 4), ((0, 2), (1, 3))),
        ((2, 4, 6, 4), ((0, 3), (1, 2))),
        ((2, 4, 6, 4), ((0, 3, 1, 2),)),
        ((2, 4, 6, 4, 6), ((0, 1), (2, 3, 4))),
        ((2, 4, 6, 4, 6), ((0, 2), (1, 3, 4))),
        ((2, 4, 6, 4, 6), ((0, 4), (1, 2, 3))),
        ((2, 4, 6, 4, 6), ((0, 1, 2), (3, 4))),
        ((2, 4, 6, 4, 6), ((0, 1, 3), (2, 4))),
        ((2, 4, 6, 4, 6), ((0, 1, 4), (2, 3))),
    ],
)
def test_fuse(
    symmetry,
    charge,
    shape,
    axes_groups,
):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=shape,
        charge=charge,
        fermionic=True,
        seed=42,
    )
    x.randomize_phases(43, inplace=True)
    x_fused = x.fuse(*axes_groups, inplace=False)
    fx = x.to_flat()
    fx_fused = fx.fuse(*axes_groups, inplace=False)
    fx_fused.check()
    y = fx_fused.to_blocksparse()
    y.check()
    y.test_allclose(x_fused)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("seed", range(10))
def test_fuse_unfuse(symmetry, charge, seed):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    rng = sr.utils.get_rng(seed)
    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=(2, 4, 6, 4, 8),
        charge=charge,
        fermionic=True,
        seed=rng,
    )
    x.randomize_phases(seed + 1, inplace=True)
    fx = x.to_flat()

    nfuse = rng.integers(1, x.ndim)
    axes = tuple(rng.choice(x.ndim, size=nfuse, replace=False))
    position = min(axes)
    new_order = (
        *range(position),
        *axes,
        *(ax for ax in range(position, x.ndim) if ax not in axes),
    )
    perm_back = sorted(range(x.ndim), key=lambda i: new_order[i])

    x_fused = x.fuse(axes)
    fx_fused = fx.fuse(axes)
    fx_fused.check()
    fx_fused.to_blocksparse().test_allclose(x_fused)

    if len(axes) > 1:
        y = x_fused.unfuse(position)
        fy = fx_fused.unfuse(position)
        fy.check()
        fy.to_blocksparse().test_allclose(y)
    else:
        y = x_fused
        fy = fx_fused

    fyt = fy.transpose(perm_back)
    fyt.check()
    fyt.to_blocksparse().test_allclose(x)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize(
    "shape,axes_groups",
    [
        ([4], [(0,)]),
        ([2, 2], [(0, 1)]),
        ([2, 2], [(0,), (1,)]),
        ([2, 2], [(1,), (0,)]),
        ([4] * 6, [(1, 3), (4, 2)]),
        ([6, 2, 4, 8], [(3, 2, 1)]),
        ([6, 2, 4, 8], [(0, 1)]),
        ([6, 2, 4, 8], [(2, 3)]),
        ([6, 2, 4, 8], [(0, 1, 2, 3)]),
        ([6, 2, 4, 8], [(2, 3, 1, 0)]),
        ([6, 2, 4, 8], [(0, 1), (2, 3)]),
        ([2, 2, 2, 2, 2], [(0, 1), (2, 3, 4)]),
        ([2, 2, 2, 2, 2, 2], [(0, 1), (2, 3), (4, 5)]),
        ([2, 2, 2, 2, 2, 2], [(0, 1), (4, 5), (2, 3)]),
        ([4, 2, 6, 2], [(0, 3)]),
        ([2, 2, 2, 2, 2, 2, 2, 2], [(5,), (7, 2, 3), (1, 4)]),
    ],
)
@pytest.mark.parametrize("charge", [0, 1])
def test_fuse_roundtrip(symmetry, shape, axes_groups, charge):
    if charge:
        pytest.xfail("oddpos not implemented yet.")

    sx = get_zn_blocksparse_flat_compat(
        symmetry, shape, charge, fermionic=True, seed=42
    )
    sx.randomize_phases(43, inplace=True)
    sy = sx.fuse(*axes_groups)
    fx = sx.to_flat()
    fx.check()
    fx.to_blocksparse().test_allclose(sx)
    fy = fx.fuse(*axes_groups)
    fy.check()
    fy.to_blocksparse().test_allclose(sy)
    xu = fy.unfuse_all()
    xu.check()
    # fuse + unfuse is identity up to permutation of axes
    axes_grouped = [i for g in axes_groups for i in g]
    axes_rem = [i for i in range(fx.ndim) if i not in axes_grouped]
    ax_g0 = min(axes_grouped)
    new_axes = axes_rem[:ax_g0] + axes_grouped + axes_rem[ax_g0:]
    sxt = sx.transpose(new_axes)
    fxus = xu.to_blocksparse()
    fxus.test_allclose(sxt)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("seed", range(50))
def test_tensordot(symmetry, seed):
    N = int(symmetry[1:])

    a, b, axes = sr.utils_test.rand_valid_tensordot(
        symmetry=symmetry,
        fermionic=True,
        dimension_multiplier=N,
        subsizes="equal",
        seed=seed,
    )
    a.randomize_phases(seed + 1, inplace=True)
    b.randomize_phases(seed + 2, inplace=True)
    c = a.tensordot(b, axes=axes, preserve_array=True)

    fa = a.to_flat()
    fb = b.to_flat()
    fc = fa.tensordot(fb, axes=axes, preserve_array=True)
    fc.check()

    if c.is_zero() and fc.is_zero():
        # both are zero, other tests might break
        return

    fc.to_blocksparse().test_allclose(c)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("ndim_a", [1, 2])
@pytest.mark.parametrize("ndim_b", [1, 2])
@pytest.mark.parametrize("charge_a", [0, 1])
@pytest.mark.parametrize("charge_b", [0, 1])
@pytest.mark.parametrize("seed", range(10))
def test_matmul(symmetry, ndim_a, ndim_b, charge_a, charge_b, seed):
    rng = sr.utils.get_rng(seed)
    da = rng.choice([12, 24, 36])
    db = rng.choice([12, 24, 36])
    dc = rng.choice([12, 24, 36])
    a = sr.utils.rand_index(symmetry, da, subsizes="equal", seed=rng)
    b = sr.utils.rand_index(symmetry, db, subsizes="equal", seed=rng)
    c = sr.utils.rand_index(symmetry, dc, subsizes="equal", seed=rng)
    if ndim_a == 1:
        shape_a = [b]
    else:
        shape_a = [a, b]
    x = sr.utils.get_rand(
        symmetry,
        shape_a,
        seed=rng,
        charge=charge_a,
        fermionic=True,
        oddpos="x",
    )
    x.randomize_phases(rng, inplace=True)
    if ndim_b == 1:
        shape_b = [b.conj()]
    else:
        shape_b = [b.conj(), c]
    y = sr.utils.get_rand(
        symmetry,
        shape_b,
        seed=rng,
        charge=charge_b,
        fermionic=True,
        oddpos="y",
    )
    y.randomize_phases(rng, inplace=True)
    z = x @ y
    fx = x.to_flat()
    fy = y.to_flat()
    fz = fx @ fy

    if hasattr(fz, "check"):
        fz.check()
        fz.to_blocksparse().test_allclose(z)
    else:
        # scalar
        assert fz == pytest.approx(z)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_block_multiply_diagonal(symmetry, charge, axis):
    import autoray as ar
    import numpy as np

    if charge:
        pytest.xfail("oddpos not implemented yet.")

    rng = np.random.default_rng(42)
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        (4, 2, 6, 2),
        seed=rng,
        charge=charge,
        fermionic=True,
    )
    sx.randomize_phases(rng, inplace=True)
    x = sx.to_flat()

    v = sr.FlatVector.rand(symmetry, x.indices[axis].charge_size)
    y = ar.do("multiply_diagonal", x, v, axis=axis)

    # check dense reference
    xd = x.to_dense()
    vd = v.to_dense()
    yd = y.to_dense()
    lhs = "abcd"
    rhs = lhs[axis]
    np.testing.assert_allclose(yd, np.einsum(f"{lhs},{rhs}->{lhs}", xd, vd))


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("fn", ["ldmul", "lddiv"])
def test_ldmul_and_lddiv(symmetry, seed, fn):
    import autoray as ar

    rng = sr.utils.get_rng(seed)
    da = rng.choice([12, 24, 36])
    db = rng.choice([12, 24, 36])
    shape_x = (da, db)
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape_x,
        charge=0,
        fermionic=True,
        seed=rng,
    )
    sx.randomize_phases(rng, inplace=True)
    vblocks = {
        c: rng.standard_normal(dc) + 1j * rng.standard_normal(dc)
        for c, dc in sx.indices[0].chargemap.items()
    }
    sv = sr.BlockVector(vblocks)
    sy = ar.do(fn, sv, sx)
    fx = sx.to_flat()
    fv = sv.to_flat()
    fy = ar.do(fn, fv, fx)
    fy.check()
    fy.to_blocksparse().test_allclose(sy)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("fn", ["rdmul", "rddiv"])
def test_rdmul_and_rddiv(symmetry, seed, fn):
    import autoray as ar

    rng = sr.utils.get_rng(seed)
    da = rng.choice([12, 24, 36])
    db = rng.choice([12, 24, 36])
    shape_x = (da, db)
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape_x,
        charge=0,
        fermionic=True,
        seed=rng,
    )
    sx.randomize_phases(rng, inplace=True)
    vblocks = {
        c: rng.standard_normal(dc) + 1j * rng.standard_normal(dc)
        for c, dc in sx.indices[1].chargemap.items()
    }
    sv = sr.BlockVector(vblocks)
    sy = ar.do(fn, sx, sv)
    fx = sx.to_flat()
    fv = sv.to_flat()
    fy = ar.do(fn, fx, fv)
    fy.check()
    fy.to_blocksparse().test_allclose(sy)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_norm(symmetry, ndim, seed):
    rng = sr.utils.get_rng(seed)
    shape = [rng.choice([2, 4]) for _ in range(ndim)]
    x = get_zn_blocksparse_flat_compat(
        symmetry,
        shape=shape,
        charge=0,
        fermionic=True,
        seed=rng,
    )
    x.randomize_phases(seed + 1, inplace=True)
    n1 = x.norm()
    fx = x.to_flat()
    n2 = fx.norm()
    assert n1 == pytest.approx(n2)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
@pytest.mark.parametrize("duals", [(False, True), (True, False)])
def test_eigh_fermionic_flat(symmetry, seed, dtype, duals):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        (4, 4),
        fermionic=True,
        seed=seed,
        duals=duals,
        dtype=dtype,
    )
    sx.randomize_phases(seed + 1, inplace=True)
    # needs to be hermitian for eigh
    for sector, block in sx.get_sector_block_pairs():
        sx.set_block(sector, block + block.conj().T)

    sw, sU = sr.linalg.eigh(sx)
    sy = sr.multiply_diagonal(sU, sw, 1) @ sU.H
    sy.test_allclose(sx)

    fx = sx.to_flat()
    w, U = sr.linalg.eigh(fx)
    w.check()
    U.check()

    assert U.to_blocksparse().test_allclose(sU)

    y = sr.multiply_diagonal(U, w, 1) @ U.H
    y.test_allclose(fx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
@pytest.mark.parametrize("duals", [(False, True), (True, False)])
def test_eigh_truncated_fermionic_flat(symmetry, seed, dtype, duals):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        (4, 4),
        fermionic=True,
        seed=seed,
        duals=duals,
        dtype=dtype,
    )
    sx.randomize_phases(seed + 1, inplace=True)
    # needs to be hermitian for eigh
    for sector, block in sx.get_sector_block_pairs():
        sx.set_block(sector, block + block.conj().T)

    fx = sx.to_flat()
    u, w, uh = sr.linalg.eigh_truncated(fx, max_bond=-1, absorb=None)
    y = sr.multiply_diagonal(u, w, 1) @ uh
    y.test_allclose(fx)

    z = u @ u.dagger_project_left() @ fx @ uh.dagger_project_right() @ uh
    z.test_allclose(fx)
