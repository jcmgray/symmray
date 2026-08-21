import random

import pytest

import symmray as sr


def get_zn_blocksparse_flat_compat(
    symmetry,
    shape,
    charge=0,
    seed=42,
    shape_relative_to_z2=True,
    **kwargs,
):
    rng = sr.utils.get_rng(seed)

    N = int(symmetry[1:])

    if shape_relative_to_z2:
        shape = [N * d // 2 for d in shape]

    if charge:
        charge = rng.integers(low=1, high=N)

    return sr.utils.get_rand(
        symmetry,
        shape=shape,
        subsizes="equal",
        charge=charge,
        seed=seed,
        **kwargs,
    )


class TestConjProject:
    def test_matches_conj(self):
        x = sr.utils.get_rand(
            "Z2",
            (2, 4, 6),
            flat=True,
            subsizes="equal",
            seed=42,
        )
        expected = x.conj()
        actual = x.conj_project(axis=1)
        actual.test_allclose(expected)

    def test_inplace(self):
        x = sr.utils.get_rand(
            "Z2",
            (2, 4, 6),
            flat=True,
            subsizes="equal",
            seed=42,
        )
        expected = x.conj()
        actual = x.conj_project(axis=1, inplace=True)
        assert actual is x
        actual.test_allclose(expected)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize(
    "shape,perm",
    [
        ([2], [0]),
        ([4, 2], [0, 1]),
        ([2, 4], [1, 0]),
        ([2, 4, 2, 4], (2, 1, 3, 0)),
        ([2, 2, 2, 2, 2, 2, 2], (2, 0, 1, 3, 5, 4, 6)),
    ],
)
@pytest.mark.parametrize("charge", [0, 1])
def test_transpose_roundtrip(symmetry, shape, perm, charge):
    sx = get_zn_blocksparse_flat_compat(symmetry, shape, charge, seed=42)
    sy = sx.transpose(perm)
    perm_inv = sorted(range(len(perm)), key=lambda i: perm[i])

    fx = sx.to_flat()
    fx.check()
    assert fx.to_blocksparse().allclose(sx)
    fy = fx.transpose(perm)
    fy.check()
    assert fy.to_blocksparse().allclose(sy)
    fz = fy.transpose(perm_inv)
    fz.check()
    assert fz.to_blocksparse().allclose(sx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
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
    sx = get_zn_blocksparse_flat_compat(symmetry, shape, charge, seed=42)
    sy = sx.fuse(*axes_groups)
    fx = sx.to_flat()
    fx.check()
    assert fx.to_blocksparse().allclose(sx)
    fy = fx.fuse(*axes_groups)
    fy.check()
    assert fy.to_blocksparse().allclose(sy)
    xu = fy.unfuse_all()
    xu.check()
    # fuse + unfuse is identity up to permutation of axes
    axes_grouped = [i for g in axes_groups for i in g]
    axes_rem = [i for i in range(fx.ndim) if i not in axes_grouped]
    ax_g0 = min(axes_grouped)
    new_axes = axes_rem[:ax_g0] + axes_grouped + axes_rem[ax_g0:]
    sxt = sx.transpose(new_axes)
    fxus = xu.to_blocksparse()
    assert fxus.allclose(sxt)


@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize(
    "symmetry,shape,newshape",
    [
        ["Z2", (2, 2, 2, 2), (4, 2, 2)],
        ["Z2", (2, 2, 2, 2), (2, 4, 2)],
        ["Z2", (2, 2, 2, 2), (2, 2, 4)],
        ["Z2", (2, 2, 2, 2), (4, 4)],
        ["Z3", (3, 3, 3, 3), (9, 3, 3)],
        ["Z3", (3, 3, 3, 3), (3, 9, 3)],
        ["Z3", (3, 3, 3, 3), (3, 3, 9)],
        ["Z3", (3, 3, 3, 3), (9, 9)],
        ["Z4", (4, 4, 4, 4), (16, 4, 4)],
        ["Z4", (4, 4, 4, 4), (4, 16, 4)],
        ["Z4", (4, 4, 4, 4), (4, 4, 16)],
        ["Z4", (4, 4, 4, 4), (16, 16)],
        ["Z4", (4, 4, 4, 4, 4), (64, 4, 4)],
        ["Z4", (4, 4, 4, 4, 4), (4, 64, 4)],
        ["Z4", (4, 4, 4, 4, 4), (4, 4, 64)],
    ],
)
def test_reshape_roundtrip(symmetry, charge, shape, newshape):
    sx = get_zn_blocksparse_flat_compat(
        symmetry,
        shape,
        charge,
        seed=42,
        shape_relative_to_z2=False,
    )
    fx = sx.to_flat()
    sy = sx.reshape(newshape)
    fy = fx.reshape(newshape)
    fy.check()
    assert fy.to_blocksparse().allclose(sy)
    fz = fy.reshape(shape)
    fz.check()
    assert fz.to_blocksparse().allclose(sx)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("charge_x", [0, 1])
@pytest.mark.parametrize("charge_y", [0, 1])
def test_matmul(symmetry, seed, charge_x, charge_y):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)
    dc = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)
    c_ind = sr.utils.rand_index(symmetry, N * dc, subsizes="equal", seed=rng)

    if charge_x:
        charge_x = rng.integers(1, N)
    if charge_y:
        charge_y = rng.integers(1, N)

    sx = sr.utils.get_rand(
        symmetry, (a_ind, b_ind.conj()), charge=charge_x, seed=rng
    )
    sy = sr.utils.get_rand(symmetry, (b_ind, c_ind), charge=charge_y, seed=rng)
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("charge_x", [0, 1])
@pytest.mark.parametrize("charge_y", [0, 1])
def test_matvec(symmetry, seed, charge_x, charge_y):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)

    if charge_x:
        charge_x = rng.integers(1, N)
    if charge_y:
        charge_y = rng.integers(1, N)

    sx = sr.utils.get_rand(
        symmetry, (a_ind, b_ind.conj()), charge=charge_x, seed=rng
    )
    sy = sr.utils.get_rand(symmetry, (b_ind,), charge=charge_y, seed=rng)
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4", "Z5"])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("charge_x", [0, 1])
@pytest.mark.parametrize("charge_y", [0, 1])
def test_vecmat(symmetry, seed, charge_x, charge_y):
    N = int(symmetry[1:])
    rng = sr.utils.get_rng(seed)

    da = rng.integers(1, 5)
    db = rng.integers(1, 5)

    a_ind = sr.utils.rand_index(symmetry, N * da, subsizes="equal", seed=rng)
    b_ind = sr.utils.rand_index(symmetry, N * db, subsizes="equal", seed=rng)

    if charge_x:
        charge_x = rng.integers(1, N)
    if charge_y:
        charge_y = rng.integers(1, N)

    sx = sr.utils.get_rand(symmetry, (a_ind,), charge=charge_x, seed=rng)
    sy = sr.utils.get_rand(
        symmetry, (a_ind.conj(), b_ind), charge=charge_y, seed=rng
    )
    sz = sx @ sy
    fx = sx.to_flat()
    fx.check()
    fy = sy.to_flat()
    fy.check()
    fz = fx @ fy
    fz.check()
    fz.to_blocksparse().allclose(sz)


class TestSortingColumnsAreRedundant:
    """Charge conservation fixes some of the columns a sector sort is keyed
    on, and the contraction and fusion routines drop them. Sorting without
    them must give exactly the same order.
    """

    @pytest.mark.parametrize("order", [2, 3, 4, 5])
    @pytest.mark.parametrize("ndim", [2, 3, 4])
    def test_tensordot_columns(self, order, ndim):
        import itertools

        import numpy as np

        from symmray.flat.flat_array_common import (
            build_cyclic_keys_conserve,
            lexsort_sectors,
            zn_combine,
        )

        for duals in itertools.product([False, True], repeat=ndim):
            sectors = build_cyclic_keys_conserve(
                ndim, order=order, duals=duals, flat=True
            )
            for ncon in range(1, ndim):
                for axes_con in itertools.permutations(range(ndim), ncon):
                    axes_keep = tuple(
                        ax for ax in range(ndim) if ax not in axes_con
                    )
                    if not axes_keep:
                        continue
                    d0 = duals[axes_con[0]]
                    ccon = zn_combine(
                        order,
                        sectors[:, axes_con],
                        duals=[duals[ax] != d0 for ax in axes_con],
                    )
                    full = (
                        ccon,
                        *(sectors[:, ax] for ax in axes_keep),
                        *(sectors[:, ax] for ax in axes_con),
                    )
                    # the last kept and contracted charges are fixed
                    dropped = (
                        ccon,
                        *(sectors[:, ax] for ax in axes_keep[:-1]),
                        *(sectors[:, ax] for ax in axes_con[:-1]),
                    )
                    assert np.array_equal(
                        lexsort_sectors(full, order),
                        lexsort_sectors(dropped, order),
                    )

    @pytest.mark.parametrize("order", [2, 3, 4])
    @pytest.mark.parametrize("ndim", [3, 4])
    def test_fuse_columns(self, order, ndim):
        import itertools

        import numpy as np

        from symmray.flat.flat_array_common import (
            build_cyclic_keys_conserve,
            calc_fuse_group_info,
            lexsort_sectors,
            zn_combine,
        )

        pairs = list(itertools.combinations(range(ndim), 2))
        groupings = [(g,) for g in pairs]
        groupings += [
            (g1, g2)
            for g1, g2 in itertools.combinations(pairs, 2)
            if not set(g1) & set(g2)
        ]

        for duals in itertools.product([False, True], repeat=ndim):
            sectors = build_cyclic_keys_conserve(
                ndim, order=order, duals=duals, flat=True
            )
            for axes_groups in groupings:
                (
                    num_groups,
                    _,
                    _,
                    _,
                    _,
                    axes_before,
                    axes_after,
                    _,
                    group_duals,
                    _,
                ) = calc_fuse_group_info(axes_groups, duals)

                fused = [
                    zn_combine(
                        order,
                        sectors[:, axs],
                        [duals[ax] != dg for ax in axs],
                    )
                    for axs, dg in zip(axes_groups, group_duals)
                ]
                axes_unfused = (*axes_before, *axes_after)
                full = (
                    *fused,
                    *(sectors[:, ax] for ax in axes_unfused),
                    *(sectors[:, ax] for group in axes_groups for ax in group),
                )
                # one charge per group is fixed, plus one overall, which
                # falls on the last fused charge if no axis is unfused
                nkept = num_groups - (0 if axes_unfused else 1)
                dropped = (
                    *fused[:nkept],
                    *(sectors[:, ax] for ax in axes_unfused[:-1]),
                    *(
                        sectors[:, ax]
                        for group in axes_groups
                        for ax in group[:-1]
                    ),
                )
                assert np.array_equal(
                    lexsort_sectors(full, order),
                    lexsort_sectors(dropped, order),
                )


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_build_cyclic_keys_conserve(ndim, order, seed):
    import numpy as np

    from symmray.flat.flat_array_common import (
        build_cyclic_keys_conserve,
        lexsort_sectors,
        zn_combine,
    )

    rng = random.Random(seed)
    charge = rng.randint(0, order - 1)
    duals = [rng.choice([True, False]) for _ in range(ndim)]

    sectors = build_cyclic_keys_conserve(
        ndim,
        order=order,
        charge=charge,
        duals=duals,
        flat=True,
    )
    scharges = zn_combine(order, sectors, duals=duals)

    assert set(map(int, scharges)) == {charge}
    assert np.all(lexsort_sectors(sectors) == np.arange(order ** (ndim - 1)))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("order", [2, 3, 4])
@pytest.mark.parametrize("seed", range(5))
def test_build_cyclic_keys_by_charge(ndim, order, seed):
    import numpy as np

    from symmray.flat.flat_array_common import (
        build_cyclic_keys_by_charge,
        lexsort_sectors,
        zn_combine,
    )

    rng = random.Random(seed)
    duals = [rng.choice([True, False]) for _ in range(ndim)]

    sectors = build_cyclic_keys_by_charge(
        ndim,
        order=order,
        duals=duals,
    )
    scharges = zn_combine(order, sectors, duals=duals)

    for i in range(order):
        # all have matching charge
        assert np.all(scharges[i] == i)
        # and are sorted within that charge
        assert np.all(
            lexsort_sectors(sectors[i]) == np.arange(order ** (ndim - 1))
        )


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("seed", range(50))
def test_tensordot(symmetry, seed):
    from symmray.utils_test import rand_valid_tensordot

    N = int(symmetry[1:])

    a, b, axes = rand_valid_tensordot(
        symmetry,
        seed=seed,
        dimension_multiplier=N,
        subsizes="equal",
    )
    c = sr.tensordot(a, b, axes, preserve_array=True)

    fa = a.to_flat()
    fb = b.to_flat()

    fc = sr.tensordot(
        fa,
        fb,
        axes,
        preserve_array=True,
    )
    fc.check()

    if c.is_zero() and fc.is_zero():
        # both are zero, other tests might break
        return

    assert (
        len(
            set(map(float, fc.blocks.round(9).flat))
            - set(map(float, c.to_flat().blocks.round(9).flat))
        )
        == 0
    )

    fc.to_blocksparse().test_allclose(c)


@pytest.mark.parametrize("symm", ["Z2", "Z4"])
def test_tensordot_fused_with_already_fused_arrays(symm):
    a, b, c, d = (
        sr.utils.rand_index(symm, d, subsizes="equal") for d in [4, 4, 4, 8]
    )
    x = sr.utils.get_rand(symm, (a, b, c), flat=True)
    y = sr.utils.get_rand(symm, (c.conj(), d), flat=True)
    z = sr.tensordot(x, y, axes=[(2,), (0,)]).fuse((0, 1))
    xf = x.fuse((0, 1))
    zf = sr.tensordot(xf, y, axes=[(1,), (0,)])
    zf.test_allclose(z)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_block_multiply_diagonal(symmetry, charge, axis):
    import autoray as ar
    import numpy as np

    rng = np.random.default_rng(42)
    x = get_zn_blocksparse_flat_compat(
        symmetry,
        (4, 2, 6, 2),
        seed=rng,
        charge=charge,
    ).to_flat()

    v = sr.FlatVector.rand(symmetry, x.indices[axis].charge_size)
    y = ar.do("multiply_diagonal", x, v, axis=axis)

    # check dense reference
    xd = x.to_dense()
    vd = v.to_dense()
    yd = y.to_dense()
    lhs = "abcd"
    rhs = lhs[axis]
    np.testing.assert_allclose(yd, np.einsum(f"{lhs},{rhs}->{lhs}", xd, vd))


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", range(10))
def test_abelian_array_slice(symmetry, ndim, seed):
    import numpy as np

    rng = sr.utils.get_rng(seed)
    sx = get_zn_blocksparse_flat_compat(
        symmetry=symmetry,
        shape=tuple(2 * rng.integers(1, 5, size=ndim)),
        seed=rng,
    )
    dx = sx.to_dense()
    x = sx.to_flat()

    for ax in range(ndim):
        for d in range(x.shape[ax]):
            # get [:, :, ..., 2, ..., :] etc.
            selector = (
                (slice(None),) * ax + (d,) + (slice(None),) * (ndim - ax - 1)
            )
            x_slc = x[selector]
            x_slc.to_blocksparse().test_allclose(sx[selector])
            dx_slc = dx[selector]

            if ndim == 2:
                # dropped zero blocks will be fully missing
                dx_slc = dx_slc[dx_slc != 0]

            np.testing.assert_allclose(x_slc.to_dense(), dx_slc)


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
def test_einsum_multi_term(symmetry):
    N = int(symmetry[1:])
    a = sr.utils.rand_index(symmetry, 2 * N, subsizes="equal", dual=False)
    b = sr.utils.rand_index(symmetry, 3 * N, subsizes="equal", dual=False)
    c = sr.utils.rand_index(symmetry, 3 * N, subsizes="equal", dual=False)
    d = sr.utils.rand_index(symmetry, 4 * N, subsizes="equal", dual=False)

    rng = sr.utils.get_rng(42)
    x = sr.utils.get_rand(
        symmetry, (a, b), subsizes="equal", seed=rng
    ).to_flat()
    y = sr.utils.get_rand(
        symmetry, (b.conj(), c), subsizes="equal", seed=rng
    ).to_flat()
    z = sr.utils.get_rand(
        symmetry, (c.conj(), d), subsizes="equal", seed=rng
    ).to_flat()

    result = sr.einsum("ab,bc,cd->ad", x, y, z)
    expected = sr.tensordot(sr.tensordot(x, y, axes=1), z, axes=1)
    result.test_allclose(expected)


@pytest.mark.parametrize("symmetry", ["Z2", "Z4"])
@pytest.mark.parametrize("shape", [(4, 8, 12), (8,)])
@pytest.mark.parametrize("charge", [0, 1])
def test_to_pytree_and_back(symmetry, shape, charge):
    x = sr.utils.get_rand(
        symmetry=symmetry,
        shape=shape,
        subsizes="equal",
        charge=charge,
        label="x",
        flat=True,
    )
    tree = x.to_pytree()
    y = type(x).from_pytree(tree)
    x.test_allclose(y)

    if len(shape) > 1:
        # test with subinfo
        xf = x.fuse(tuple(range(x.ndim)))
        tree = xf.to_pytree()
        yf = type(xf).from_pytree(tree)
        xf.test_allclose(yf)
        yf.unfuse_all().test_allclose(x)


def _get_fused_subinfo(x):
    # fuse only two of three axes, so the fused index is not locked to the
    # total charge and retains all fused charge sectors
    subinfo = x.fuse((1, 2)).indices[1].subinfo
    assert subinfo.ncharge > 1
    return subinfo


def _get_rand_3d(symmetry="Z2"):
    return sr.utils.get_rand(
        symmetry=symmetry,
        shape=(4, 4, 6),
        subsizes="equal",
        flat=True,
        seed=42,
    )


def test_subinfo_select_charge_python_int():
    import numpy as np

    subinfo = _get_fused_subinfo(_get_rand_3d())
    new = subinfo.select_charge(1)
    assert new.ncharge == 1
    np.testing.assert_array_equal(
        np.asarray(new.subkeys[0]), np.asarray(subinfo.subkeys[1])
    )


@pytest.mark.parametrize("backend", ("numpy", "jax", "torch"))
def test_subinfo_select_charge_backend_charge(
    backend,
    require_backend,
    convert_backend,
):
    import numpy as np

    require_backend(backend)

    x = _get_rand_3d()
    expected = np.asarray(_get_fused_subinfo(x).select_charge(1).subkeys)

    # NOTE: convert before fusing, so the subkeys are backend arrays too
    subinfo = _get_fused_subinfo(x.to(backend))

    # plain python int charge
    got = subinfo.select_charge(1).subkeys
    np.testing.assert_array_equal(np.asarray(got), expected)

    # 0-dim backend scalar charge, as passed by e.g. align_axes
    charge = convert_backend(np.asarray(1), backend)
    got = subinfo.select_charge(charge).subkeys
    np.testing.assert_array_equal(np.asarray(got), expected)

    if backend == "jax":
        import jax

        # under jit the charge is a tracer: a tuple index (charge,)
        # raises here
        got = jax.jit(lambda c: subinfo.select_charge(c).subkeys)(1)
        np.testing.assert_array_equal(np.asarray(got), expected)
