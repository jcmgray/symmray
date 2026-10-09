import numpy as np
import pytest

import symmray as sr

all_symmetries = ("Z2", "Z3", "U1", "Z2Z2", "U1U1")


def assert_matches(xp, xs):
    """Check packed array ``xp`` against block sparse array ``xs``."""
    assert isinstance(xp, sr.AbelianArrayPacked)
    xp.check()
    if xs.num_blocks == 0:
        # sparse drops every charge of an empty result, packed keeps them
        assert xp.num_blocks == 0
        return
    xp.to_blocksparse().test_allclose(xs)
    np.testing.assert_allclose(xp.to_dense(), xs.to_dense())


def get_rand(symmetry, shape, seed, **kwargs):
    x = sr.utils.get_rand(symmetry, shape, seed=seed, **kwargs)
    return x, x.to_packed()


class TestIndex:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_blockindex_roundtrip(self, symmetry):
        x = sr.utils.get_rand(symmetry, (3, 4, 5), seed=0)
        ix = x.fuse((0, 2), (1,)).indices[0]
        pix = ix.to_packed()
        pix.check()
        assert pix.subinfo.charge_starts[-1] == len(pix.subinfo.subkeys)
        assert pix.chargemap == ix.chargemap
        assert pix.to_blockindex().subinfo.extents == ix.subinfo.extents

    def test_select_charge_ids(self):
        PackedIndex = sr.packed.packed_index.PackedIndex
        ix = PackedIndex(charges=(-1, 0, 2), sizes=(2, 3, 4), dual=True)
        new = ix.select_charge_ids([0, 2])
        assert new.chargemap == {-1: 2, 2: 4}
        assert new.dual


class TestConversion:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_roundtrip(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6), seed=1)
        assert_matches(xp, x)
        assert xp.shape == x.shape
        assert xp.num_blocks == x.num_blocks
        assert set(xp.sectors) == set(x.sectors)

    def test_get_rand_layout(self):
        x = sr.utils.get_rand("U1", (4, 5), seed=2, layout="packed")
        assert isinstance(x, sr.U1ArrayPacked)
        assert sr.utils.get_array_cls("U1", layout="packed") is (
            sr.U1ArrayPacked
        )
        with pytest.raises(ValueError):
            sr.utils.get_array_cls("U1", flat=True, layout="packed")


class TestBuilders:
    def test_local_operators(self):
        x = sr.spin_local_operators.spin_operator_local_array(
            "Z2", layout="packed"
        )
        assert isinstance(x, sr.Z2ArrayPacked)
        ham = sr.hamiltonians.ham_fermi_hubbard_from_edges(
            "U1", [(0, 1), (1, 2)], layout="packed"
        )
        assert all(
            isinstance(h, sr.U1FermionicArrayPacked) for h in ham.values()
        )

    def test_networks(self):
        pytest.importorskip("quimb")
        tn = sr.TN_fermionic_from_edges_rand(
            "U1", [(0, 1), (1, 2)], 3, phys_dim=4, seed=0, layout="packed"
        )
        assert all(isinstance(t.data, sr.U1FermionicArrayPacked) for t in tn)


class TestStructural:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize("perm", ((2, 0, 3, 1), (0, 1, 3, 2), None))
    def test_transpose(self, symmetry, perm, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6, 3), seed=3)
        assert_matches(xp.transpose(perm), x.transpose(perm))

    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize(
        "groups",
        (((0, 2), (1, 3)), ((3, 1),), ((1, 2, 3),), ((2,), (0, 1))),
    )
    def test_fuse_unfuse(self, symmetry, groups, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6, 3), seed=4)
        f = x.fuse(*groups)
        fp = xp.fuse(*groups)
        assert_matches(fp, f)
        assert_matches(fp.unfuse_all(), f.unfuse_all())

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_fuse_empty_group(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5), seed=5)
        assert_matches(xp.fuse((0,), (), (1,)), x.fuse((0,), (), (1,)))
        assert_matches(
            xp.fuse((0,), (), (1,)).unfuse_all(),
            x.fuse((0,), (), (1,)).unfuse_all(),
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_reshape(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6, 3), seed=6)
        assert_matches(xp.reshape((20, 18)), x.reshape((20, 18)))

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_expand_dims_squeeze(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5, 3), seed=7)
        assert_matches(xp.expand_dims(1), x.expand_dims(1))
        assert_matches(xp.expand_dims(1).squeeze(1), x)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_conj(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5, 3), seed=8, dtype="complex128")
        assert_matches(xp.conj(), x.conj())


class TestTensordot:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize(
        "axes", (((2, 0), (0, 1)), ((2,), (0,)), ((0,), (1,)))
    )
    @pytest.mark.parametrize("seed", range(3))
    def test_matches_sparse(self, symmetry, axes, seed, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6, 3), seed=seed)
        y, yp = get_rand(
            symmetry,
            [x.indices[2].conj(), x.indices[0].conj(), 7],
            seed=seed + 10,
        )
        assert_matches(xp.tensordot(yp, axes), x.tensordot(y, axes))

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_full_contraction(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6), seed=11)
        axes = ((0, 1, 2), (0, 1, 2))
        expected = x.tensordot(x.conj(), axes)
        assert xp.tensordot(xp.conj(), axes) == pytest.approx(expected)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_outer(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5), seed=12)
        v, vp = get_rand(symmetry, (3, 2), seed=13)
        assert_matches(xp.tensordot(vp, 0), x.tensordot(v, 0))

    @pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
    def test_misaligned_sectors(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (6, 7), seed=14, subsizes="maximal")
        y = sr.utils.get_rand(
            symmetry, [x.indices[1].conj(), 5], seed=15, subsizes="maximal"
        )
        # drop some blocks so only part of each array is aligned
        for sector in y.sectors[::2]:
            y.del_block(sector)
        assert_matches(xp @ y.to_packed(), x @ y)
        assert_matches(xp.tensordot(y.to_packed(), 1), x.tensordot(y, 1))

    @pytest.mark.parametrize("symmetry", ("U1", "U1U1"))
    def test_align_axes(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (6, 7, 5), seed=30, subsizes="maximal")
        y = sr.utils.get_rand(
            symmetry,
            [x.indices[1].conj(), 4, x.indices[2].conj()],
            seed=31,
            subsizes="maximal",
        )
        for sector in y.sectors[::2]:
            y.del_block(sector)
        axes = ((1, 2), (0, 2))
        a, b = x.align_axes(y, axes)
        ap, bp = xp.align_axes(y.to_packed(), axes)
        assert_matches(ap, a)
        assert_matches(bp, b)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_already_fused(self, symmetry, kernels):
        x, _ = get_rand(symmetry, (4, 5, 6, 3), seed=16)
        xf = x.fuse((0, 1), (2, 3))
        y = xf.conj().transpose()
        assert_matches(xf.to_packed() @ y.to_packed(), xf @ y)


class TestMultiplyDiagonal:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize("power", (1, -1))
    def test_matches_sparse(self, symmetry, power, kernels):
        x = sr.utils.get_rand(symmetry, (4, 5, 6), seed=17)
        u, s, _ = x.fuse((0, 1), (2,)).svd()
        up = u.to_packed()
        sp = s.to_packed()
        assert_matches(
            up.multiply_diagonal(sp, 1, power=power),
            u.multiply_diagonal(s, 1, power=power),
        )

    def test_missing_vector_blocks_drop(self, kernels):
        x = sr.utils.get_rand("U1", (6, 6), seed=18, subsizes="maximal")
        charges = list(x.indices[1].charges)
        v = sr.BlockVector({c: np.ones(1) for c in charges[1:]})
        assert_matches(
            x.to_packed().multiply_diagonal(v.to_packed(), 1),
            x.multiply_diagonal(v, 1),
        )
        with pytest.raises(ZeroDivisionError):
            x.to_packed().multiply_diagonal(v.to_packed(), 1, power=-1)


class TestElementwise:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_arithmetic(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (4, 5, 6), seed=19)
        assert_matches(xp + xp, x + x)
        assert_matches(xp - 2 * xp, x - 2 * x)
        assert_matches(-xp / 3, -x / 3)
        assert xp.norm() == pytest.approx(x.norm())
        assert xp.sum() == pytest.approx(x.sum())

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_expm(self, symmetry, kernels):
        y = sr.utils.get_rand(symmetry, (4, 5), seed=24)
        # square blocks with zero charge
        x = y.tensordot(y.conj(), ((1,), (1,)))
        assert_matches(
            sr.scipy.linalg.expm(x.to_packed()), sr.scipy.linalg.expm(x)
        )

    def test_mismatched_structure_add(self, kernels):
        x, xp = get_rand("U1", (4, 5), seed=20)
        y = x.copy()
        y.del_block(y.sectors[0])
        assert_matches(xp + y.to_packed(), x + y)


class TestLinalg:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_svd_reconstruct(self, symmetry, kernels):
        x, _ = get_rand(symmetry, (4, 5, 6), seed=21)
        xf = x.fuse((0, 1), (2,))
        u, s, vh = xf.svd()
        up, sp, vhp = xf.to_packed().svd()
        assert isinstance(sp, sr.PackedVector)
        sp.test_allclose(s)
        assert_matches(
            up.multiply_diagonal(sp, 1) @ vhp, u.multiply_diagonal(s, 1) @ vh
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_qr(self, symmetry, kernels):
        x, xp = get_rand(symmetry, (6, 5), seed=22)
        q, r = sr.linalg.qr(x)
        qp, rp = sr.linalg.qr(xp)
        assert_matches(qp, q)
        assert_matches(rp, r)
        assert_matches(qp @ rp, q @ r)

    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
    def test_eigh(self, symmetry, kernels):
        from symmray.packed.packed_array_common import FALLBACK_COUNTS

        x = sr.utils_test.rand_herm(symmetry, 7, seed=24)
        w, u = sr.linalg.eigh(x)
        before = sum(FALLBACK_COUNTS.values())
        wp, up = sr.linalg.eigh(x.to_packed())
        assert sum(FALLBACK_COUNTS.values()) == before
        assert isinstance(wp, sr.PackedVector)
        wp.test_allclose(w)
        assert_matches(up, u)
        assert_matches(up.multiply_diagonal(wp, 1) @ up.H, x)

    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
    @pytest.mark.parametrize("upper", (False, True))
    def test_cholesky(self, symmetry, upper, kernels):
        from symmray.packed.packed_array_common import FALLBACK_COUNTS

        x = sr.utils_test.rand_posdef(symmetry, 7, seed=30)
        before = sum(FALLBACK_COUNTS.values())
        result = sr.linalg.cholesky(x.to_packed(), upper=upper)
        assert sum(FALLBACK_COUNTS.values()) == before
        assert_matches(result, sr.linalg.cholesky(x, upper=upper))

    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
    @pytest.mark.parametrize("b_ndim", (1, 2))
    @pytest.mark.parametrize("odd", (False, True))
    def test_solve(self, symmetry, b_ndim, odd, kernels):
        from symmray.packed.packed_array_common import FALLBACK_COUNTS

        odd_charges = {"Z2": 1, "U1": 1, "Z2Z2": (1, 0), "U1U1": (1, 0)}
        charge = odd_charges[symmetry] if odd else None
        ind = sr.utils.rand_index(symmetry, 7, seed=31)
        a = sr.utils.get_rand(
            symmetry, (ind, ind.conj()), seed=32, dtype="complex128"
        )
        shape = (ind,) if b_ndim == 1 else (ind, 5)
        b = sr.utils.get_rand(
            symmetry, shape, seed=33, dtype="complex128", charge=charge
        )
        before = sum(FALLBACK_COUNTS.values())
        result = sr.linalg.solve(a.to_packed(), b.to_packed())
        assert sum(FALLBACK_COUNTS.values()) == before
        assert_matches(result, sr.linalg.solve(a, b))

    def test_svd_keeps_fused_index(self, kernels):
        x, xp = get_rand("U1", (4, 5, 6), seed=23)
        xfp = xp.fuse((0, 1), (2,))
        u, _, _ = xfp.svd()
        assert u.indices[0] is xfp.indices[0]
        assert_matches(u.unfuse(0), x.fuse((0, 1), (2,)).svd()[0].unfuse(0))


class TestSplit:
    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
    @pytest.mark.parametrize("method", ("svd", "qr", "lq", "svd:eig"))
    @pytest.mark.parametrize("absorb", (-1, 0, 1, None))
    @pytest.mark.parametrize("charge_side", ("auto", "left", "right"))
    def test_static(self, symmetry, method, absorb, charge_side, kernels):
        if method in ("qr", "lq") and absorb not in (-1, 1):
            pytest.skip("qr and lq only absorb to one side")
        if method == "qr" and absorb == -1:
            pytest.skip("not a qr")
        if method == "lq" and absorb == 1:
            pytest.skip("not an lq")
        x = sr.utils.get_rand(symmetry, (6, 7), seed=25, charge=None)
        opts = {"method": method, "absorb": absorb}
        expected = x._split(charge_side=charge_side, **opts)
        result = x.to_packed()._split(charge_side=charge_side, **opts)
        assert_split_matches(result, expected)

    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
    @pytest.mark.parametrize(
        "cutoff_mode", ("abs", "rel", "sum2", "rsum2", "sum1", "rsum1")
    )
    @pytest.mark.parametrize("max_bond", (-1, 5))
    @pytest.mark.parametrize("absorb", (-1, 0, 1, None))
    def test_dynamic(self, symmetry, cutoff_mode, max_bond, absorb, kernels):
        x = sr.utils.get_rand(symmetry, (6, 7, 8), seed=26).fuse((0, 1), (2,))
        opts = {
            "method": "svd",
            "absorb": absorb,
            "cutoff": 0.1,
            "cutoff_mode": cutoff_mode,
            "max_bond": max_bond,
        }
        assert_split_matches(x.to_packed()._split(**opts), x._split(**opts))

    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
    @pytest.mark.parametrize("max_bond", (1, 3, 9))
    @pytest.mark.parametrize("cutoff", (0.0, 0.2))
    def test_eager(self, symmetry, max_bond, cutoff, kernels):
        x = sr.utils.get_rand(symmetry, (6, 7, 8), seed=27).fuse((0, 1), (2,))
        opts = {
            "method": "svd",
            "absorb": 0,
            "cutoff": cutoff,
            "max_bond": max_bond,
            "max_bond_mode": "eager",
        }
        assert_split_matches(x.to_packed()._split(**opts), x._split(**opts))

    @pytest.mark.parametrize("symmetry", ("Z2", "U1"))
    def test_eigh_positive(self, symmetry, kernels):
        y = sr.utils.get_rand(symmetry, (5, 6), seed=28)
        x = y.tensordot(y.conj(), ((1,), (1,)))
        opts = {"method": "eigh", "absorb": None, "cutoff": 1e-3}
        assert_split_matches(x.to_packed()._split(**opts), x._split(**opts))

    def test_native(self, kernels):
        from symmray.packed.packed_array_common import FALLBACK_COUNTS

        x = sr.utils.get_rand("U1", (6, 7), seed=29, layout="packed")
        before = FALLBACK_COUNTS["_split_abelian"]
        x.svd_truncated(max_bond=3)
        assert FALLBACK_COUNTS["_split_abelian"] == before


def assert_split_matches(result, expected):
    for a, b in zip(result, expected):
        if b is None:
            assert a is None
        elif isinstance(b, sr.BlockVector):
            assert isinstance(a, sr.PackedVector)
            a.test_allclose(b)
        else:
            assert_matches(a, b)


class TestValidation:
    def test_plan_cache_distinguishes_symmetry(self, kernels):
        # identical structure, but Z2 and U1 fuse charges differently
        blocks = {(0, 0, 0, 0): np.ones((1, 1, 1, 1))}
        blocks[1, 1, 1, 1] = 2 * np.ones((1, 1, 1, 1))
        duals = (False, False, True, True)
        for symmetry in ("Z2", "U1"):
            x = sr.utils.get_array_cls(symmetry).from_blocks(blocks, duals)
            expected = x.tensordot(x.conj(), ((2, 3), (2, 3)))
            got = x.to_packed().tensordot(
                x.conj().to_packed(), ((2, 3), (2, 3))
            )
            assert_matches(got, expected)

    def test_multiply_diagonal_wrong_size_raises(self, kernels):
        x = sr.U1Array.from_blocks(
            {(0, 0): np.arange(4.0).reshape(2, 2)}, (False, True)
        ).to_packed()
        with pytest.raises(ValueError):
            x.multiply_diagonal(sr.PackedVector({0: np.array([2.0])}), 1)
        with pytest.raises(ValueError):
            x.multiply_diagonal(sr.PackedVector({0: np.ones(2)}), 2)

    def test_tensordot_incompatible_raises(self, kernels):
        a = sr.U1Array.from_blocks({(0, 0): np.ones((2, 2))}, (False, True))
        b = sr.U1Array.from_blocks({(0, 0): np.ones((3, 1))}, (False, True))
        with pytest.raises(ValueError):
            a.to_packed() @ b.to_packed()
        c = sr.U1Array.from_blocks({(0, 0): np.ones((2, 1))}, (True, False))
        with pytest.raises(ValueError):
            a.to_packed() @ c.to_packed()

    def test_scalar_arrays(self, kernels):
        x, xp = get_rand("U1", (4, 5), seed=32)
        full = x.tensordot(x.conj(), 2, preserve_array=True)
        fullp = xp.tensordot(xp.conj(), 2, preserve_array=True)
        assert fullp.to_blocksparse().to_packed().item() == pytest.approx(
            full.item()
        )
        empty = sr.U1ArrayPacked(sectors=np.zeros((0, 0)), data=[], indices=())
        assert empty.num_blocks == 0
        got = xp.isel(0, 0).isel(0, 0)
        expected = x.isel(0, 0).isel(0, 0)
        assert got.ndim == 0
        assert got.get_scalar_element() == pytest.approx(
            expected.get_scalar_element()
        )

    def test_integer_dtypes(self, kernels):
        x = sr.U1Array.from_blocks(
            {(0, 0): np.arange(4).reshape(2, 2)}, (False, True)
        )
        v = sr.BlockVector({0: np.array([2, 2])})
        got = x.to_packed().multiply_diagonal(v.to_packed(), 1, power=-1)
        np.testing.assert_allclose(got.to_dense(), [[0, 0.5], [1, 1.5]])
        np.testing.assert_allclose(
            (x.to_packed() @ x.to_packed()).to_dense(), (x @ x).to_dense()
        )


class TestPackedVector:
    def test_arithmetic_different_block_sizes(self):
        a = sr.BlockVector({0: np.array([1.0, 2.0]), 1: np.array([3.0])})
        b = sr.BlockVector({0: np.array([10.0]), 1: np.array([20.0, 30.0])})
        (a.to_packed() + b.to_packed()).test_allclose(a + b)

    def test_arithmetic_same_structure(self):
        a = sr.BlockVector({0: np.array([1.0, 2.0]), 1: np.array([3.0])})
        b = sr.BlockVector({0: np.array([4.0, 5.0]), 1: np.array([6.0])})
        (a.to_packed() * b.to_packed()).test_allclose(a * b)


class TestKeyOverflow:
    def test_numpy_align_many_columns(self):
        from symmray.packed.kernels import load_kernels

        k = load_kernels("numpy")
        radices = np.full(63, 2, dtype=np.int64)
        sub_a = np.array([[0] * 63, [1] * 63], dtype=np.int64)
        sub_b = np.array([[1] * 63, [0] * 62 + [1]], dtype=np.int64)
        keep_a, keep_b = k.tensordot_align(sub_a, sub_b, radices)
        assert keep_a.tolist() == [False, True]
        assert keep_b.tolist() == [True, False]

    def test_fuse_group_large_charge_values(self):
        pytest.importorskip("numba")
        from symmray.packed.kernels import load_kernels

        # U1U1-like charges far apart on one axis, plus a trivial axis

        big = 2**32
        signed = np.array(
            [[0, 0], [0, big], [big, 0], [big, big], [0, 0]], dtype=np.int64
        )
        args = (
            np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int64),
            np.array([4, 1], dtype=np.int64),
            signed,
            np.array([0, 4, 5], dtype=np.int64),
            0,
            np.ones(5, dtype=np.int64),
        )
        expected = load_kernels("numpy").fuse_group(*args)
        got = load_kernels("numba").fuse_group(*args)
        for e, g in zip(expected, got):
            np.testing.assert_array_equal(e, g)


class TestPlanCache:
    def test_stored_on_second_sight(self, kernels):
        from symmray.packed import packed_array_common as pac

        pac.clear_plan_cache()
        _, xp = get_rand("U1", (4, 5, 6), seed=33)
        xp.transpose((2, 0, 1))
        assert len(pac._plans) == 0
        xp.transpose((2, 0, 1))
        assert len(pac._plans) == 1
        hits = pac._plan_cache_stats["hit"]
        xp.transpose((2, 0, 1))
        assert pac._plan_cache_stats["hit"] == hits + 1
