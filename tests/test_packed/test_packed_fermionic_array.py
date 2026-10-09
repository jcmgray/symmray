import numpy as np
import pytest

import symmray as sr

all_symmetries = ("Z2", "U1", "Z2Z2", "U1U1")


def assert_matches(xp, xs):
    """Check packed fermionic array ``xp`` against block sparse ``xs``."""
    assert isinstance(xp, sr.FermionicArrayPacked)
    xp.check()
    if xs.num_blocks == 0:
        # sparse drops every charge of an empty result, packed keeps them
        assert xp.num_blocks == 0
        return
    assert xp.dummy_modes == xs.dummy_modes
    xp.to_blocksparse().test_allclose(xs)
    np.testing.assert_allclose(xp.to_dense(), xs.to_dense())


def get_rand(symmetry, seed, bra=False, odd=False):
    """Random fermionic array with random lazy phases. Optionally odd, and as a
    bra with dual dummy modes.
    """

    charge = None
    if odd:
        charge = {"Z2": 1, "U1": 1, "Z2Z2": (1, 0), "U1U1": (1, 0)}[symmetry]
    while True:
        x = sr.utils.get_rand(
            symmetry,
            (4, 5, 6, 3),
            seed=seed,
            fermionic=True,
            charge=charge,
            label="x",
        )
        if x.num_blocks:
            break
        # odd charges can leave no valid sectors for some random indices
        seed += 1000
    x.randomize_phases(seed=seed, inplace=True)
    if bra:
        x = x.conj(phase_dual=True)
    return x, x.to_packed()


cases = pytest.mark.parametrize(
    "symmetry, bra, odd",
    [
        (symmetry, bra, odd)
        for symmetry in all_symmetries
        for bra in (False, True)
        for odd in (False, True)
    ],
)


class TestPhases:
    @cases
    def test_roundtrip(self, symmetry, bra, odd, kernels):
        x, xp = get_rand(symmetry, 0, bra, odd)
        assert_matches(xp, x)

    @cases
    def test_phase_ops(self, symmetry, bra, odd, kernels):
        x, xp = get_rand(symmetry, 1, bra, odd)
        assert_matches(xp.phase_flip(1, 3), x.phase_flip(1, 3))
        perm = (3, 2, 1, 0)
        assert_matches(xp.phase_transpose(perm), x.phase_transpose(perm))
        assert_matches(xp.phase_global(), x.phase_global())
        assert_matches(xp.phase_sync(), x.phase_sync())

    @cases
    def test_transpose(self, symmetry, bra, odd, kernels):
        x, xp = get_rand(symmetry, 2, bra, odd)
        perm = (2, 0, 3, 1)
        assert_matches(xp.transpose(perm), x.transpose(perm))
        assert_matches(
            xp.transpose(perm, phase=False), x.transpose(perm, phase=False)
        )

    @cases
    @pytest.mark.parametrize(
        "groups", (((0, 2), (1, 3)), ((3, 1),), ((1, 2, 3),))
    )
    def test_fuse_unfuse(self, symmetry, bra, odd, groups, kernels):
        x, xp = get_rand(symmetry, 3, bra, odd)
        f = x.fuse(*groups)
        fp = xp.fuse(*groups)
        assert_matches(fp, f)
        assert_matches(fp.unfuse_all(), f.unfuse_all())


class TestConjugation:
    @cases
    @pytest.mark.parametrize(
        "kwargs",
        (
            {},
            {"phase_dual": True},
            {"phase_permutation": False},
            {"phase_dual": (0, 2)},
        ),
    )
    def test_conj(self, symmetry, bra, odd, kwargs, kernels):
        x, xp = get_rand(symmetry, 4, bra, odd)
        assert_matches(xp.conj(**kwargs), x.conj(**kwargs))

    @cases
    def test_conj_twice(self, symmetry, bra, odd, kernels):
        _, xp = get_rand(symmetry, 5, bra, odd)
        twice = xp.conj(phase_dual=True).conj(phase_dual=True)
        np.testing.assert_allclose(twice.to_dense(), xp.to_dense())

    @cases
    def test_dagger(self, symmetry, bra, odd, kernels):
        x, xp = get_rand(symmetry, 6, bra, odd)
        assert_matches(xp.dagger(), x.dagger())
        assert_matches(xp.dagger(phase_dual=True), x.dagger(phase_dual=True))

    @cases
    @pytest.mark.parametrize("axes", ((0,), (1,), (2,), (3,), (2, 0)))
    def test_conj_project(self, symmetry, bra, odd, axes, kernels):
        x, xp = get_rand(symmetry, 7, bra, odd)
        assert_matches(xp.conj_project(axes=axes), x.conj_project(axes=axes))


class TestContraction:
    @cases
    def test_norm(self, symmetry, bra, odd, kernels):
        _, xp = get_rand(symmetry, 8, bra, odd)
        axes = ((0, 1, 2, 3),) * 2
        n2 = xp.conj(phase_dual=True).tensordot(xp, axes)
        assert n2 == pytest.approx(xp.norm() ** 2)

    @cases
    @pytest.mark.parametrize(
        "axes", (((2, 0), (2, 0)), ((1,), (1,)), ((0, 3), (3, 0)))
    )
    def test_tensordot(self, symmetry, bra, odd, axes, kernels):
        x, xp = get_rand(symmetry, 9, bra, odd)
        y = sr.utils.get_rand(
            symmetry,
            [x.indices[ax].conj() for ax in axes[0]] + [5],
            seed=19,
            fermionic=True,
            label="y",
        )
        y.randomize_phases(seed=20, inplace=True)
        ax = (axes[0], tuple(range(len(axes[0]))))
        assert_matches(xp.tensordot(y.to_packed(), ax), x.tensordot(y, ax))
        # contracted axes on the right of y
        yr = y.transpose(tuple(range(y.ndim))[::-1])
        ax = (axes[0], tuple(range(y.ndim - 1, 0, -1)))
        assert_matches(xp.tensordot(yr.to_packed(), ax), x.tensordot(yr, ax))

    @cases
    def test_gram(self, symmetry, bra, odd, kernels):
        x, xp = get_rand(symmetry, 10, bra, odd)
        assert_matches(xp.gram(axes=1), x.gram(axes=1))


class TestLinalg:
    @cases
    @pytest.mark.parametrize("absorb", (-1, 0, 1, None))
    def test_svd_truncated(self, symmetry, bra, odd, absorb, kernels):
        x, _ = get_rand(symmetry, 11, bra, odd)
        xf = x.fuse((0, 1), (2, 3))
        opts = {"absorb": absorb, "max_bond": 8, "cutoff": 0.0}
        expected = xf.svd_truncated(**opts)
        result = xf.to_packed().svd_truncated(**opts)
        for a, b in zip(result, expected):
            if b is None:
                assert a is None
            elif isinstance(b, sr.BlockVector):
                a.test_allclose(b)
            else:
                assert_matches(a, b)

    @cases
    def test_multiply_diagonal(self, symmetry, bra, odd, kernels):
        x, _ = get_rand(symmetry, 12, bra, odd)
        u, s, vh = x.fuse((0, 1), (2, 3)).svd()
        up, sp, vhp = (y.to_packed() for y in (u, s, vh))
        assert_matches(up.multiply_diagonal(sp, 1), u.multiply_diagonal(s, 1))
        assert_matches(
            vhp.multiply_diagonal(sp, 0, power=-1),
            vh.multiply_diagonal(s, 0, power=-1),
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize("bra", (False, True))
    @pytest.mark.parametrize("phase_eigenvalues", (False, True))
    def test_eigh(self, symmetry, bra, phase_eigenvalues, kernels):
        x = sr.utils_test.rand_herm(symmetry, 7, seed=14, fermionic=True)
        x.randomize_phases(seed=14, inplace=True)
        if bra:
            x = x.conj(phase_dual=True)
        w, u = x.eigh(phase_eigenvalues=phase_eigenvalues)
        wp, up = x.to_packed().eigh(phase_eigenvalues=phase_eigenvalues)
        wp.test_allclose(w)
        assert_matches(up, u)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize("upper", (False, True))
    def test_cholesky(self, symmetry, upper, kernels):
        x = sr.utils_test.rand_posdef(symmetry, 7, seed=15, fermionic=True)
        assert_matches(
            sr.linalg.cholesky(x.to_packed(), upper=upper),
            sr.linalg.cholesky(x, upper=upper),
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize("bra", (False, True))
    @pytest.mark.parametrize("b_ndim", (1, 2))
    @pytest.mark.parametrize("odd", (False, True))
    def test_solve(self, symmetry, bra, b_ndim, odd, kernels):
        odd_charges = {"Z2": 1, "U1": 1, "Z2Z2": (1, 0), "U1U1": (1, 0)}
        charge = odd_charges[symmetry] if odd else None
        ind = sr.utils.rand_index(symmetry, 7, seed=16)
        a = sr.utils.get_rand(
            symmetry, (ind, ind.conj()), seed=17, fermionic=True
        )
        shape = (ind,) if b_ndim == 1 else (ind, 5)
        b = sr.utils.get_rand(
            symmetry,
            shape,
            seed=18,
            fermionic=True,
            charge=charge,
            label="b",
        )
        a.randomize_phases(seed=19, inplace=True)
        b.randomize_phases(seed=20, inplace=True)
        if bra:
            a = a.conj(phase_dual=True)
            b = b.conj(phase_dual=True)
        assert_matches(
            sr.linalg.solve(a.to_packed(), b.to_packed()),
            sr.linalg.solve(a, b),
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_squeeze_odd_to_dummy_mode(self, symmetry, kernels):
        x, _ = get_rand(symmetry, 13)
        c = {"Z2": 1, "U1": 1, "Z2Z2": (1, 0), "U1U1": (1, 0)}[symmetry]
        e = x.expand_dims(1, c=c)
        assert_matches(e.to_packed().squeeze(1), e.squeeze(1))


class TestReplacePhases:
    def test_copy_with_and_modify(self, kernels):
        _, xp = get_rand("Z2", 14)
        phases = np.where(np.arange(xp.num_blocks) % 2, -1, 1)
        new = xp.copy_with(phases=phases)
        assert new.phases.tolist() == phases.tolist()
        assert new.copy_with(phases=None)._phases is None
        xp.modify(phases=phases)
        assert xp.phases.tolist() == phases.tolist()
