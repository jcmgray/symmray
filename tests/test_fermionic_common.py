import itertools

import numpy as np
import pytest

import symmray as sr
from symmray.fermionic_common import (
    _annihilate_sorted_phase,
    _koszul_sort_phase,
)
from symmray.fermionic_local_operators import FermionicOperator
from symmray.flat.flat_fermionic_array import perm_to_swaps


def _sorted_modes(modes):
    """Sort into the canonical order `_annihilate_sorted_phase` expects."""
    return sorted(modes)


def _reference_reduce(modes):
    """Independent ground truth: trace out a conjugate pair wherever it sits,
    by sliding its right half left onto its left half and paying
    ``(-1) ** (p * p_k)`` per mode crossed, plus the ket-bra sign. Unlike
    `_annihilate_sorted_phase` this does not assume any ordering, and it keeps
    going until no pair is left at all.
    """
    modes = list(modes)
    phase = 1
    while True:
        pairs = [
            (i, j)
            for i in range(len(modes))
            for j in range(i + 1, len(modes))
            if modes[i].label == modes[j].label
            and modes[i].dual != modes[j].dual
        ]
        if not pairs:
            return modes, phase

        i, j = min(pairs, key=lambda ij: ij[1] - ij[0])
        for k in range(i + 1, j):
            if modes[k].parity and modes[j].parity:
                phase = -phase
        b = modes.pop(j)
        modes.insert(i + 1, b)
        if b.dual and b.parity:
            phase = -phase
        modes.pop(i)
        modes.pop(i)


def _random_pair_modes(rng, labels, parity="random", wrap=int):
    """Modes over random unique ``(dual, label)`` combos, where a label's ket
    and bra share a parity, as they do when one comes from an array and the
    other from its conjugate. ``parity="odd"`` makes every mode occupied, as
    the sparse backend always does, which exercises every crossing sign.
    """
    combos = [(bool(d), label) for label in labels for d in (0, 1)]
    k = int(rng.integers(2, len(combos) + 1))
    picked = [combos[i] for i in rng.permutation(len(combos))[:k]]
    if parity == "odd":
        parities = {label: 1 for label in labels}
    else:
        parities = {label: int(rng.integers(0, 2)) for label in labels}
    return [
        FermionicOperator(label, dual=dual, parity=wrap(parities[label]))
        for dual, label in picked
    ]


def _reference_swap_phase(modes):
    """The original implementation: sort the modes into ascending label order
    via a sequence of adjacent swaps, multiplying ``(-1) ** (p_i * p_j)`` for
    each swap. This is the exact code that ``_koszul_sort_phase`` replaced.
    """
    modes = list(modes)
    perm = tuple(sorted(range(len(modes)), key=modes.__getitem__))
    phase = 1
    for i, j in perm_to_swaps(perm):
        a, b = modes[i], modes[j]
        phase = phase * ((a.parity * b.parity) * -2 + 1)
        modes[i], modes[j] = b, a
    return phase


def _random_sort_modes(rng, n, wrap):
    """Build ``n`` random modes. Labels are a mix of ints and tuples (as in
    ``unpack``'s position vs ('squeeze', ...) modes), duals are random, and the
    parity is wrapped by ``wrap`` to select the backend/type under test. The
    ``(dual, label)`` combos are unique, matching real use where dummy mode
    labels derive from unique array labels.
    """
    modes = []
    seen = set()
    while len(modes) < n:
        if rng.integers(0, 2):
            label = int(rng.integers(0, n))
        else:
            label = ("squeeze", int(rng.integers(0, n)), 4)
        dual = bool(rng.integers(0, 2))
        if (dual, label) in seen:
            continue
        seen.add((dual, label))
        parity = int(rng.integers(0, 2))
        modes.append(FermionicOperator(label, dual=dual, parity=wrap(parity)))
    return modes


def _wrap_python(p):
    return int(p)


def _wrap_numpy(p):
    import numpy as np

    return np.int64(p)


def _wrap_jax(p):
    import jax.numpy as jnp

    return jnp.asarray(p, dtype=jnp.int32)


def _wrap_torch(p):
    import torch

    return torch.tensor(p, dtype=torch.int32)


_KOSZUL_WRAPPERS = {
    "python": (_wrap_python, "numpy"),  # all-static -> backend unused
    "numpy": (_wrap_numpy, "numpy"),
    "jax": (_wrap_jax, "jax"),
    "torch": (_wrap_torch, "torch"),
}


class TestAnnihilateSortedPhase:
    """Tracing conjugate dummy mode pairs out of a sorted sequence."""

    def test_no_pairs_is_free(self):
        # the amplitude case: every label distinct, so nothing to trace out
        modes = _sorted_modes(
            FermionicOperator(("squeeze", i, 0), dual=bool(i % 2), parity=1)
            for i in range(5)
        )
        survivors, phase = _annihilate_sorted_phase(modes, "numpy")
        assert survivors == modes
        # a plain int, so the caller can skip the phase multiply entirely
        assert isinstance(phase, int)
        assert phase == 1

    def test_adjacent_pair(self):
        modes = _sorted_modes(
            [
                FermionicOperator("a", dual=True, parity=1),
                FermionicOperator("a", dual=False, parity=1),
            ]
        )
        survivors, phase = _annihilate_sorted_phase(modes, "numpy")
        assert survivors == []
        assert phase == 1

    def test_crossing_an_unpaired_mode_flips_the_sign(self):
        # sorted order splits the pair, so tracing it out crosses `a+`
        modes = _sorted_modes(
            [
                FermionicOperator("b", dual=True, parity=1),
                FermionicOperator("b", dual=False, parity=1),
                FermionicOperator("a", dual=True, parity=1),
            ]
        )
        assert [str(m) for m in modes] == ["b+", "a+", "b-"]
        survivors, phase = _annihilate_sorted_phase(modes, "numpy")
        assert [str(m) for m in survivors] == ["a+"]
        assert phase == -1

    def test_nested_pairs_cancel(self):
        # every crossed mode belongs to a pair, so both halves cross and the
        # signs cancel
        modes = _sorted_modes(
            FermionicOperator(label, dual=dual, parity=1)
            for label in "abc"
            for dual in (True, False)
        )
        assert [str(m) for m in modes] == ["c+", "b+", "a+", "a-", "b-", "c-"]
        survivors, phase = _annihilate_sorted_phase(modes, "numpy")
        assert survivors == []
        assert phase == 1

    def test_statically_even_pair_contributes_no_sign(self):
        modes = _sorted_modes(
            [
                FermionicOperator("b", dual=True, parity=0),
                FermionicOperator("b", dual=False, parity=0),
                FermionicOperator("a", dual=True, parity=1),
            ]
        )
        survivors, phase = _annihilate_sorted_phase(modes, "numpy")
        assert [str(m) for m in survivors] == ["a+"]
        assert phase == 1

    @pytest.mark.parametrize("duals", [(True, True), (False, False)])
    def test_duplicate_non_conjugate_labels_raise(self, duals):
        modes = [
            FermionicOperator("a", dual=duals[0], parity=1),
            FermionicOperator("a", dual=duals[1], parity=1),
        ]
        with pytest.raises(ValueError, match="unique conjugate pairs"):
            _annihilate_sorted_phase(modes, "numpy")

    @pytest.mark.parametrize("parity", ["random", "odd"])
    @pytest.mark.parametrize("seed", range(20))
    def test_matches_reference_reduction(self, seed, parity):
        rng = sr.utils.get_rng(seed)
        modes = _sorted_modes(_random_pair_modes(rng, "abcd", parity))

        survivors, phase = _annihilate_sorted_phase(modes, "numpy")
        # no conjugate pair may survive
        assert not any(
            a.label == b.label and a.dual != b.dual
            for a, b in itertools.combinations(survivors, 2)
        )

        expected, expected_phase = _reference_reduce(modes)
        assert [(m.dual, m.label) for m in survivors] == [
            (m.dual, m.label) for m in expected
        ]
        assert int(phase) == expected_phase

    @pytest.mark.parametrize("backend", ["jax", "torch"])
    @pytest.mark.parametrize("parity", ["random", "odd"])
    @pytest.mark.parametrize("seed", range(10))
    def test_traced_parities_match_static(
        self, backend, seed, parity, require_backend
    ):
        """Only the sign depends on the parities, so wrapping them as backend
        scalars must leave the survivors alone and the sign unchanged.
        """
        require_backend(backend)
        import autoray as ar

        rng = sr.utils.get_rng(seed)
        modes = _sorted_modes(_random_pair_modes(rng, "abcd", parity))
        traced = [
            FermionicOperator(
                m.label,
                dual=m.dual,
                parity=ar.do("asarray", m.parity, like=backend),
            )
            for m in modes
        ]

        expected, expected_phase = _annihilate_sorted_phase(modes, "numpy")
        survivors, phase = _annihilate_sorted_phase(traced, backend)

        assert [(m.dual, m.label) for m in survivors] == [
            (m.dual, m.label) for m in expected
        ]
        assert int(phase) == int(expected_phase)

    def test_under_jit(self):
        """The survivors are static, so a single graph covers every parity."""
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        labels = "abc"
        modes = _sorted_modes(
            FermionicOperator(label, dual=dual, parity=1)
            for label in labels
            for dual in (True, False)
        )
        # drop one half so that a pair has to cross an unpaired mode
        modes = [m for m in modes if not (m.label == "a" and not m.dual)]

        def f(parities):
            traced = [
                FermionicOperator(
                    m.label,
                    dual=m.dual,
                    parity=parities[labels.index(m.label)],
                )
                for m in modes
            ]
            return _annihilate_sorted_phase(traced, "jax")[1]

        compiled_f = jax.jit(f)
        for ps in itertools.product((0, 1), repeat=len(labels)):
            static = [
                FermionicOperator(
                    m.label, dual=m.dual, parity=ps[labels.index(m.label)]
                )
                for m in modes
            ]
            expected = _annihilate_sorted_phase(static, "numpy")[1]
            assert int(compiled_f(jnp.asarray(ps))) == int(expected)


class TestContractionTracesOutDummyModes:
    """Both backends should trace conjugate pairs out during contraction, and
    agree on what is left.
    """

    @staticmethod
    def get_rand(charge, duals, flat, seed=42):
        return sr.utils.get_rand(
            "Z2",
            shape=(4,) * len(duals),
            duals=duals,
            charge=charge,
            fermionic=True,
            label="x",
            flat=flat,
            subsizes="equal",
            seed=seed,
        )

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_gram_leaves_no_dummy_modes(self, flat, charge, duals):
        x = self.get_rand(charge, duals, flat)
        g = sr.tensordot(x.conj(), x, axes=((1, 2), (1, 2)))
        assert g.dummy_modes == ()

    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_backends_agree(self, charge, duals):
        xs = self.get_rand(charge, duals, flat=False)
        xf = self.get_rand(charge, duals, flat=True)
        gs = sr.tensordot(xs.conj(), xs, axes=((1, 2), (1, 2)))
        gf = sr.tensordot(xf.conj(), xf, axes=((1, 2), (1, 2)))
        assert gs.dummy_modes == gf.dummy_modes
        np.testing.assert_allclose(gs.to_dense(), gf.to_dense())

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("seed", range(10))
    def test_chain_of_odd_arrays_is_fully_reduced(self, flat, seed):
        """Contract a chain of odd-parity arrays against its own conjugate:
        every dummy mode should pair up and go.
        """
        rng = sr.utils.get_rng(seed)
        n = 3
        bonds = [
            sr.utils.rand_index("Z2", 4, subsizes="equal", seed=rng)
            for _ in range(n - 1)
        ]
        ts = []
        for i in range(n):
            shape = []
            if i > 0:
                shape.append(bonds[i - 1].conj())
            if i < n - 1:
                shape.append(bonds[i])
            shape.append(4)
            ts.append(
                sr.utils.get_rand(
                    "Z2",
                    shape=tuple(shape),
                    charge=1,
                    fermionic=True,
                    label=i,
                    flat=flat,
                    subsizes="equal",
                    seed=rng,
                )
            )

        ket = ts[0]
        for i in range(1, n):
            ket = sr.tensordot(ket, ts[i], axes=((ket.ndim - 2,), (0,)))

        axes = tuple(range(ket.ndim - 1))
        z = sr.tensordot(ket.conj(), ket, axes=(axes, axes))
        assert z.dummy_modes == ()


class TestKoszulSortPhase:
    """The vectorized `_koszul_sort_phase` must reproduce, exactly, the
    original swap-by-swap accumulation of the dummy mode sorting sign that
    used to live inline in `_resolve_dummy_modes_combine`. Parities are
    exercised as plain python ints, numpy scalars and jax arrays, the last
    both eagerly and under `jax.jit`, that is the traced-compilation case.
    """

    @pytest.mark.parametrize("kind", ["python", "numpy", "jax", "torch"])
    @pytest.mark.parametrize("seed", range(10))
    def test_matches_swap_loop(self, kind, seed):
        if kind in ("jax", "torch"):
            pytest.importorskip(kind)
        wrap, backend = _KOSZUL_WRAPPERS[kind]

        rng = sr.utils.get_rng(seed)
        n = int(rng.integers(1, 9))
        modes = _random_sort_modes(rng, n, wrap)

        expected = int(_reference_swap_phase(modes))
        got = int(_koszul_sort_phase(modes, backend))

        assert got in (-1, 1)
        assert got == expected

    @pytest.mark.parametrize("kind", ["numpy", "jax", "torch"])
    @pytest.mark.parametrize("seed", range(10))
    def test_mixed_static_traced(self, kind, seed):
        """Dummy modes merged from different arrays can mix plain int parities
        with backend scalars: check the vectorized path folds the static ones in
        as python data (torch.stack, for one, rejects non-tensors).
        """
        if kind in ("jax", "torch"):
            pytest.importorskip(kind)
        wrap, backend = _KOSZUL_WRAPPERS[kind]

        rng = sr.utils.get_rng(seed)
        n = int(rng.integers(2, 9))
        k = iter(range(n))
        modes = _random_sort_modes(
            rng, n, lambda p: wrap(p) if next(k) % 2 else int(p)
        )

        expected = int(_reference_swap_phase(modes))
        got = int(_koszul_sort_phase(modes, backend))

        assert got in (-1, 1)
        assert got == expected

    @pytest.mark.parametrize("seed", range(10))
    def test_under_jit(self, seed):
        """The real use case: parities are traced. Build the modes *inside* a
        jitted function from a traced parity vector and check the (concrete) result
        still matches the eager swap-loop reference.
        """
        jax = pytest.importorskip("jax")
        import jax.numpy as jnp

        rng = sr.utils.get_rng(seed)
        n = int(rng.integers(2, 9))

        # static structure (labels, duals) + a parity vector that will be traced
        labels = [
            int(rng.integers(0, n))
            if rng.integers(0, 2)
            else ("squeeze", int(rng.integers(0, n)), 4)
            for _ in range(n)
        ]
        duals = [bool(rng.integers(0, 2)) for _ in range(n)]
        parities = [int(rng.integers(0, 2)) for _ in range(n)]

        ref_modes = [
            FermionicOperator(labels[k], dual=duals[k], parity=parities[k])
            for k in range(n)
        ]
        expected = int(_reference_swap_phase(ref_modes))

        def f(p):
            modes = [
                FermionicOperator(labels[k], dual=duals[k], parity=p[k])
                for k in range(n)
            ]
            return _koszul_sort_phase(modes, "jax")

        got = int(jax.jit(f)(jnp.asarray(parities, dtype=jnp.int32)))

        assert got in (-1, 1)
        assert got == expected

    def test_explicit_cases(self):
        """A few hand-checked cases, including a full reversal of odd modes."""
        # all-even -> always +1
        even = [FermionicOperator(k, parity=0) for k in (3, 2, 1, 0)]
        assert _koszul_sort_phase(even, "numpy") == 1

        # two odd modes out of order -> one swap -> -1
        swap_two = [
            FermionicOperator(1, parity=1),
            FermionicOperator(0, parity=1),
        ]
        assert _koszul_sort_phase(swap_two, "numpy") == -1
        assert int(_reference_swap_phase(swap_two)) == -1

        # reversed sequence of m odd modes -> sign (-1) ** (m * (m - 1) // 2)
        for m in range(1, 7):
            modes = [FermionicOperator(m - 1 - k, parity=1) for k in range(m)]
            sign = -1 if ((m * (m - 1) // 2) % 2) else 1
            assert _koszul_sort_phase(modes, "numpy") == sign
            assert int(_reference_swap_phase(modes)) == sign

        # an even mode interleaved among odd modes must not change the sign
        with_even = [
            FermionicOperator(2, parity=1),
            FermionicOperator(1, parity=0),  # even, transparent
            FermionicOperator(0, parity=1),
        ]
        assert _koszul_sort_phase(with_even, "numpy") == int(
            _reference_swap_phase(with_even)
        )

    def test_mixed_explicit_torch(self):
        torch = pytest.importorskip("torch")

        # inverted pair with one static and one traced parity: the plain int
        # must not reach torch.stack
        mixed = [
            FermionicOperator(1, parity=1),
            FermionicOperator(0, parity=torch.tensor(1)),
        ]
        assert int(_koszul_sort_phase(mixed, "torch")) == -1

        # a statically-even mode prunes even when other parities are traced,
        # leaving no contributing pair -> plain int 1, no graph ops
        pruned = [
            FermionicOperator(1, parity=0),
            FermionicOperator(0, parity=torch.tensor(1)),
        ]
        res = _koszul_sort_phase(pruned, "torch")
        assert isinstance(res, int) and res == 1


class TestConjPhaseDualDummyModes:
    """Check dual dummy mode phases for kets and conjugated bras."""

    @staticmethod
    def get_arrays(flat, charge, duals, seed=42):
        x = sr.utils.get_rand(
            "Z2",
            shape=(4,) * len(duals),
            duals=duals,
            charge=charge,
            fermionic=True,
            label="x",
            flat=flat,
            subsizes="equal",
            seed=seed,
        )
        return x, x.conj(phase_dual=True), x.dagger(phase_dual=True)

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_norm(self, flat, charge, duals):
        for x in self.get_arrays(flat, charge, duals):
            n2 = sr.tensordot(x.conj(phase_dual=True), x, x.ndim)
            assert float(n2) == pytest.approx(x.norm() ** 2)

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_involution(self, flat, charge, duals):
        for x in self.get_arrays(flat, charge, duals):
            x.conj(phase_dual=True).conj(phase_dual=True).test_allclose(x)
            x.dagger(phase_dual=True).dagger(phase_dual=True).test_allclose(x)

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_gram_of_bra(self, flat, charge, duals):
        _, x, _ = self.get_arrays(flat, charge, duals)
        for axis in range(x.ndim):
            m = x.gram(axis)
            physical = m.phase_flip(1) if not m.duals[1] else m
            d = physical.to_dense()
            assert np.linalg.eigvalsh(d).min() > -1e-10
            assert np.trace(d).real == pytest.approx(x.norm() ** 2)


class TestConjOuterAxesAndInnerDummyModes:
    """Check selected outer axes and paired dummy modes in a network."""

    @staticmethod
    def get_array(flat, charge, duals, seed=42):
        return sr.utils.get_rand(
            "Z2",
            shape=(4,) * len(duals),
            duals=duals,
            charge=charge,
            fermionic=True,
            label="x",
            flat=flat,
            subsizes="equal",
            seed=seed,
        )

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_all_axes_matches_true(self, flat, charge, duals):
        x = self.get_array(flat, charge, duals)
        for x in (x, x.conj(phase_dual=True)):
            axes = tuple(range(x.ndim))
            x.conj(phase_dual=axes).test_allclose(x.conj(phase_dual=True))
            x.dagger(phase_dual=axes).test_allclose(x.dagger(phase_dual=True))

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_axes_match_phase_flip(self, flat, charge, duals):
        # a ket, and a bra with dual dummy modes
        x = self.get_array(flat, charge, duals)
        for x in (x, x.conj(phase_dual=True)):
            ndim = x.ndim
            dummy_parity = sum(m.parity for m in x.dummy_modes if m.dual)
            for r in range(ndim + 1):
                # includes axes=(), which still phases the dummy modes
                for axes in itertools.combinations(range(ndim), r):
                    expected = x.conj()
                    flip = [ax for ax in axes if not expected.indices[ax].dual]
                    expected = expected.phase_flip(*flip)
                    expected = expected.phase_global(parity=dummy_parity)
                    x.conj(phase_dual=axes).test_allclose(expected)

                    expected = x.dagger()
                    flip = [
                        ndim - 1 - ax
                        for ax in axes
                        if not expected.indices[ndim - 1 - ax].dual
                    ]
                    expected = expected.phase_flip(*flip)
                    expected = expected.phase_global(parity=dummy_parity)
                    x.dagger(phase_dual=axes).test_allclose(expected)

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("phase_dual", [False, True])
    def test_inner_dummy_modes_relabelled_and_involution(
        self, flat, phase_dual
    ):
        x = self.get_array(flat, 1, (False, True, False))
        (m,) = x.dummy_modes
        if phase_dual:
            # skipped inner phases leave this sign after two calls
            x_twice = x.phase_global(parity=m.parity)
        else:
            x_twice = x

        y = x.conj(phase_dual=phase_dual, inner_dummy_labels={"x"})
        assert y.dummy_modes == (m.vconj,)
        z = y.conj(phase_dual=phase_dual, inner_dummy_labels={("vconj", "x")})
        assert z.dummy_modes == x.dummy_modes
        z.test_allclose(x_twice)

        y = x.dagger(phase_dual=phase_dual, inner_dummy_labels={"x"})
        assert y.dummy_modes == (m.vconj,)
        z = y.dagger(
            phase_dual=phase_dual, inner_dummy_labels={("vconj", "x")}
        )
        assert z.dummy_modes == x.dummy_modes
        z.test_allclose(x_twice)

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize(
        "duals", list(itertools.product((False, True), repeat=3))
    )
    def test_region_with_pair_norm(self, flat, duals):
        # two odd arrays with a conjugate dummy mode pair
        dpa, db, dpb = duals
        a = self.get_array(flat, 1, (dpa, db), seed=1)
        b = self.get_array(flat, 1, (not db, dpb), seed=2)
        a.modify(dummy_modes=(FermionicOperator("r"),))
        b.modify(dummy_modes=(FermionicOperator("r", dual=True),))

        r = sr.tensordot(a, b, ((1,), (0,)))
        assert r.dummy_modes == ()
        expected = r.norm() ** 2

        # physical axes are outer, the bond and dummy pair are inner
        ac = a.conj(phase_dual=(0,), inner_dummy_labels={"r"})
        bc = b.conj(phase_dual=(1,), inner_dummy_labels={"r"})

        # bra and ket regions first
        rc = sr.tensordot(ac, bc, ((1,), (0,)))
        n2 = sr.tensordot(rc, r, ((0, 1), (0, 1)))
        assert float(n2) == pytest.approx(expected)

        # site by site
        la = sr.tensordot(ac, a, ((0,), (0,)))
        lb = sr.tensordot(bc, b, ((1,), (1,)))
        n2 = sr.tensordot(la, lb, ((0, 1), (0, 1)))
        assert float(n2) == pytest.approx(expected)

        # ket and bra arrays interleaved
        t = sr.tensordot(ac, a, ((0,), (0,)))
        t = sr.tensordot(t, b, ((1,), (0,)))
        n2 = sr.tensordot(t, bc, ((0, 1), (0, 1)))
        assert float(n2) == pytest.approx(expected)

        t = sr.tensordot(ac, r, ((0,), (0,)))
        n2 = sr.tensordot(t, bc, ((0, 1), (0, 1)))
        assert float(n2) == pytest.approx(expected)

        # conjugating the region twice gives it back
        inner = {("vconj", "r")}
        acc = ac.conj(phase_dual=(0,), inner_dummy_labels=inner)
        bcc = bc.conj(phase_dual=(1,), inner_dummy_labels=inner)
        assert acc.dummy_modes == a.dummy_modes
        assert bcc.dummy_modes == b.dummy_modes
        sr.tensordot(acc, bcc, ((1,), (0,))).test_allclose(r)
