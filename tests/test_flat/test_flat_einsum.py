import autoray as ar
import numpy as np
import pytest

import symmray as sr


def make_array(
    eq, symmetry="Z2", fermionic=False, charge=0, reverse=False, flat=True
):
    lhs = eq.split("->")[0]
    duals = tuple((c in lhs[:i]) != reverse for i, c in enumerate(lhs))
    return sr.utils.get_rand(
        symmetry,
        (2 * int(symmetry[1:]),) * len(lhs),
        duals=duals,
        charge=charge,
        fermionic=fermionic,
        label="x",
        flat=flat,
        subsizes="equal",
        dtype="complex128",
        seed=42,
    )


class TestFlatEinsum:
    @pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z5"])
    @pytest.mark.parametrize("fermionic", [False, True])
    @pytest.mark.parametrize("charge", [0, 1])
    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.parametrize(
        "eq",
        ["abc->cab", "aa->", "abab->", "abcb->ca", "abcbdc->ad", "aba->b"],
    )
    def test_against_sparse(self, eq, symmetry, fermionic, charge, reverse):
        x = make_array(eq, symmetry, fermionic, charge, reverse)
        if fermionic:
            x.phase_flip(0, inplace=True)
        # keep pending phases aligned when blocks are shuffled
        order = np.random.default_rng(5).permutation(x.num_blocks)
        kwargs = {"phases": x.phases[order]} if fermionic else {}
        x = x.copy_with(
            sectors=x.sectors[order], blocks=x.blocks[order], **kwargs
        )
        original = x.copy(deep=True)
        expected = x.to_blocksparse().einsum(eq, preserve_array=True)
        y = x.einsum(eq, preserve_array=True)
        y.check()
        y.to_blocksparse().test_allclose(expected)
        assert y.label == x.label
        if fermionic:
            assert y.dummy_modes == x.dummy_modes
            assert y._phases is None
        x.test_allclose(original)
        np.testing.assert_array_equal(x.blocks, original.blocks)
        if not eq.split("->")[1]:
            assert x.einsum(eq) == pytest.approx(expected.to_dense())
        elif len(eq.split("->")[1]) == 1:
            assert y.indices[0].num_charges == 1

    @pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
    @pytest.mark.parametrize("fermionic", [False, True])
    @pytest.mark.parametrize(
        "eq",
        ["abc->cab", "aa->", "abab->", "abcb->ca", "abcbdc->ad", "aba->b"],
    )
    def test_backend(self, eq, fermionic, backend, require_backend):
        require_backend(backend)
        x = make_array(eq, fermionic=fermionic)
        expected = x.einsum(eq, preserve_array=True).to_dense()
        x = x.to(backend)
        y = ar.do("einsum", eq, x, preserve_array=True)
        y.check()
        assert y.backend == backend
        assert ar.to_numpy(y.to_dense()) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "eq", ["aa->", "abab->", "abcb->ca", "abcbdc->ad"]
    )
    def test_dense(self, eq):
        x = make_array(eq)
        expected = np.einsum(eq, x.to_dense())
        y = sr.einsum(eq, x)
        actual = y.to_dense() if hasattr(y, "to_dense") else y
        assert actual == pytest.approx(expected)

    def test_unequal_dimensions(self):
        x = sr.utils.get_rand(
            "Z2",
            (4, 6, 8, 6),
            duals=(False, False, True, True),
            flat=True,
            subsizes="equal",
            seed=42,
        )
        y = x.einsum("abcb->ca")
        assert y.shape == (8, 4)
        assert y.to_dense() == pytest.approx(
            np.einsum("abcb->ca", x.to_dense())
        )

    @pytest.mark.parametrize("fermionic", [False, True])
    def test_single_charge(self, fermionic):
        x = make_array("aa->", fermionic=fermionic).select_charge(0, 1)
        assert x.einsum("aa->") == pytest.approx(x.trace())
        x.einsum("ab->ba").test_allclose(x.transpose())
        x = x.expand_dims(2, c=0, dual=False)
        with pytest.raises(NotImplementedError, match="Flat partial traces"):
            x.einsum("aab->b")

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("fermionic", [False, True])
    @pytest.mark.parametrize(
        "eq,error",
        [
            ("abc", ValueError),
            ("ab->a", ValueError),
            ("abc->aa", ValueError),
            ("abc->d", ValueError),
            ("aab->ab", NotImplementedError),
            ("abc->ab", NotImplementedError),
            ("aaa->", NotImplementedError),
            ("...->...", NotImplementedError),
        ],
    )
    def test_invalid(self, fermionic, flat, eq, error):
        sr.utils.set_debug(False)
        x = make_array("abc->abc", fermionic=fermionic, flat=flat)
        with pytest.raises(error):
            x.einsum(eq)

    @pytest.mark.parametrize("flat", [False, True])
    @pytest.mark.parametrize("fermionic", [False, True])
    def test_incompatible_pairs(self, fermionic, flat):
        sr.utils.set_debug(False)
        x = make_array("abc->abc", fermionic=fermionic, flat=flat)
        with pytest.raises(ValueError, match="opposite duals"):
            x.einsum("aab->b")
        x = sr.utils.get_rand(
            "Z2",
            (4, 6),
            duals=(False, True),
            fermionic=fermionic,
            flat=flat,
            subsizes="equal",
            seed=42,
        )
        with pytest.raises(ValueError, match="matching sizes"):
            x.einsum("aa->")

    @pytest.mark.parametrize("fermionic", [False, True])
    def test_dispatch(self, fermionic):
        x = make_array("ab->ab", fermionic=fermionic)
        y = x.conj()
        for eq, axes, perm in [
            ("ab,bc->ca", ((1,), (0,)), (1, 0)),
            ("ab,cd->acbd", ((), ()), (0, 2, 1, 3)),
        ]:
            actual = sr.einsum(eq, x, y)
            expected = sr.tensordot(x, y, axes=axes).transpose(perm)
            actual.test_allclose(expected)
        assert sr.einsum("ab,ab->", x, y) == pytest.approx(
            sr.tensordot(x, y, axes=((0, 1), (0, 1)))
        )
        sr.einsum(x, ("a", "b"), ("b", "a")).test_allclose(x.transpose())

    @pytest.mark.parametrize("fermionic", [False, True])
    def test_trace_outer_product(self, fermionic):
        x = make_array("abc->abc", fermionic=fermionic)
        y = x.conj()
        outer = sr.tensordot(x, y, axes=0)
        actual = outer.einsum("abcabc->")
        expected = sr.tensordot(x, y, axes=((0, 1, 2), (0, 1, 2)))
        assert actual == pytest.approx(expected)
        with pytest.raises(NotImplementedError, match="Flat partial traces"):
            outer.einsum("abcabd->dc")

    @pytest.mark.parametrize("backend", ["numpy", "jax", "torch"])
    def test_zero_and_scalar(self, backend, require_backend):
        require_backend(backend)
        x = make_array("abab->", charge=1)
        x = x.copy_with(blocks=x.blocks.real.astype("float32")).to(backend)
        y = x.einsum("abab->", preserve_array=True)
        assert y.ndim == 0
        assert y.blocks.dtype == x.blocks.dtype
        assert ar.to_numpy(y.einsum("->")) == 0
        z = y.einsum("->", preserve_array=True)
        z.check()
        np.testing.assert_array_equal(
            ar.to_numpy(z.blocks), ar.to_numpy(y.blocks)
        )


class TestFlatEinsumAutodiff:
    @pytest.mark.parametrize("fermionic", [False, True])
    @pytest.mark.parametrize(
        "eq", ["abab->", "abcb->ca", "abcbdc->ad", "aba->b"]
    )
    def test_jit(self, eq, fermionic, require_backend):
        require_backend("jax")
        import jax
        import jax.numpy as jnp

        x = make_array(eq, fermionic=fermionic, charge=1)
        if fermionic:
            x.phase_flip(0, inplace=True)
        expected = x.einsum(eq, preserve_array=True)
        x = x.to("jax")

        def f(sectors, blocks, phases):
            kwargs = {"phases": phases} if fermionic else {}
            y = x.copy_with(sectors=sectors, blocks=blocks, **kwargs).einsum(
                eq, preserve_array=True
            )
            return y.sectors, y.blocks

        f = jax.jit(f)
        phases = x.phases if fermionic else jnp.ones(x.num_blocks)
        sectors, blocks = f(x.sectors, x.blocks, phases)
        np.testing.assert_array_equal(sectors, expected.sectors)
        assert ar.to_numpy(blocks) == pytest.approx(expected.blocks)
        # change sector order on a second compiled call
        order = jnp.arange(x.num_blocks - 1, -1, -1)
        sectors, blocks = f(x.sectors[order], x.blocks[order], phases[order])
        assert ar.to_numpy(blocks) == pytest.approx(expected.blocks)

    @pytest.mark.parametrize("backend", ["jax", "torch"])
    @pytest.mark.parametrize("eq", ["abab->", "abcb->ca", "abcbdc->ad"])
    def test_gradients(self, eq, backend, require_backend):
        require_backend(backend)
        x = make_array(eq)
        x = x.copy_with(blocks=x.blocks.real.copy())
        lhs = eq.split("->")[0]
        expected = np.zeros_like(x.blocks)
        # differentiate the dense trace definition one block entry at a time
        for sector, block in zip(x.sectors, expected):
            if any(
                sector[i] != sector[j]
                for i in range(len(lhs))
                for j in range(i)
                if lhs[i] == lhs[j]
            ):
                continue
            for coord in np.ndindex(block.shape):
                basis = np.zeros_like(block)
                basis[coord] = 1
                block[coord] = np.einsum(eq, basis).sum()
        x = x.to(backend)

        def loss(blocks):
            y = x.copy_with(blocks=blocks).einsum(eq, preserve_array=True)
            return ar.do("sum", y.blocks)

        if backend == "jax":
            import jax

            actual = jax.jit(jax.grad(loss))(x.blocks)
        else:
            blocks = x.blocks.detach().requires_grad_()
            loss(blocks).backward()
            actual = blocks.grad
        assert ar.to_numpy(actual) == pytest.approx(expected)
