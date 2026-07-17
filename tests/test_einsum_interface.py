"""Tests for the functional einsum interface of symmray objects, including
that contracting a symmray vector into an array, e.g. via ``cotengra.einsum``
or ``cotengra.array_contract``, dispatches to ``multiply_diagonal``.
"""

import autoray as ar
import cotengra as ctg
import numpy as np
import pytest
from numpy.testing import assert_allclose

import symmray as sr

all_symmetries = ["Z2", "Z3", "U1", "Z2Z2", "U1U1"]


def get_rand_compatible_vector(x, axis, seed=None):
    """Get a random BlockVector matching the chargemap of ``x.indices[axis]``,
    i.e. contractible into that axis as a diagonal matrix.
    """
    rng = np.random.default_rng(seed)
    return sr.BlockVector(
        {c: rng.normal(size=d) for c, d in x.indices[axis].chargemap.items()}
    )


class TestParseTensordotEq:
    @pytest.mark.parametrize(
        "eq,expected",
        [
            ("ab,bc->ac", ((1,), (0,), None)),
            ("ab,bc->ca", ((1,), (0,), (1, 0))),
            ("ab,cb->ac", ((1,), (1,), None)),
            ("ab,ab->", ((0, 1), (0, 1), None)),
            ("ijk,jkl->il", ((1, 2), (0, 1), None)),
            ("ab,cd->acbd", ((), (), (0, 2, 1, 3))),
            # batched index
            ("ab,ab->ab", None),
            ("ab,bc->abc", None),
            # summed index
            ("ab,bc->c", None),
            # repeated index
            ("aab,bc->ac", None),
        ],
    )
    def test_patterns(self, eq, expected):
        from symmray.interface import _parse_tensordot_eq

        assert _parse_tensordot_eq(eq) == expected


class TestEinsumGeneral:
    @pytest.mark.parametrize("eq", ["abc->cab", "abc->bca"])
    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_single_term(self, eq, symmetry):
        x = sr.utils.get_rand(symmetry, (3, 4, 5), seed=42)
        y = ar.do("einsum", eq, x)
        y.check()
        assert_allclose(y.to_dense(), np.einsum(eq, x.to_dense()))

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_single_term_trace(self, symmetry):
        a = sr.utils.rand_index(symmetry, 4, seed=42)
        x = sr.utils.get_rand(symmetry, (a, a.conj()), seed=42)
        y = ar.do("einsum", "aa->", x)
        assert y == pytest.approx(np.einsum("aa->", x.to_dense()))

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_two_term_tensordot_dispatch(self, symmetry):
        rng = sr.utils.get_rng(42)
        a = sr.utils.rand_index(symmetry, 3, seed=rng)
        b = sr.utils.rand_index(symmetry, 4, seed=rng)
        c = sr.utils.rand_index(symmetry, 5, seed=rng)
        x = sr.utils.get_rand(symmetry, (a, b), seed=rng)
        y = sr.utils.get_rand(symmetry, (b.conj(), c), seed=rng)
        z = ar.do("einsum", "ab,bc->ac", x, y)
        z.check()
        z.test_allclose(sr.tensordot(x, y, axes=1))

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_two_term_permuted_output(self, symmetry):
        rng = sr.utils.get_rng(42)
        a = sr.utils.rand_index(symmetry, 3, seed=rng)
        b = sr.utils.rand_index(symmetry, 4, seed=rng)
        c = sr.utils.rand_index(symmetry, 5, seed=rng)
        x = sr.utils.get_rand(symmetry, (a, b), seed=rng)
        y = sr.utils.get_rand(symmetry, (b.conj(), c), seed=rng)
        z = ar.do("einsum", "ab,bc->ca", x, y)
        z.check()
        # compare against explicit tensordot, since e.g. U1 tensordot can
        # drop empty charge sectors and so change the dense shape
        expected = sr.tensordot(x, y, axes=((1,), (0,))).transpose((1, 0))
        z.test_allclose(expected)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_two_term_full_contraction(self, symmetry):
        rng = sr.utils.get_rng(42)
        a = sr.utils.rand_index(symmetry, 3, seed=rng)
        b = sr.utils.rand_index(symmetry, 4, seed=rng)
        x = sr.utils.get_rand(symmetry, (a, b), seed=rng)
        y = sr.utils.get_rand(symmetry, (a.conj(), b.conj()), seed=rng)
        s = ar.do("einsum", "ab,ab->", x, y)
        expected = np.einsum("ab,ab->", x.to_dense(), y.to_dense())
        assert s == pytest.approx(expected)

    def test_two_term_batched_raises(self):
        rng = sr.utils.get_rng(42)
        a = sr.utils.rand_index("Z2", 3, seed=rng)
        b = sr.utils.rand_index("Z2", 4, seed=rng)
        x = sr.utils.get_rand("Z2", (a, b), seed=rng)
        y = sr.utils.get_rand("Z2", (a.conj(), b.conj()), seed=rng)
        with pytest.raises(NotImplementedError):
            ar.do("einsum", "ab,ab->ab", x, y)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_multi_term(self, symmetry):
        rng = sr.utils.get_rng(42)
        a = sr.utils.rand_index(symmetry, 3, seed=rng)
        b = sr.utils.rand_index(symmetry, 4, seed=rng)
        c = sr.utils.rand_index(symmetry, 5, seed=rng)
        x = sr.utils.get_rand(symmetry, (a, b), seed=rng)
        y = sr.utils.get_rand(symmetry, (b.conj(), c), seed=rng)
        z = sr.utils.get_rand(symmetry, (c.conj(), a.conj()), seed=rng)
        s = ar.do("einsum", "ab,bc,ca->", x, y, z)
        expected = np.einsum(
            "ab,bc,ca->", x.to_dense(), y.to_dense(), z.to_dense()
        )
        assert s == pytest.approx(expected)

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_interleaved_input(self, symmetry):
        rng = sr.utils.get_rng(42)
        a = sr.utils.rand_index(symmetry, 3, seed=rng)
        b = sr.utils.rand_index(symmetry, 4, seed=rng)
        c = sr.utils.rand_index(symmetry, 5, seed=rng)
        x = sr.utils.get_rand(symmetry, (a, b), seed=rng)
        y = sr.utils.get_rand(symmetry, (b.conj(), c), seed=rng)
        z = ar.do("einsum", x, ("a", "b"), y, ("b", "c"), ("a", "c"))
        z.check()
        z.test_allclose(sr.tensordot(x, y, axes=1))


class TestParseMultiplyDiagonalEq:
    @pytest.mark.parametrize(
        "eq,expected",
        [
            ("i,ijk->ijk", ("left", 0, None)),
            ("j,ijk->ijk", ("left", 1, None)),
            ("ijk,j->ijk", ("right", 1, None)),
            ("j,ijk->jki", ("left", 1, (1, 2, 0))),
            ("ijk,k->kij", ("right", 2, (2, 0, 1))),
            ("i,i->i", ("both", 0, None)),
            ("i,i->", None),
            ("j,ijk->ik", None),
            ("ij,ijk->ijk", None),
            ("i,iij->iij", None),
            ("i,jk->ijk", None),
        ],
    )
    def test_patterns(self, eq, expected):
        from symmray.interface import _parse_multiply_diagonal_eq

        assert _parse_multiply_diagonal_eq(eq) == expected


class TestEinsumVectorDispatch:
    @pytest.mark.parametrize("symmetry", all_symmetries)
    @pytest.mark.parametrize("fermionic", [False, True])
    def test_multiply_diagonal_sparse(self, symmetry, fermionic):
        x = sr.utils.get_rand(
            symmetry, (4, 5, 6), seed=42, fermionic=fermionic
        )
        v = get_rand_compatible_vector(x, axis=1, seed=43)
        z = ctg.einsum("j,ijk->ijk", v, x)
        z.check()
        assert_allclose(
            z.to_dense(),
            np.einsum("j,ijk->ijk", v.to_dense(), x.to_dense()),
        )

    @pytest.mark.parametrize("symmetry", ["Z2", "Z3", "Z4"])
    def test_multiply_diagonal_flat(self, symmetry):
        N = int(symmetry[1:])
        x = sr.utils.get_rand(
            symmetry, (2 * N, 3 * N, 2 * N), seed=42, subsizes="equal"
        )
        v = get_rand_compatible_vector(x, axis=1, seed=43)
        xf, vf = x.to_flat(), v.to_flat()
        zf = ctg.einsum("j,ijk->ijk", vf, xf)
        zf.check()
        assert_allclose(
            zf.to_dense(),
            np.einsum("j,ijk->ijk", vf.to_dense(), xf.to_dense()),
        )

    @pytest.mark.parametrize("eq", ["ijk,j->ijk", "j,ijk->jki", "ijk,k->kij"])
    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_argument_order_and_output_permutation(self, eq, symmetry):
        x = sr.utils.get_rand(symmetry, (4, 5, 6), seed=42)
        lhs, _ = eq.split("->")
        ta, tb = lhs.split(",")
        tv = ta if len(ta) == 1 else tb
        tx = tb if len(ta) == 1 else ta
        v = get_rand_compatible_vector(x, axis=tx.index(tv), seed=43)
        args = (v, x) if len(ta) == 1 else (x, v)
        z = ctg.einsum(eq, *args)
        z.check()
        assert_allclose(
            z.to_dense(),
            np.einsum(eq, *(a.to_dense() for a in args)),
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_array_contract(self, symmetry):
        x = sr.utils.get_rand(symmetry, (4, 5, 6), seed=42)
        v = get_rand_compatible_vector(x, axis=0, seed=43)
        z = ctg.array_contract(
            [v, x], inputs=[("i",), ("i", "j", "k")], output=("i", "j", "k")
        )
        z.check()
        assert_allclose(
            z.to_dense(),
            np.einsum("i,ijk->ijk", v.to_dense(), x.to_dense()),
        )

    @pytest.mark.parametrize("symmetry", all_symmetries)
    def test_vector_vector(self, symmetry):
        x = sr.utils.get_rand(symmetry, (4, 5), seed=42)
        v = get_rand_compatible_vector(x, axis=0, seed=43)
        w = get_rand_compatible_vector(x, axis=0, seed=44)
        z = ctg.einsum("i,i->i", v, w)
        assert_allclose(z.to_dense(), v.to_dense() * w.to_dense())

    @pytest.mark.parametrize(
        "eq", ["j,ijk->ik", "ijk,j->ik", "i,i->", "j,ik->jik"]
    )
    def test_unsupported_patterns_raise(self, eq):
        x = sr.utils.get_rand("Z2", (4, 5, 6), seed=42)
        x2 = sr.utils.get_rand("Z2", (4, 6), seed=42)
        v = get_rand_compatible_vector(x, axis=1, seed=43)
        w = get_rand_compatible_vector(x, axis=1, seed=44)
        args = {
            "j,ijk->ik": (v, x),
            "ijk,j->ik": (x, v),
            "i,i->": (v, w),
            # outer product with vector
            "j,ik->jik": (v, x2),
        }[eq]
        with pytest.raises(TypeError):
            ctg.einsum(eq, *args)

    def test_unsupported_direct_dispatch_raises(self):
        # direct symmray einsum dispatch raises rather than recursing back
        # into cotengra
        x = sr.utils.get_rand("Z2", (4, 5, 6), seed=42)
        v = get_rand_compatible_vector(x, axis=1, seed=43)
        with pytest.raises(NotImplementedError):
            ar.do("einsum", "j,ijk->ik", v, x)
