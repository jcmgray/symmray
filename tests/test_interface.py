import autoray as ar
import numpy as np
import pytest

import symmray as sr


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "complex64", "complex128"],
)
@pytest.mark.parametrize("flat", [False, True])
def test_finfo(dtype, flat):
    x = sr.utils.get_rand(
        "Z2",
        (4, 4),
        seed=42,
        dtype=dtype,
        flat=flat,
        subsizes="equal",
    )

    expected = np.finfo(dtype)
    assert sr.finfo(x.dtype).eps == expected.eps
    assert ar.get_namespace(x).finfo(x.dtype).eps == expected.eps


class TestApplyToBlocks:
    @pytest.mark.parametrize("layout", ("sparse", "flat", "packed"))
    def test_expm_matches_dense(self, layout):
        import scipy.linalg

        x = sr.utils.get_rand(
            "Z2", (4, 4), duals=(False, True), subsizes="equal", seed=7
        )
        # square blocks
        x = x.tensordot(x.conj(), ((1,), (1,)))
        if layout == "flat":
            x = x.to_flat()
        elif layout == "packed":
            x = x.to_packed()
        np.testing.assert_allclose(
            sr.scipy.linalg.expm(x).to_dense(),
            scipy.linalg.expm(x.to_dense()),
        )

    @pytest.mark.parametrize("layout", ("sparse", "flat", "packed"))
    @pytest.mark.parametrize("conj", (False, True))
    def test_expm_fermionic(self, layout, conj):
        x = 0.3 * sr.utils_test.rand_herm("Z2", 8, seed=3, fermionic=True)
        if conj:
            # both duals reversed, and lazy phases
            x = x.conj()
        if layout == "flat":
            x = x.to_flat()
        elif layout == "packed":
            x = x.to_packed()
        e1 = sr.scipy.linalg.expm(x)
        e1.test_allclose(sr.scipy.linalg.expm(x.phase_sync()))
        (e1 @ e1).test_allclose(sr.scipy.linalg.expm(2 * x))

    @pytest.mark.parametrize("layout", ("sparse", "packed"))
    def test_expm_fermionic_random_phases(self, layout):
        x = 0.3 * sr.utils_test.rand_herm("U1", 7, seed=3, fermionic=True)
        x.randomize_phases(seed=4, inplace=True)
        if layout == "packed":
            x = x.to_packed()
        e1 = sr.scipy.linalg.expm(x)
        e1.test_allclose(sr.scipy.linalg.expm(x.phase_sync()))
        (e1 @ e1).test_allclose(sr.scipy.linalg.expm(2 * x))

    @pytest.mark.parametrize("cls", ("BlockVector", "PackedVector"))
    def test_vector_blocks(self, cls):
        v = getattr(sr, cls)({0: np.array([1.0, 2.0]), 1: np.array([3.0])})
        v.apply_to_blocks(lambda b: b[::-1])
        np.testing.assert_allclose(v.blocks[0], [2.0, 1.0])
