import itertools

import autoray as ar
import numpy as np
import pytest

import symmray as sr


@pytest.mark.parametrize(
    "symmetry, flat", [("Z2", False), ("U1", False), ("Z2", True)]
)
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("charge", [0, 1])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_gram(symmetry, flat, fermionic, charge, ndim):
    for duals in itertools.product([False, True], repeat=ndim):
        x = sr.utils.get_rand(
            symmetry,
            (4,) * ndim,
            duals=duals,
            charge=charge,
            fermionic=fermionic,
            flat=flat,
            subsizes="equal",
            dtype="complex128",
            seed=42,
            **({"label": 0} if fermionic else {}),
        )
        original = x.copy()
        for axis in range(ndim):
            m = sr.gram(x, axes=axis)
            m.check()
            assert m.duals == (not duals[axis], duals[axis])
            m.test_allclose(ar.do("gram", x, axes=axis - ndim))
            if fermionic:
                assert not m.dummy_modes
                m.test_allclose(m.dagger())
                physical = m.phase_flip(1) if not m.duals[1] else m
            else:
                physical = m
            d = physical.to_dense()
            np.testing.assert_allclose(d, d.conj().T, atol=1e-12)
            assert np.linalg.eigvalsh(d).min() > -1e-10
            assert np.trace(d).real == pytest.approx(x.norm() ** 2)
            if not fermionic:
                a = np.moveaxis(x.to_dense(), axis, -1)
                a = a.reshape(-1, a.shape[-1])
                expected = a.conj().T @ a
                if d.shape != expected.shape:
                    # sparse contraction drops unused charge sectors
                    keep = np.any(expected != 0, axis=0)
                    expected = expected[np.ix_(keep, keep)]
                np.testing.assert_allclose(d, expected, atol=1e-12)
        x.test_allclose(original)


@pytest.mark.parametrize("flat", [False, True])
def test_gram_invalid_axis(flat):
    x = sr.utils.get_rand("Z2", (4, 4), flat=flat, subsizes="equal", seed=42)
    for axis in (-3, 2):
        with pytest.raises(ValueError, match="out of bounds"):
            x.gram(axis)
    with pytest.raises(TypeError, match="integers"):
        x.gram(0.5)


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("flat", [False, True])
def test_gram_backend(backend, flat, convert_backend):
    x = sr.utils.get_rand(
        "Z2",
        (4, 4, 4),
        duals=(True, False, True),
        charge=1,
        fermionic=True,
        flat=flat,
        subsizes="equal",
        dtype="complex128",
        seed=42,
        label=0,
    )
    expected = x.gram(1).to_dense()
    x.set_params(convert_backend(x.get_params(), backend))
    m = x.gram(1)
    assert m.backend == backend
    assert not m.dummy_modes
    np.testing.assert_allclose(ar.to_numpy(m.to_dense()), expected, atol=1e-12)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("fermionic", [False, True])
@pytest.mark.parametrize("inplace", [False, True])
def test_unfuse_empty_group_roundtrip(flat, fermionic, inplace):
    x = sr.utils.get_rand(
        "Z2",
        (4, 4),
        duals=(True, False),
        fermionic=fermionic,
        flat=flat,
        subsizes="equal",
        dtype="complex128",
        seed=42,
    )
    # an empty group expands to a size-1 axis recording the empty fusion
    y = x.fuse((), (0, 1))
    assert y.shape == (1, 8)
    assert y.indices[0].subinfo is not None
    original = y.copy()

    # unfusing it removes the axis again, here via a negative axis
    z = y.unfuse(-y.ndim, inplace=inplace)
    assert (z is y) is inplace
    z.check()
    assert z.ndim == 1
    z.unfuse(0).test_allclose(x)
    if not inplace:
        y.test_allclose(original)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("fermionic", [False, True])
def test_unfuse_unfused_axis_raises(flat, fermionic):
    x = sr.utils.get_rand(
        "Z2",
        (4, 4),
        duals=(True, False),
        fermionic=fermionic,
        flat=flat,
        subsizes="equal",
        seed=42,
    )
    with pytest.raises(ValueError, match="not fused"):
        x.unfuse(0)
