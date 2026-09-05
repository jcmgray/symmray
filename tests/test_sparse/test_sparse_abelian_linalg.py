import warnings

import autoray as ar
import pytest
from numpy.testing import assert_allclose

import symmray as sr


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_qr_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand(symmetry, (d0, d1), duals=[f0, f1], charge=c)
    x.check()
    q, r = sr.linalg.qr(x)
    q.check()
    r.check()
    assert (q @ r).allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1", "Z2Z2"))
@pytest.mark.parametrize("seed", range(5))
def test_qr_with_expand_dims(symmetry, seed):
    x = sr.utils.get_rand(
        symmetry,
        [4, 5, 6],
        subsizes="maximal",
        seed=seed,
    )
    y = x.reshape((1, 4 * 5 * 6))
    q, r = sr.linalg.qr(y)
    z = (q @ r).reshape((4, 5, 6))
    assert z.allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_svd_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        duals=[f0, f1],
        charge=c,
        subsizes="maximal",
    )
    x.check()
    u, s, vh = sr.linalg.svd(x)
    u.check()
    s.check()
    vh.check()
    usvh = ar.do("einsum", "ij,j,jk->ik", u, s, vh)
    assert usvh.allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_svd_via_eig_basics(symmetry, d0, d1, f0, f1, c):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        duals=[f0, f1],
        charge=c,
        subsizes="maximal",
    )
    x.check()
    u, s, vh = x.svd_via_eig()
    u.check()
    s.check()
    vh.check()
    usvh = ar.do("einsum", "ij,j,jk->ik", u, s, vh)
    usvh.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
def test_svd_via_eig_truncated(symmetry, d0, d1, absorb):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        subsizes="maximal",
    )
    x.check()

    u, s, vh = x.svd_via_eig_truncated(absorb=absorb)
    if u is not None:
        u.check()
    if vh is not None:
        vh.check()

    if absorb is None:
        s.check()
        xr = ar.do("einsum", "ij,j,jk->ik", u, s, vh)
    else:
        assert s is None
        xr = sr.tensordot(u, vh, 1)

    xr.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d0", [4, 5])
@pytest.mark.parametrize("d1", [4, 5])
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize("seed", [42])
def test_svd_via_eig_truncated_max_bond(symmetry, d0, d1, absorb, seed):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        subsizes="maximal",
        seed=seed,
    )
    x.check()

    u, s, vh = x.svd_via_eig_truncated(max_bond=2, absorb=absorb)

    if u is not None:
        u.check()
    if vh is not None:
        vh.check()
    if s is not None:
        s.check()
        assert s.size <= 2


@pytest.mark.parametrize("symmetry", ("Z2", "U1"))
@pytest.mark.parametrize("d0", [4, 5])
@pytest.mark.parametrize("d1", [4, 5])
def test_svd_via_eig_truncated_ar_dispatch(symmetry, d0, d1, seed=42):
    """Check that autoray dispatch works for svd_via_eig_truncated."""
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        subsizes="maximal",
        seed=seed,
    )

    u, s, vh = ar.do("svd_via_eig_truncated", x, max_bond=2, absorb=None)
    u.check()
    vh.check()
    s.check()
    assert s.size <= 2


@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize(
    "max_bond, expected_chargemap",
    [
        (6, {0: 4, 1: 1, 2: 1}),
        (3, {0: 1, 1: 1, 2: 1}),
        (2, {0: 1, 1: 1}),
    ],
)
def test_svd_rand_truncated_max_bond(
    absorb,
    max_bond,
    expected_chargemap,
):
    x = sr.utils.get_rand(
        "Z3",
        shape=({0: 10, 1: 2, 2: 1}, {0: 10, 1: 2, 2: 1}),
        duals=(False, True),
        seed=42,
    )

    with warnings.catch_warnings():
        # every blockwise randomized SVD should receive a finite max_bond
        warnings.filterwarnings(
            "error",
            message="Using 'svd:rand' without `max_bond`",
        )
        u, s, vh = ar.do(
            "svd_rand_truncated",
            x,
            max_bond=max_bond,
            absorb=absorb,
            seed=42,
        )

    u.check()
    vh.check()
    assert u.indices[1].chargemap == expected_chargemap
    assert vh.indices[0].chargemap == expected_chargemap

    if absorb is None:
        s.check()
        assert s.size == max_bond
    else:
        assert s is None

    assert u.shape[0] == x.shape[0]
    assert vh.shape[1] == x.shape[1]
    if max_bond >= len(x.sectors):
        if absorb is None:
            xr = sr.einsum("ij,j,jk->ik", u, s, vh)
        else:
            xr = u @ vh
        xr.check()
        assert xr.shape == x.shape
        assert xr.charge == x.charge


@pytest.mark.parametrize(
    "cutoff, cutoff_mode",
    [
        (0.5, None),
        (100.0, "abs"),
    ],
)
def test_svd_rand_truncated_dynamic_cutoff(cutoff, cutoff_mode):
    import numpy as np

    x = sr.utils.get_rand(
        "Z3",
        shape=({0: 10, 1: 2, 2: 1}, {0: 10, 1: 2, 2: 1}),
        duals=(False, True),
        seed=42,
    )
    x.set_block((1, 1), np.diag([1000.0, 0.0]))

    kwargs = {}
    if cutoff_mode is not None:
        kwargs["cutoff_mode"] = cutoff_mode

    with warnings.catch_warnings():
        # every blockwise randomized SVD should receive a finite max_bond
        warnings.filterwarnings(
            "error",
            message="Using 'svd:rand' without `max_bond`",
        )
        u, s, vh = x.svd_rand_truncated(
            max_bond=2,
            cutoff=cutoff,
            absorb=None,
            seed=42,
            **kwargs,
        )

    reference_mode = "rel" if cutoff_mode is None else cutoff_mode
    _, s_reference, _ = x.svd_truncated(
        max_bond=2,
        cutoff=cutoff,
        cutoff_mode=reference_mode,
        absorb=None,
    )

    u.check()
    s.check()
    vh.check()
    assert s.sectors == (1,)
    s.test_allclose(s_reference)


def test_svd_truncated_default_relative_cutoff_max_bond_exact():
    x = sr.utils.get_rand(
        "Z3",
        shape=({0: 10, 1: 1, 2: 1}, {0: 10, 1: 1, 2: 1}),
        duals=(False, True),
        seed=42,
    )

    _, s, _ = x.svd_truncated(
        max_bond=6,
        cutoff=1e-12,
        absorb=None,
    )
    _, s_reference, _ = x.svd_truncated(
        max_bond=6,
        cutoff=1e-12,
        cutoff_mode="rsum2",
        absorb=None,
    )

    s.check()
    assert s.sectors == s_reference.sectors
    s.test_allclose(s_reference)


@pytest.mark.parametrize(
    "method",
    [
        "svd_truncated",
        "svd_via_eig_truncated",
        "eigh_truncated",
    ],
)
@pytest.mark.parametrize(
    "max_bond_mode, expected_sizes, expected_values",
    [
        (None, {1: 2, 2: 1}, [80.0, 90.0, 100.0]),
        ("global", {1: 2, 2: 1}, [80.0, 90.0, 100.0]),
        ("eager", {0: 1, 1: 1, 2: 1}, [10.0, 80.0, 100.0]),
    ],
)
def test_truncated_max_bond_modes(
    method,
    max_bond_mode,
    expected_sizes,
    expected_values,
):
    import numpy as np

    x = sr.AbelianArray.from_blocks(
        {
            (0, 0): np.diag(np.arange(1.0, 11.0)),
            (1, 1): np.diag([100.0, 90.0]),
            (2, 2): np.array([[80.0]]),
        },
        duals=(False, True),
        symmetry="Z3",
    )

    kwargs = {}
    if max_bond_mode is not None:
        kwargs["max_bond_mode"] = max_bond_mode

    _, s, _ = getattr(x, method)(
        cutoff=0.0,
        max_bond=3,
        absorb=None,
        **kwargs,
    )

    s.check()
    actual_sizes = {c: len(s.get_block(c)) for c in s.sectors}
    assert actual_sizes == expected_sizes
    assert_allclose(sorted(s.to_dense()), expected_values)


def test_svd_rand_truncated_global_max_bond():
    import numpy as np

    x = sr.AbelianArray.from_blocks(
        {
            (0, 0): np.diag(np.arange(1.0, 11.0)),
            (1, 1): np.diag([100.0, 90.0]),
            (2, 2): np.array([[80.0]]),
        },
        duals=(False, True),
        symmetry="Z3",
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using 'svd:rand' without `max_bond`",
        )
        _, s, _ = x.svd_rand_truncated(
            max_bond=3,
            max_bond_mode="global",
            absorb=None,
            seed=42,
        )

    s.check()
    actual_sizes = {c: len(s.get_block(c)) for c in s.sectors}
    assert actual_sizes == {1: 2, 2: 1}
    assert_allclose(sorted(s.to_dense()), [80.0, 90.0, 100.0])


def test_truncated_global_max_bond_preserves_degeneracy():
    import numpy as np

    x = sr.AbelianArray.from_blocks(
        {
            (0, 0): np.array([[5.0]]),
            (1, 1): np.diag([2.0, 2.0]),
            (2, 2): np.diag([2.0, 2.0, 1.0]),
        },
        duals=(False, True),
        symmetry="Z3",
    )

    _, s, _ = x.svd_truncated(
        cutoff=0.0,
        max_bond=3,
        max_bond_mode="global",
        absorb=None,
    )

    s.check()
    assert s.size == 5
    assert_allclose(sorted(s.to_dense()), [2.0, 2.0, 2.0, 2.0, 5.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (
            {"cutoff": 1e-3, "cutoff_mode": "rsum2"},
            "cumulative cutoff modes",
        ),
        ({"cutoff": 1e-3, "renorm": True}, "renorm"),
    ],
)
def test_truncated_eager_unsupported_options(kwargs, message):
    x = sr.utils.get_rand(
        "Z3",
        shape=({0: 4, 1: 2, 2: 1}, {0: 4, 1: 2, 2: 1}),
        duals=(False, True),
        seed=42,
    )

    with pytest.raises(ValueError, match=message):
        x.svd_truncated(
            max_bond=3,
            max_bond_mode="eager",
            **kwargs,
        )


def test_truncated_invalid_max_bond_mode():
    x = sr.utils.get_rand("Z2", (4, 4), seed=42)

    with pytest.raises(ValueError, match="max_bond_mode"):
        x.svd_truncated(max_bond=2, max_bond_mode="invalid")


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
@pytest.mark.parametrize("seed", range(1))
def test_eigh(symmetry, d, seed):
    x = sr.utils_test.rand_herm(symmetry, d, seed=seed)
    el, ev = ar.do("linalg.eigh", x)
    el.check()
    ev.check()
    xr = ar.do("einsum", "ij,j,jk->ik", ev, el, ev.H)
    xr.check()
    assert x.allclose(xr)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 5, 7))
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize("seed", range(1))
def test_eigh_truncated(symmetry, d, absorb, seed):
    x = sr.utils_test.rand_matrix(
        symmetry,
        d,
        seed=seed,
        matrix_type="posdef" if absorb == 0 else "hermitian",
    )

    u, s, vh = sr.linalg.eigh_truncated(x, absorb=absorb)
    if u is not None:
        u.check()
    if vh is not None:
        vh.check()

    if absorb is None:
        s.check()
        xr = ar.do("einsum", "ij,j,jk->ik", u, s, vh)
    else:
        assert s is None
        xr = sr.tensordot(u, vh, 1)

    xr.test_allclose(x)

    if absorb is None:
        z = u @ u.dagger_project_left() @ x @ vh.dagger_project_right() @ vh
        z.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (5, 7))
@pytest.mark.parametrize("absorb", [None, -1, 0, 1])
@pytest.mark.parametrize("seed", [42])
def test_eigh_truncated_max_bond(symmetry, d, absorb, seed):
    x = sr.utils_test.rand_matrix(
        symmetry,
        d,
        seed=seed,
        matrix_type="posdef" if absorb == 0 else "hermitian",
    )

    u, s, vh = sr.linalg.eigh_truncated(x, max_bond=2, absorb=absorb)

    if u is not None:
        u.check()
    if vh is not None:
        vh.check()
    if s is not None:
        s.check()
        assert s.size <= 2


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("seed", range(5))
def test_eigh_truncated_cutoff_max_bond(symmetry, seed):
    rng = sr.utils.get_rng(seed)

    x = sr.utils_test.rand_herm(symmetry, 20, seed=rng)

    # cutoff only
    _, s, _ = sr.linalg.eigh_truncated(
        x,
        cutoff=3e-1,
        absorb=None,
    )
    assert s.size < x.shape[0]

    # max_bond only
    _, s, _ = sr.linalg.eigh_truncated(
        x,
        cutoff=0.0,
        max_bond=3,
        absorb=None,
    )
    assert s.size <= 3

    # both
    _, s, _ = sr.linalg.eigh_truncated(
        x,
        cutoff=1e-2,
        max_bond=7,
        absorb=None,
    )
    assert s.size <= 7


def test_eigh_truncated_negative_relative_cutoff_max_bond():
    import numpy as np

    x = sr.AbelianArray.from_blocks(
        {
            (0, 0): np.diag([-9.0, -8.0, 1.0]),
            (1, 1): np.diag([-2.0, 1.0]),
            (2, 2): np.array([[-3.0]]),
        },
        duals=(False, True),
        symmetry="Z3",
    )

    _, s, _ = x.eigh_truncated(
        cutoff=0.5,
        cutoff_mode="rel",
        max_bond=2,
        absorb=None,
        positive=False,
    )

    s.check()
    assert s.sectors == (0,)
    assert_allclose(sorted(abs(s.to_dense())), [8.0, 9.0])


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky(symmetry, d, seed, dtype):
    x = sr.utils_test.rand_posdef(symmetry, d, seed=seed, dtype=dtype)

    left = sr.linalg.cholesky(x, upper=False)
    left.check()
    assert left.ndim == 2
    assert left.dtype == dtype
    # roundtrip: L @ L^H should equal A
    y = left @ left.H
    y.check()
    y.test_allclose(x)

    right = sr.linalg.cholesky(x, upper=True)
    right.check()
    assert right.ndim == 2
    assert right.dtype == dtype
    # roundtrip: R^H @ R should equal A
    y = right.H @ right
    y.check()
    y.test_allclose(x)

    # check left/lower and right/upper are consistent with each other
    y = left @ right
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 5))
@pytest.mark.parametrize("absorb", [-12, 0, 12])
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_cholesky_regularized(symmetry, d, absorb, seed, dtype):
    x = sr.utils_test.rand_posdef(symmetry, d, seed=seed, dtype=dtype)

    left, s, right = sr.linalg.cholesky_regularized(x, absorb=absorb)
    assert s is None

    if absorb == -12:
        assert right is None
        left.check()
        # reconstruct
        y = left @ left.H
        y.check()
        y.test_allclose(x)
    elif absorb == 12:
        assert left is None
        right.check()
        # reconstruct
        y = right.H @ right
        y.check()
        y.test_allclose(x)
    else:
        left.check()
        right.check()
        # roundtrip: L @ L^H == A
        y = left @ right
        y.check()
        y.test_allclose(x)


def test_cholesky_regularized_ar_dispatch():
    """Check that autoray dispatch works for cholesky_regularized."""
    x = sr.utils_test.rand_posdef("Z2", 2, seed=0, dtype="complex128")
    left, s, right = ar.do("cholesky_regularized", x)
    assert s is None
    left.check()
    right.check()
    y = left @ right
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d0", (3, 5))
@pytest.mark.parametrize("d1", (3, 7))
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_lq_via_cholesky(symmetry, d0, d1, dtype, seed):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        dtype=dtype,
        seed=seed,
        subsizes="maximal",
    )
    x.check()

    # default absorb: left => (L, None, Q)
    l, s, q = sr.linalg.lq_via_cholesky(x)
    assert s is None
    l.check()
    q.check()

    # roundtrip: L @ Q == x
    y = l @ q
    y.check()
    y.test_allclose(x)

    # lfactor absorb: (L, None, None)
    l2, _, q2 = sr.linalg.lq_via_cholesky(x, absorb="lfactor")
    assert q2 is None
    l2.check()

    # rorthog absorb: (None, None, Q)
    l3, _, q3 = sr.linalg.lq_via_cholesky(x, absorb="rorthog")
    assert l3 is None
    q3.check()


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d0", (3, 5))
@pytest.mark.parametrize("d1", (3, 7))
@pytest.mark.parametrize("dtype", ("complex128", "float64"))
@pytest.mark.parametrize("seed", range(1))
def test_qr_via_cholesky(symmetry, d0, d1, dtype, seed):
    x = sr.utils.get_rand(
        symmetry,
        (d0, d1),
        dtype=dtype,
        seed=seed,
        subsizes="maximal",
    )
    x.check()

    # default absorb: right => (Q, None, R)
    q, s, r = sr.linalg.qr_via_cholesky(x)
    assert s is None
    q.check()
    r.check()

    # roundtrip: Q @ R == x
    y = q @ r
    y.check()
    y.test_allclose(x)

    # rfactor absorb: (None, None, R)
    q2, _, r2 = sr.linalg.qr_via_cholesky(x, absorb="rfactor")
    assert q2 is None
    r2.check()

    # lorthog absorb: (Q, None, None)
    q3, _, r3 = sr.linalg.qr_via_cholesky(x, absorb="lorthog")
    assert r3 is None
    q3.check()


def test_lq_via_cholesky_ar_dispatch():
    """Check that autoray dispatch works for lq_via_cholesky."""
    x = sr.utils.get_rand("Z2", (4, 6), dtype="complex128", subsizes="maximal")
    l, s, q = ar.do("lq_via_cholesky", x)
    assert s is None
    l.check()
    q.check()
    y = l @ q
    y.check()
    y.test_allclose(x)


def test_qr_via_cholesky_ar_dispatch():
    """Check that autoray dispatch works for qr_via_cholesky."""
    x = sr.utils.get_rand("Z2", (4, 6), dtype="complex128", subsizes="maximal")
    q, s, r = ar.do("qr_via_cholesky", x)
    assert s is None
    q.check()
    r.check()
    y = q @ r
    y.check()
    y.test_allclose(x)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 7, 31))
def test_solve(symmetry, d):
    ind = sr.utils.rand_index(symmetry, d)

    a = sr.utils.get_rand(
        symmetry,
        shape=(ind, ind.conj()),
        dtype="complex128",
    )
    b = sr.utils.get_rand(
        symmetry,
        shape=[ind],
        dtype="complex128",
    )
    x = sr.linalg.solve(a, b)
    x.check()
    assert (a @ x).allclose(b)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d0", (2, 7, 31))
@pytest.mark.parametrize("d1", (2, 3, 4))
def test_solve_2d(symmetry, d0, d1):
    ind = sr.utils.rand_index(symmetry, d0)
    ind1 = sr.utils.rand_index(symmetry, d1)

    a = sr.utils.get_rand(
        symmetry,
        shape=(ind, ind.conj()),
        dtype="complex128",
    )
    b = sr.utils.get_rand(
        symmetry,
        shape=(ind, ind1),
        dtype="complex128",
    )
    x = sr.linalg.solve(a, b)
    x.check()
    assert (a @ x).allclose(b)


@pytest.mark.parametrize("symmetry", ("Z2", "U1", "Z2Z2", "U1U1"))
@pytest.mark.parametrize("d", (2, 3, 4, 5, 7))
def test_expm_with_reshape(symmetry, d):
    pytest.importorskip("scipy")

    x = sr.utils.get_rand(
        symmetry,
        (d, d, d, d),
        duals=[0, 0, 1, 1],
        subsizes="equal",
    )
    x_matrix = ar.do("reshape", x, (d**2, d**2))
    # == x_matrix = x.fuse((0, 1), (2, 3))
    xe_matrix = ar.do("scipy.linalg.expm", x_matrix)
    xe_matrix.check()
    xe = ar.do("reshape", xe_matrix, (d, d, d, d))
    xe.check()
    #  == xe = xe_matrix.unfuse_all()
    xe_dense = ar.do(
        "scipy.linalg.expm", x.to_dense().reshape((d**2, d**2))
    ).reshape((d, d, d, d))
    assert_allclose(xe.to_dense(), xe_dense)
