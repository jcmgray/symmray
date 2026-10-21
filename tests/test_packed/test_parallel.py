import os

import autoray as ar
import numpy as np
import pytest

import symmray as sr


@pytest.fixture
def threaded(monkeypatch):
    """Numba kernels, with every parallel path on whatever the size. Yields
    the parallel module, tests set its thread count.
    """
    pytest.importorskip("numba")
    from symmray.packed import parallel
    from symmray.packed.kernels import (
        get_kernels_name,
        load_kernels,
        set_packed_kernels,
    )

    monkeypatch.setattr(load_kernels("numba"), "PAR_MIN", 0)
    for name in (
        "SPLIT_POOL_MIN",
        "TSQR_MIN",
        "GATHER_MIN",
        "ELEMENTWISE_MIN",
    ):
        monkeypatch.setattr(parallel, name, 0)
    # several uneven chunks even for small arrays
    monkeypatch.setattr(parallel, "ELEMENTWISE_CHUNK", 7)
    old_kernels = get_kernels_name()
    old_threads = parallel.get_num_threads()
    set_packed_kernels("numba")
    yield parallel
    parallel.set_num_threads(old_threads)
    set_packed_kernels(old_kernels)


def serial_and_threaded(parallel, fn):
    """The results of ``fn()`` on one thread, then on four."""
    parallel.set_num_threads(1)
    serial = fn()
    parallel.set_num_threads(4)
    return serial, fn()


def assert_same(x, y):
    x.check()
    y.check()
    np.testing.assert_array_equal(x.sectors, y.sectors)
    np.testing.assert_allclose(x._data, y._data, rtol=1e-12, atol=1e-12)


class TestNumThreads:
    @pytest.mark.parametrize(
        "env, expected",
        [
            ({}, None),
            ({"OMP_NUM_THREADS": "3"}, 3),
            ({"OMP_NUM_THREADS": "4,2"}, 4),
            ({"MKL_NUM_THREADS": "6"}, 6),
            ({"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "6"}, 3),
            ({"OPENBLAS_NUM_THREADS": "5"}, 5),
            ({"MKL_NUM_THREADS": "6", "OPENBLAS_NUM_THREADS": "5"}, 6),
            ({"NUMBA_NUM_THREADS": "5", "OMP_NUM_THREADS": "3"}, 5),
            ({"SYMMRAY_NUM_THREADS": "2", "OMP_NUM_THREADS": "3"}, 2),
            ({"SYMMRAY_NUM_THREADS": "", "OMP_NUM_THREADS": "3"}, 3),
            ({"SYMMRAY_NUM_THREADS": "x"}, None),
        ],
    )
    def test_default_from_env(self, monkeypatch, env, expected):
        from symmray.packed import parallel

        for var in parallel._NUM_THREADS_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        if expected is None:
            # nothing set, so every core available
            if hasattr(os, "sched_getaffinity"):
                expected = len(os.sched_getaffinity(0))
            else:
                # e.g. macos and windows
                expected = os.cpu_count() or 1
        assert parallel._default_num_threads() == expected

    def test_set_num_threads(self):
        from symmray.packed import parallel

        old = parallel.get_num_threads()
        try:
            parallel.set_num_threads(3)
            assert parallel.get_num_threads() == 3
            assert parallel.get_pool()._max_workers == 3
            with pytest.raises(ValueError):
                parallel.set_num_threads(0)
        finally:
            parallel.set_num_threads(old)

    def test_blas_pool_needs_threadpoolctl(self, monkeypatch):
        from symmray.packed import parallel

        old = parallel.get_num_threads()
        try:
            parallel.set_num_threads(4)
            monkeypatch.setattr(parallel, "_blas_controller", lambda: object())
            assert parallel.use_blas_pool()
            monkeypatch.setattr(parallel, "_blas_controller", lambda: None)
            assert not parallel.use_blas_pool()
            parallel.set_num_threads(1)
            monkeypatch.setattr(parallel, "_blas_controller", lambda: object())
            assert not parallel.use_blas_pool()
        finally:
            parallel.set_num_threads(old)


class TestKernels:
    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
    def test_transpose(self, threaded, symmetry):
        x = sr.utils.get_rand(symmetry, (5, 6, 7), seed=1).to_packed()
        a, b = serial_and_threaded(threaded, lambda: x.transpose((2, 0, 1)))
        assert_same(a, b)

    @pytest.mark.parametrize("symmetry", ("Z2", "U1", "U1U1"))
    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    def test_tensordot(self, threaded, symmetry, dtype):
        x = sr.utils.get_rand(symmetry, (4, 5, 6), seed=2, dtype=dtype)
        y = sr.utils.get_rand(
            symmetry,
            [x.indices[2].conj(), x.indices[0].conj(), 7],
            seed=3,
            dtype=dtype,
        )
        xp, yp = x.to_packed(), y.to_packed()
        a, b = serial_and_threaded(
            threaded, lambda: xp.tensordot(yp, ((2, 0), (0, 1)))
        )
        assert_same(a, b)

    @pytest.mark.parametrize("blas_pool", (True, False))
    def test_tensordot_big_and_small_blocks(
        self, threaded, monkeypatch, blas_pool
    ):
        pytest.importorskip("threadpoolctl")
        if not blas_pool:
            monkeypatch.setattr(threaded, "_blas_controller", lambda: None)
        # products above and below the plain loop limit
        ix = {0: 20, 1: 1, 2: 16, 3: 2}
        x = sr.utils.get_rand("U1", (ix, ix, ix), seed=5)
        y = sr.utils.get_rand(
            "U1", [x.indices[2].conj(), x.indices[1].conj(), ix], seed=6
        )
        xp, yp = x.to_packed(), y.to_packed()
        a, b = serial_and_threaded(
            threaded, lambda: xp.tensordot(yp, ((2, 1), (0, 1)))
        )
        assert_same(a, b)
        assert_same(b, sr.tensordot(x, y, ((2, 1), (0, 1))).to_packed())

    @pytest.mark.parametrize("power", (1, -1))
    def test_multiply_diagonal(self, threaded, power):
        x = sr.utils.get_rand("U1", (4, 5, 6), seed=4)
        u, s, _ = x.fuse((0, 1), (2,)).svd()
        up, sp = u.to_packed(), s.to_packed()
        a, b = serial_and_threaded(
            threaded, lambda: up.multiply_diagonal(sp, 1, power=power)
        )
        assert_same(a, b)

    def test_multiply_diagonal_several_axes(self, threaded):
        x = sr.utils.get_rand("U1", (4, 5, 6), seed=9).to_packed()
        rng = np.random.default_rng(10)
        vs = [
            sr.PackedVector(
                {
                    c: rng.uniform(0.5, 2.0, size=d)
                    for c, d in x.indices[ax].chargemap.items()
                }
            )
            for ax in (2, 0)
        ]
        a, b = serial_and_threaded(
            threaded, lambda: x.multiply_diagonal(vs, (2, 0))
        )
        assert_same(a, b)

    def test_fermionic_phases(self, threaded):
        x = sr.utils.get_rand(
            "U1", (4, 5, 6), seed=5, fermionic=True, charge=1, label="x"
        ).to_packed()
        a, b = serial_and_threaded(
            threaded,
            lambda: x.conj(phase_dual=True).transpose((1, 2, 0)),
        )
        assert_same(a, b)


class TestTsqr:
    @pytest.mark.parametrize("num_threads", (1, 4))
    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    def test_matches_qr(self, threaded, num_threads, dtype):
        from symmray.linalg_common import array_split

        rng = np.random.default_rng(6)
        x = rng.standard_normal((2000, 12))
        if dtype == "complex128":
            x = x + 1j * rng.standard_normal((2000, 12))
        threaded.set_num_threads(num_threads)
        kwargs = {"method": "qr", "absorb": 1}
        q, r = threaded._tsqr(array_split, x, kwargs)
        q_expected, _, r_expected = array_split(x, **kwargs)
        np.testing.assert_allclose(r, r_expected, atol=1e-12)
        np.testing.assert_allclose(q, q_expected, atol=1e-12)
        np.testing.assert_allclose(q.conj().T @ q, np.eye(12), atol=1e-12)

    @pytest.mark.parametrize("num_threads", (1, 4))
    @pytest.mark.parametrize("method, absorb", [("qr", 1), ("lq", -1)])
    @pytest.mark.parametrize("fermionic", (False, True))
    def test_split(
        self, threaded, monkeypatch, num_threads, method, absorb, fermionic
    ):
        calls = []
        tsqr = threaded._tsqr

        def record(*args, **kwargs):
            calls.append(None)
            return tsqr(*args, **kwargs)

        monkeypatch.setattr(threaded, "_tsqr", record)
        long, short = {0: 200, 1: 160}, {0: 5, 1: 4}
        shape = [long, short] if method == "qr" else [short, long]
        x = sr.utils.get_rand(
            "U1", shape, seed=7, duals=(False, True), fermionic=fermionic
        )
        threaded.set_num_threads(num_threads)
        opts = {"method": "qr", "absorb": absorb}
        left, _, right = x.to_packed()._split(**opts)
        el, _, er = x._split(**opts)
        assert calls
        left.to_blocksparse().test_allclose(el)
        right.to_blocksparse().test_allclose(er)

    @pytest.mark.parametrize(
        "num_threads, has_qr_auto, blas_pool, expected",
        [
            (1, True, True, "qr:auto"),
            (1, False, True, "tsqr"),
            (4, True, True, "tsqr"),
            (4, True, False, "qr:auto"),
        ],
    )
    @pytest.mark.parametrize("method, absorb", [("qr", 1), ("lq", -1)])
    def test_tall_qr_route(
        self,
        threaded,
        monkeypatch,
        num_threads,
        has_qr_auto,
        blas_pool,
        expected,
        method,
        absorb,
    ):
        decomp = pytest.importorskip("quimb.tensor.decomp")
        if has_qr_auto and not threaded._has_qr_auto():
            pytest.skip("quimb has no qr:auto")
        if blas_pool:
            pytest.importorskip("threadpoolctl")
        else:
            monkeypatch.setattr(threaded, "_blas_controller", lambda: None)
        monkeypatch.setattr(threaded, "CHOLQR_MIN", 0)
        monkeypatch.setattr(threaded, "_has_qr_auto", lambda: has_qr_auto)

        used = []
        tsqr = threaded._tsqr

        def record_tsqr(*args, **kwargs):
            used.append("tsqr")
            return tsqr(*args, **kwargs)

        monkeypatch.setattr(threaded, "_tsqr", record_tsqr)
        if has_qr_auto:
            qr_auto = decomp._SPLIT_FNS["qr:auto"]

            # quimb only passes absorb to drivers with an absorb parameter
            def record_qr_auto(x, absorb, **kwargs):
                used.append("qr:auto")
                return qr_auto(x, absorb=absorb, **kwargs)

            monkeypatch.setitem(decomp._SPLIT_FNS, "qr:auto", record_qr_auto)

        long, short = {0: 200, 1: 160}, {0: 5, 1: 4}
        shape = [long, short] if method == "qr" else [short, long]
        x = sr.utils.get_rand("U1", shape, seed=8, duals=(False, True))
        threaded.set_num_threads(num_threads)
        opts = {"method": "qr", "absorb": absorb}
        left, _, right = x.to_packed()._split(**opts)
        el, _, er = x._split(**opts)
        assert set(used) == {expected}
        left.to_blocksparse().test_allclose(el)
        right.to_blocksparse().test_allclose(er)


class TestGather:
    def test_matches_concatenate(self, threaded):
        rng = np.random.default_rng(8)
        arrays = [
            rng.standard_normal((50, 7)),
            rng.standard_normal((9, 30)).T,
            rng.standard_normal((1, 3)),
            rng.standard_normal((0, 4)),
        ]
        threaded.set_num_threads(4)
        np.testing.assert_array_equal(
            threaded.gather(arrays),
            np.concatenate([x.reshape(-1) for x in arrays]),
        )


class TestElementwise:
    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    def test_unary_and_binary(self, threaded, dtype):
        x = sr.utils.get_rand("U1", (4, 5, 6), seed=11, dtype=dtype)
        xp = x.to_packed()
        for fn in (
            lambda y: y / 3,
            lambda y: 2 * y,
            lambda y: -y,
            lambda y: y.abs(),
            lambda y: y.isfinite(),
            lambda y: y + y,
            lambda y: y * y,
        ):
            a, b = serial_and_threaded(threaded, lambda fn=fn: fn(xp))
            assert_same(a, b)
            assert b._data.dtype == fn(x).to_packed()._data.dtype

    def test_conversion_runs_whole(self, threaded):
        x = sr.utils.get_rand("U1", (4, 5, 6), seed=14).to_packed()
        threaded.set_num_threads(4)
        x.apply_to_arrays(ar.lazy.array)
        assert isinstance(x._data, ar.lazy.LazyArray)

    @pytest.mark.parametrize("name", ("max", "min", "sum", "all", "any"))
    def test_reductions(self, threaded, name):
        x = sr.utils.get_rand("U1", (4, 5, 6), seed=12).to_packed()
        if name in ("all", "any"):
            x = x.isfinite()
        a, b = serial_and_threaded(threaded, getattr(x, name))
        assert b == pytest.approx(a)

    @pytest.mark.parametrize("dtype", ("float64", "complex128"))
    def test_norm(self, threaded, dtype):
        x = sr.utils.get_rand("U1", (4, 5, 6), seed=13, dtype=dtype)
        a, b = serial_and_threaded(threaded, x.to_packed().norm)
        assert b == pytest.approx(a)
        assert b == pytest.approx(x.norm())
