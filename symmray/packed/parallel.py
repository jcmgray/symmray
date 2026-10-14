"""Threading for the packed backend.

The thread count comes from the first set env var of ``SYMMRAY_NUM_THREADS``,
``NUMBA_NUM_THREADS``, ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS`` and
``OPENBLAS_NUM_THREADS``, else it is 1. Change it with :func:`set_num_threads`. With more than one thread:

- numba data kernels (copy, gemm, multiply diagonal, scale) run chunks of
  blocks in parallel, if there is enough work.
- big blocks of a split are decomposed on a thread pool. This needs the
  decompositions to release the GIL, as numpy, scipy and quimb's numba ones do.
- elementwise ops and reductions on big arrays run on chunks of the data, on
  the thread pool.

BLAS is limited to one thread during these, if ``threadpoolctl`` is
installed, so that threads don't multiply. Without it there is a warning on
first use.

Tall QR blocks and wide LQ blocks of a split use quimb's Cholesky QR based
``"qr:auto"`` on one thread. With more threads, or if quimb has no
``"qr:auto"``, they use a threaded TSQR.
"""

import contextlib
import functools
import os

_NUM_THREADS_ENV_VARS = (
    "SYMMRAY_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)


def _default_num_threads():
    for var in _NUM_THREADS_ENV_VARS:
        # OMP_NUM_THREADS can be a list of nested levels, e.g. "4,2"
        value = os.environ.get(var, "").split(",")[0].strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return 1


_num_threads = _default_num_threads()
_pool = None


def get_num_threads():
    """Get the number of threads the packed backend uses."""
    return _num_threads


def set_num_threads(num_threads):
    """Set the number of threads the packed backend uses. Numba kernels use
    at most ``numba.config.NUMBA_NUM_THREADS``.
    """
    global _num_threads, _pool

    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError(f"num_threads must be at least 1, got {num_threads}")
    if _pool is not None:
        _pool.shutdown()
        _pool = None
    _num_threads = num_threads


def get_pool():
    """Get the shared thread pool, with one worker per thread."""
    global _pool

    if _pool is None:
        from concurrent.futures import ThreadPoolExecutor

        _pool = ThreadPoolExecutor(_num_threads, thread_name_prefix="symmray")
    return _pool


def map_threaded(fn, items):
    """``list(map(fn, items))``, on the thread pool if there is more than
    one thread.
    """
    if _num_threads == 1:
        return list(map(fn, items))
    return list(get_pool().map(fn, items))


@functools.cache
def _blas_controller():
    try:
        from threadpoolctl import ThreadpoolController
    except ImportError:
        import warnings

        warnings.warn(
            "threadpoolctl is not installed, so the packed backend can't "
            "limit BLAS to one thread while it runs its own threads. The "
            "cores may be oversubscribed. Install threadpoolctl, or set e.g. "
            "OPENBLAS_NUM_THREADS=1 or MKL_NUM_THREADS=1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return ThreadpoolController()


def limit_blas():
    """Context in which BLAS uses one thread. Does nothing if
    ``threadpoolctl`` is not installed.
    """
    controller = _blas_controller()
    if controller is None:
        return contextlib.nullcontext()
    return controller.limit(limits=1, user_api="blas")


# blocks with at least this much work, m * n * min(m, n), go to the pool
SPLIT_POOL_MIN = 64**3

# QR blocks with at least TSQR_MIN elements, and at least TSQR_ASPECT times
# longer than wide, use tsqr. LQ blocks the same, transposed
TSQR_MIN = 2**20
TSQR_ASPECT = 16

# on one thread, QR blocks at least CHOLQR_ASPECT times longer than wide,
# with m * n**2 at least CHOLQR_MIN, use "qr:auto". Same as quimb's defaults
CHOLQR_MIN = 2**20
CHOLQR_ASPECT = 8

# factors with at least this many elements in total are gathered on the pool
GATHER_MIN = 2**18


def _tsqr(fn, x, kwargs):
    """QR of tall ``x``. Row chunks are decomposed on the thread pool, then
    their stacked R factors. ``fn(chunk, **kwargs)`` should return the
    ``(q, None, r)`` of each chunk.
    """
    import numpy as np

    m, n = x.shape
    # chunks of about 1 MB stay in cache, which helps even on one thread
    nchunks = max(_num_threads, (m * n * x.itemsize) >> 20)
    # chunks need >= n rows for n x n R factors, aim for 4n
    nchunks = max(1, min(nchunks, m // (4 * n)))
    bounds = np.linspace(0, m, nchunks + 1).astype(np.int64)

    def qr_chunk(i):
        # chunks of a transposed block are strided
        chunk = np.ascontiguousarray(x[bounds[i] : bounds[i + 1]])
        return fn(chunk, **kwargs)

    qrs = map_threaded(qr_chunk, range(nchunks))
    q2, _, r = fn(np.concatenate([ri for _, _, ri in qrs]), **kwargs)
    q = np.empty((m, n), dtype=np.result_type(x, q2))

    # Q of each chunk times its n rows of the second Q
    def finish(i):
        qi = q[bounds[i] : bounds[i + 1]]
        np.matmul(qrs[i][0], q2[i * n : (i + 1) * n], out=qi)

    map_threaded(finish, range(nchunks))
    return q, r


@functools.cache
def _has_qr_auto():
    try:
        from quimb.tensor.decomp import qr_auto  # noqa: F401
    except ImportError:
        return False
    return True


def _maybe_tall_qr(fn, block, kwargs):
    """Split a tall QR or wide LQ ``block`` with ``"qr:auto"`` or tsqr. Returns
    None if neither applies.
    """
    from ..linalg_common import Absorb

    m, n = block.shape
    absorb = Absorb.parse(kwargs["absorb"])
    # sides w.r.t. the isometric factor
    if absorb == Absorb.U_sVH:
        long, short = m, n
    elif absorb == Absorb.Us_VH:
        long, short = n, m
    else:
        return None

    if (
        _num_threads == 1
        and long >= CHOLQR_ASPECT * short
        and long * short**2 >= CHOLQR_MIN
        and _has_qr_auto()
    ):
        return fn(block, **{**kwargs, "method": "qr:auto"})

    if long * short < TSQR_MIN or long < TSQR_ASPECT * short:
        return None
    qr_kwargs = {**kwargs, "absorb": Absorb.U_sVH}
    if absorb == Absorb.U_sVH:
        q, r = _tsqr(fn, block, qr_kwargs)
        return q, None, r
    q, r = _tsqr(fn, block.T, qr_kwargs)
    return r.T, None, q.T


def decompose_blocks(fn, blocks, kwargs, sub_max_bonds):
    """Decompose each 2D block with ``fn(block, **kwargs)``. Where
    ``sub_max_bonds`` gives a value, it is also passed as ``max_bond``. Returns
    a list of ``(left, s, right)`` results, with None where the max bond is 0.
    """
    from ..linalg_common import array_split

    # "qr:auto" and tsqr only replace plain qr or lq
    use_tall_qr = (
        fn is array_split
        and kwargs.get("method") == "qr"
        and set(kwargs) <= {"method", "absorb", "stabilized"}
    )
    use_pool = _num_threads > 1
    results = [None] * len(blocks)
    futures = {}
    with limit_blas() if use_pool else contextlib.nullcontext():
        for i, block in enumerate(blocks):
            max_bond = sub_max_bonds[i]
            if max_bond == 0:
                continue
            if max_bond is None:
                split_kwargs = kwargs
                if use_tall_qr:
                    results[i] = _maybe_tall_qr(fn, block, kwargs)
                    if results[i] is not None:
                        continue
            else:
                split_kwargs = {**kwargs, "max_bond": max_bond}
            m, n = block.shape
            if use_pool and (m * n * min(m, n) >= SPLIT_POOL_MIN):
                futures[i] = get_pool().submit(fn, block, **split_kwargs)
            else:
                results[i] = fn(block, **split_kwargs)
        for i, future in futures.items():
            results[i] = future.result()
    return results


def gather(arrays):
    """Concatenate the flattened 2D ``arrays``. With more than one thread
    and enough data, row chunks are copied on the thread pool. Arrays can be
    strided, e.g. transposed factors.
    """
    import numpy as np

    total = sum(x.size for x in arrays)
    if _num_threads == 1 or total < GATHER_MIN:
        return np.concatenate([x.reshape(-1) for x in arrays])

    out = np.empty(total, dtype=np.result_type(*arrays))
    # about four tasks per thread
    share = max(total // (4 * _num_threads), 1)
    tasks = []
    start = 0
    for x in arrays:
        m, n = x.shape
        step = max(1, share // max(n, 1))
        for r0 in range(0, m, step):
            r1 = min(m, r0 + step)
            tasks.append((x, start + r0 * n, r0, r1, n))
        start += x.size

    def copy(task):
        x, pos, r0, r1, n = task
        out[pos : pos + (r1 - r0) * n].reshape(r1 - r0, n)[...] = x[r0:r1]

    map_threaded(copy, tasks)
    return out


# elementwise ops on at least this many elements use the pool
ELEMENTWISE_MIN = 2**20

# elements per chunk. Chunk results are temporaries until copied into the
# output, so small chunks keep the extra memory small
ELEMENTWISE_CHUNK = 2**18

# reductions where reducing the chunk results gives the full result
CHUNK_REDUCTIONS = ("max", "min", "sum", "all", "any")


def use_chunks(x):
    """Whether to work on 1D ``x`` in chunks on the thread pool."""
    import numpy as np

    return (
        _num_threads > 1
        and isinstance(x, np.ndarray)
        and x.size >= ELEMENTWISE_MIN
    )


def _chunk_slices(n):
    return [
        slice(i, min(i + ELEMENTWISE_CHUNK, n))
        for i in range(0, n, ELEMENTWISE_CHUNK)
    ]


def map_elementwise(fn, *xs):
    """``fn(*xs)`` for elementwise ``fn`` and 1D arrays ``xs`` of equal
    size. Runs on chunks on the thread pool if the first array is big enough.
    """
    import numpy as np

    x = xs[0]
    if not use_chunks(x):
        return fn(*xs)

    # one element gives the output dtype. fns not returning a numpy array,
    # e.g. a conversion to another backend, run on the whole array
    probe = fn(*(y[:1] for y in xs))
    if not (isinstance(probe, np.ndarray) and probe.shape == (1,)):
        return fn(*xs)
    out = np.empty(x.size, dtype=probe.dtype)

    def work(s):
        out[s] = fn(*(y[s] for y in xs))

    map_threaded(work, _chunk_slices(x.size))
    return out


def reduce_chunks(name, x):
    """Numpy reduction ``name``, one of ``CHUNK_REDUCTIONS``, of 1D ``x``.
    Chunks are reduced on the thread pool, then their results.
    """
    import numpy as np

    fn = getattr(np, name)
    parts = map_threaded(lambda s: fn(x[s]), _chunk_slices(x.size))
    return fn(np.array(parts))


def norm_chunks(x):
    """2-norm of 1D ``x``. Sums of squares of chunks are found on the
    thread pool.
    """
    import numpy as np

    parts = map_threaded(
        lambda s: np.vdot(x[s], x[s]).real, _chunk_slices(x.size)
    )
    return np.sqrt(np.sum(parts))
