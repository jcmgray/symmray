"""Packed kernels compiled with numba. Plans are just arrays per block, and the
kernels loop over blocks and elements directly.
"""

import numba
import numpy as np

NAME = "numba"


@numba.njit(cache=True, nogil=True)
def _copy_blocks(
    src,
    dst,
    src_base,
    src_strides,
    dst_base,
    dst_strides,
    shapes,
    scales,
):
    n, nd = shapes.shape
    idx = np.zeros(max(nd, 1), dtype=np.int64)
    use_scales = scales.shape[0] > 0
    for b in range(n):
        size = 1
        for d in range(nd):
            size *= shapes[b, d]
        if size == 0:
            continue
        si = src_base[b]
        di = dst_base[b]
        if nd == 0:
            if use_scales:
                dst[di] = src[si] * scales[b]
            else:
                dst[di] = src[si]
            continue
        for d in range(nd):
            idx[d] = 0
        n_in = shapes[b, nd - 1]
        s_in = src_strides[b, nd - 1]
        d_in = dst_strides[b, nd - 1]
        while True:
            if use_scales:
                f = scales[b]
                for k in range(n_in):
                    dst[di + k * d_in] = src[si + k * s_in] * f
            else:
                for k in range(n_in):
                    dst[di + k * d_in] = src[si + k * s_in]
            # step the outer dims like a counter, carrying into earlier dims
            d = nd - 2
            while d >= 0:
                idx[d] += 1
                si += src_strides[b, d]
                di += dst_strides[b, d]
                if idx[d] < shapes[b, d]:
                    break
                si -= src_strides[b, d] * shapes[b, d]
                di -= dst_strides[b, d] * shapes[b, d]
                idx[d] = 0
                d -= 1
            if d < 0:
                break


@numba.njit(cache=True, nogil=True)
def _coalesce_row(src_strides, dst_strides, shapes, b):
    """Merge neighbouring dims of entry ``b`` inplace, where they are
    contiguous in both source and destination. This makes the innermost loop as
    long as possible. Merged away dims become leading size 1 dims.
    """
    nd = shapes.shape[1]
    # merged dims are written from the end, slot k onwards
    k = nd
    for d in range(nd - 1, -1, -1):
        dim = shapes[b, d]
        if dim == 1:
            continue
        if (
            k < nd
            and src_strides[b, d] == src_strides[b, k] * shapes[b, k]
            and dst_strides[b, d] == dst_strides[b, k] * shapes[b, k]
        ):
            shapes[b, k] *= dim
            continue
        k -= 1
        src_strides[b, k] = src_strides[b, d]
        dst_strides[b, k] = dst_strides[b, d]
        shapes[b, k] = dim
    if k == nd:
        # a single element
        k -= 1
        src_strides[b, k] = 1
        dst_strides[b, k] = 1
        shapes[b, k] = 1
    for d in range(k):
        src_strides[b, d] = 0
        dst_strides[b, d] = 0
        shapes[b, d] = 1


@numba.njit(cache=True, nogil=True)
def _coalesce(src_strides, dst_strides, shapes):
    src_strides = src_strides.copy()
    dst_strides = dst_strides.copy()
    shapes = shapes.copy()
    for b in range(shapes.shape[0]):
        _coalesce_row(src_strides, dst_strides, shapes, b)
    return src_strides, dst_strides, shapes


def plan_copy(
    src_base,
    src_strides,
    dst_base,
    dst_strides,
    shapes,
    dst_size,
    fill_zero,
    coalesced=False,
):
    c = np.ascontiguousarray
    src_strides = c(src_strides, dtype=np.int64)
    dst_strides = c(dst_strides, dtype=np.int64)
    shapes = c(shapes, dtype=np.int64)
    if shapes.shape[1] > 1 and not coalesced:
        src_strides, dst_strides, shapes = _coalesce(
            src_strides, dst_strides, shapes
        )
    return (
        c(src_base, dtype=np.int64),
        src_strides,
        c(dst_base, dtype=np.int64),
        dst_strides,
        shapes,
        int(dst_size),
        bool(fill_zero),
    )


_NO_SCALES = {}


def run_copy(plan, src, scales=None):
    src_base, src_strides, dst_base, dst_strides, shapes, size, fill = plan
    if fill:
        dst = np.zeros(size, dtype=src.dtype)
    else:
        dst = np.empty(size, dtype=src.dtype)
    if scales is None:
        try:
            scales = _NO_SCALES[src.dtype]
        except KeyError:
            scales = _NO_SCALES[src.dtype] = np.zeros(0, dtype=src.dtype)
    else:
        scales = np.asarray(scales, dtype=src.dtype)
    _copy_blocks(
        src,
        dst,
        src_base,
        src_strides,
        dst_base,
        dst_strides,
        shapes,
        scales,
    )
    return dst


@numba.njit(cache=True, nogil=True)
def _muldiag(src, dst, offsets, pre, mid, post, vbase, v, power):
    n = pre.shape[0]
    for b in range(n):
        o = offsets[b]
        vo = vbase[b]
        m = mid[b]
        q = post[b]
        if q == 1:
            # the axis is the last one, so the vector runs contiguously
            for i in range(pre[b]):
                r = o + i * m
                if power == 1:
                    for j in range(m):
                        dst[r + j] = src[r + j] * v[vo + j]
                else:
                    for j in range(m):
                        dst[r + j] = src[r + j] / v[vo + j]
            continue
        for i in range(pre[b]):
            for j in range(m):
                vj = v[vo + j]
                r = o + (i * m + j) * q
                if power == 1:
                    for k in range(q):
                        dst[r + k] = src[r + k] * vj
                else:
                    for k in range(q):
                        dst[r + k] = src[r + k] / vj


def plan_muldiag(offsets, shapes, axis, vbase):
    c = np.ascontiguousarray
    return (
        c(offsets[:-1], dtype=np.int64),
        c(shapes[:, :axis].prod(axis=1), dtype=np.int64),
        c(shapes[:, axis], dtype=np.int64),
        c(shapes[:, axis + 1 :].prod(axis=1), dtype=np.int64),
        c(vbase, dtype=np.int64),
    )


def run_muldiag(plan, src, v, power):
    offsets, pre, mid, post, vbase = plan
    # true division always gives a floating dtype
    dtype = (
        np.result_type(src, v) if power == 1 else np.result_type(src, v, 1.0)
    )
    if src.dtype != dtype:
        src = src.astype(dtype)
    if v.dtype != dtype:
        v = v.astype(dtype)
    dst = np.empty(src.shape[0], dtype=dtype)
    _muldiag(src, dst, offsets, pre, mid, post, vbase, v, int(power))
    return dst


# up to this many multiply-adds, a plain loop is faster than calling blas
_SMALL_GEMM = 256


@numba.njit(cache=True, nogil=True)
def _gemms(a, b, out, a_off, rows, inner, b_off, cols, c_off):
    for i in range(a_off.shape[0]):
        r = rows[i]
        k = inner[i]
        c = cols[i]
        if r * k * c > _SMALL_GEMM:
            np.dot(
                a[a_off[i] : a_off[i] + r * k].reshape(r, k),
                b[b_off[i] : b_off[i] + k * c].reshape(k, c),
                out[c_off[i] : c_off[i] + r * c].reshape(r, c),
            )
            continue
        # out[x, z] = sum_y a[x, y] * b[y, z], inner loop contiguous
        ai = a_off[i]
        bi = b_off[i]
        ci = c_off[i]
        for x in range(r):
            for z in range(c):
                out[ci + x * c + z] = 0
            for y in range(k):
                f = a[ai + x * k + y]
                for z in range(c):
                    out[ci + x * c + z] += f * b[bi + y * c + z]


def plan_gemm(a_off, rows, inner, b_off, cols, c_off):
    c = np.ascontiguousarray
    return tuple(
        c(x, dtype=np.int64) for x in (a_off, rows, inner, b_off, cols, c_off)
    )


def run_gemm(plan, a, b, out_size):
    dtype = np.result_type(a, b)
    if dtype.kind not in "fc":
        # numba's dot only supports floating dtypes
        from .numpy_kernels import run_gemm as run_gemm_numpy

        return run_gemm_numpy(
            list(zip(*(x.tolist() for x in plan))), a, b, out_size
        )
    if a.dtype != dtype:
        a = a.astype(dtype)
    if b.dtype != dtype:
        b = b.astype(dtype)
    out = np.empty(out_size, dtype=dtype)
    _gemms(a, b, out, *plan)
    return out


@numba.njit(cache=True, nogil=True)
def _row_keys(rows, radices):
    n, m = rows.shape
    keys = np.zeros(n, dtype=np.int64)
    for i in range(n):
        key = 0
        for j in range(m):
            key = key * radices[j] + rows[i, j]
        keys[i] = key
    return keys


# above this many possible keys, sort by columns rather than by single keys
_MAX_KEY_RANGE = 1 << 22


@numba.njit(cache=True, nogil=True)
def _key_range(radices):
    total = 1
    for r in radices:
        total *= max(r, 1)
        if total > _MAX_KEY_RANGE:
            return -1
    return total


@numba.njit(cache=True, nogil=True)
def _counting_argsort(keys, key_range):
    """Stable argsort of integer ``keys`` below ``key_range``."""
    n = keys.shape[0]
    counts = np.zeros(key_range + 1, dtype=np.int64)
    for i in range(n):
        counts[keys[i] + 1] += 1
    for c in range(key_range):
        counts[c + 1] += counts[c]
    order = np.empty(n, dtype=np.int64)
    for i in range(n):
        k = keys[i]
        order[counts[k]] = i
        counts[k] += 1
    return order


@numba.njit(cache=True, nogil=True)
def _argsort_rows(rows, radices):
    """Stable lexicographic argsort of ``rows``, small non-negative integers
    with column ``j`` below ``radices[j]``. Uses one counting sort if the
    combined keys have a small range, else one counting sort per column, last
    column first.
    """
    n, m = rows.shape
    key_range = _key_range(radices)
    # a table over all keys is cheap if not much bigger than the rows
    if key_range > 0 and key_range <= 4 * n + 1024:
        return _counting_argsort(_row_keys(rows, radices), key_range)
    order = np.arange(n)
    tmp = np.empty(n, dtype=np.int64)
    for j in range(m - 1, -1, -1):
        r = radices[j]
        if r <= 1:
            continue
        counts = np.zeros(r + 1, dtype=np.int64)
        for i in range(n):
            counts[rows[i, j] + 1] += 1
        for c in range(r):
            counts[c + 1] += counts[c]
        for idx in range(n):
            i = order[idx]
            c = rows[i, j]
            tmp[counts[c]] = i
            counts[c] += 1
        order, tmp = tmp, order
    return order


@numba.njit(cache=True, nogil=True)
def _unique_rows(rows, radices):
    """Find the sorted unique rows. Returns the first original position of
    each, and the unique index of each original row.
    """
    n, m = rows.shape
    first = np.empty(n, dtype=np.int64)
    inv = np.empty(n, dtype=np.int64)
    u = 0
    key_range = _key_range(radices)
    if key_range > 0 and key_range <= 4 * n + 1024:
        # first occurrence of each key, scanned in key order
        keys = _row_keys(rows, radices)
        slot = np.full(key_range, -1, dtype=np.int64)
        for i in range(n):
            if slot[keys[i]] < 0:
                slot[keys[i]] = i
        for k in range(key_range):
            i = slot[k]
            if i >= 0:
                first[u] = i
                slot[k] = u
                u += 1
        for i in range(n):
            inv[i] = slot[keys[i]]
        return first[:u], inv

    order = _argsort_rows(rows, radices)
    for idx in range(n):
        i = order[idx]
        new = idx == 0
        if not new:
            p = order[idx - 1]
            for j in range(m):
                if rows[i, j] != rows[p, j]:
                    new = True
                    break
        if new:
            first[u] = i
            u += 1
        inv[i] = u - 1
    return first[:u], inv


@numba.njit(cache=True, nogil=True)
def _fuse_group(sub, radices, signed, voff, modulus, sizes):
    nb, m = sub.shape
    k = signed.shape[1]

    first, inv = _unique_rows(sub, radices)
    u = first.shape[0]

    # new charge value and size of each unique subsector
    usub = np.empty((u, m), dtype=np.int64)
    values = np.zeros((u, k), dtype=np.int64)
    usize = np.ones(u, dtype=np.int64)
    for s in range(u):
        i = first[s]
        for j in range(m):
            c = sub[i, j]
            usub[s, j] = c
            for q in range(k):
                values[s, q] += signed[voff[j] + c, q]
            usize[s] *= sizes[voff[j] + c]
        if modulus > 0:
            for q in range(k):
                values[s, q] %= modulus

    # sort new charges by value, subsectors stay sorted within each
    ckeys = np.zeros(u, dtype=np.int64)
    key_range = 1
    if u > 0:
        for q in range(k):
            lo = values[:, q].min()
            span = values[:, q].max() - lo + 1
            if key_range > (1 << 62) // span:
                # the combined keys could overflow
                return _fuse_group_failed(nb, m, k)
            key_range *= span
            for s in range(u):
                ckeys[s] = ckeys[s] * span + values[s, q] - lo
    if key_range <= 4 * u + 1024:
        corder = _counting_argsort(ckeys, key_range)
    else:
        corder = np.argsort(ckeys, kind="mergesort")
    charge_of_sub = np.empty(u, dtype=np.int64)
    charge_values = np.empty((u, k), dtype=np.int64)
    totals = np.zeros(u, dtype=np.int64)
    owner = np.empty(u, dtype=np.int64)
    offsets = np.empty(u, dtype=np.int64)
    sorted_sizes = np.empty(u, dtype=np.int64)
    subkeys = np.empty((u, m), dtype=np.int64)
    sub_offsets = np.empty(u, dtype=np.int64)
    nc = 0
    for idx in range(u):
        s = corder[idx]
        if idx == 0 or ckeys[s] != ckeys[corder[idx - 1]]:
            for q in range(k):
                charge_values[nc, q] = values[s, q]
            nc += 1
        c = nc - 1
        charge_of_sub[s] = c
        owner[idx] = c
        offsets[idx] = totals[c]
        sub_offsets[s] = totals[c]
        sorted_sizes[idx] = usize[s]
        totals[c] += usize[s]
        for j in range(m):
            subkeys[idx, j] = usub[s, j]

    block_charge = np.empty(nb, dtype=np.int64)
    block_offset = np.empty(nb, dtype=np.int64)
    for i in range(nb):
        block_charge[i] = charge_of_sub[inv[i]]
        block_offset[i] = sub_offsets[inv[i]]
    charge_starts = np.zeros(nc + 1, dtype=np.int64)
    for idx in range(u):
        charge_starts[owner[idx] + 1] += 1
    for c in range(nc):
        charge_starts[c + 1] += charge_starts[c]
    return (
        block_charge,
        block_offset,
        charge_values[:nc],
        totals[:nc],
        subkeys,
        charge_starts,
        offsets,
        sorted_sizes,
        True,
    )


@numba.njit(cache=True, nogil=True)
def _fuse_group_failed(nb, m, k):
    e = np.empty(0, dtype=np.int64)
    return (
        e,
        e,
        np.empty((0, k), dtype=np.int64),
        e,
        np.empty((0, m), dtype=np.int64),
        e,
        e,
        e,
        False,
    )


def fuse_group(sub, radices, signed, voff, modulus, sizes):
    c = np.ascontiguousarray
    *out, ok = _fuse_group(
        c(sub, dtype=np.int64),
        c(radices, dtype=np.int64),
        c(signed, dtype=np.int64),
        c(voff, dtype=np.int64),
        int(modulus),
        c(sizes, dtype=np.int64),
    )
    if not ok:
        from .numpy_kernels import fuse_group as fuse_group_numpy

        return fuse_group_numpy(sub, radices, signed, voff, modulus, sizes)
    return tuple(out)


@numba.njit(cache=True, nogil=True)
def _fuse_assemble(
    cols,
    suboffs,
    radices,
    new_sizes,
    noff,
    shapes,
    perm,
    glens,
    src_base,
    panel_last,
):
    nb, nnew = cols.shape
    nd = shapes.shape[1]

    first, block_of = _unique_rows(cols, radices)
    nnb = first.shape[0]

    new_sectors = np.empty((nnb, nnew), dtype=np.int64)
    new_shapes = np.empty((nnb, nnew), dtype=np.int64)
    new_offsets = np.zeros(nnb + 1, dtype=np.int64)
    new_strides = np.empty((nnb, nnew), dtype=np.int64)
    for j in range(nnb):
        size = 1
        for a in range(nnew):
            c = cols[first[j], a]
            new_sectors[j, a] = c
            new_shapes[j, a] = new_sizes[noff[a] + c]
            size *= new_shapes[j, a]
        new_offsets[j + 1] = new_offsets[j] + size
        stride = 1
        for a in range(nnew - 1, -1, -1):
            new_strides[j, a] = stride
            stride *= new_shapes[j, a]

    iter_shapes = np.empty((nb, nd), dtype=np.int64)
    src_strides = np.empty((nb, nd), dtype=np.int64)
    dst_strides = np.empty((nb, nd), dtype=np.int64)
    dst_base = np.empty(nb, dtype=np.int64)
    ast = np.empty(nnew, dtype=np.int64)
    cst = np.empty(nd, dtype=np.int64)
    total_src = 0
    for i in range(nb):
        j = block_of[i]
        stride = 1
        for ax in range(nd - 1, -1, -1):
            cst[ax] = stride
            stride *= shapes[i, ax]
        total_src += stride
        for d in range(nd):
            iter_shapes[i, d] = shapes[i, perm[d]]
            src_strides[i, d] = cst[perm[d]]
        for a in range(nnew):
            ast[a] = new_strides[j, a]
        base = new_offsets[j]
        if panel_last:
            # in a panel the last axis has the subsector's size, and the panel
            # starts after earlier panels of the block
            sub_size = 1
            for d in range(nd - glens[nnew - 1], nd):
                sub_size *= iter_shapes[i, d]
            rest = 1
            for a in range(nnew - 1):
                rest *= new_shapes[j, a]
            for a in range(nnew - 1):
                ast[a] = ast[a] // new_shapes[j, nnew - 1] * sub_size
            ast[nnew - 1] = 1
            base += suboffs[i, nnew - 1] * rest
            for a in range(nnew - 1):
                base += suboffs[i, a] * ast[a]
        else:
            for a in range(nnew):
                base += suboffs[i, a] * ast[a]
        dst_base[i] = base
        d = 0
        for a in range(nnew):
            d += glens[a]
            stride = ast[a]
            for dd in range(d - 1, d - glens[a] - 1, -1):
                dst_strides[i, dd] = stride
                stride *= iter_shapes[i, dd]
        if nd > 1:
            _coalesce_row(src_strides, dst_strides, iter_shapes, i)

    return (
        new_sectors,
        new_shapes,
        new_offsets,
        src_base,
        src_strides,
        dst_base,
        dst_strides,
        iter_shapes,
        total_src != new_offsets[nnb],
    )


def fuse_assemble(
    cols,
    suboffs,
    radices,
    new_sizes,
    noff,
    shapes,
    perm,
    glens,
    src_base,
    panel_last,
):
    c = np.ascontiguousarray
    return _fuse_assemble(
        c(cols, dtype=np.int64),
        c(suboffs, dtype=np.int64),
        c(radices, dtype=np.int64),
        c(new_sizes, dtype=np.int64),
        c(noff, dtype=np.int64),
        c(shapes, dtype=np.int64),
        c(perm, dtype=np.int64),
        c(glens, dtype=np.int64),
        c(src_base, dtype=np.int64),
        bool(panel_last),
    )


@numba.njit(cache=True, nogil=True)
def _tensordot_align(sub_a, sub_b, radices):
    keys_a = _row_keys(sub_a, radices)
    nb, m = sub_b.shape
    valid_b = np.ones(nb, dtype=np.bool_)
    for i in range(nb):
        for j in range(m):
            if sub_b[i, j] < 0:
                valid_b[i] = False
    keys_b = _row_keys(np.where(sub_b >= 0, sub_b, 0), radices)

    key_range = _key_range(radices)
    if key_range > 0 and key_range <= 4 * (keys_a.shape[0] + nb) + 1024:
        # mark which keys are present on each side
        in_a = np.zeros(key_range, dtype=np.bool_)
        in_b = np.zeros(key_range, dtype=np.bool_)
        for i in range(keys_a.shape[0]):
            in_a[keys_a[i]] = True
        for i in range(nb):
            if valid_b[i]:
                in_b[keys_b[i]] = True
        keep_a = np.empty(keys_a.shape[0], dtype=np.bool_)
        for i in range(keys_a.shape[0]):
            keep_a[i] = in_b[keys_a[i]]
        keep_b = np.empty(nb, dtype=np.bool_)
        for i in range(nb):
            keep_b[i] = valid_b[i] and in_a[keys_b[i]]
        return keep_a, keep_b

    sorted_a = np.sort(keys_a)
    sorted_b = np.sort(keys_b[valid_b])
    keep_a = np.empty(keys_a.shape[0], dtype=np.bool_)
    for i in range(keys_a.shape[0]):
        p = np.searchsorted(sorted_b, keys_a[i])
        keep_a[i] = p < sorted_b.shape[0] and sorted_b[p] == keys_a[i]
    keep_b = np.empty(nb, dtype=np.bool_)
    for i in range(nb):
        p = np.searchsorted(sorted_a, keys_b[i])
        keep_b[i] = (
            valid_b[i] and p < sorted_a.shape[0] and sorted_a[p] == keys_b[i]
        )
    return keep_a, keep_b


def tensordot_align(sub_a, sub_b, radices):
    c = np.ascontiguousarray
    return _tensordot_align(
        c(sub_a, dtype=np.int64),
        c(sub_b, dtype=np.int64),
        c(radices, dtype=np.int64),
    )


@numba.njit(cache=True, nogil=True)
def _row_charges(counts, starts):
    # the charge of each subsector row
    owner = np.empty(starts[-1] + counts[-1], dtype=np.int64)
    for c in range(counts.shape[0]):
        for i in range(starts[c], starts[c] + counts[c]):
            owner[i] = c
    return owner


@numba.njit(cache=True, nogil=True)
def _soff_radices(soff):
    radices = np.empty(soff.shape[0] - 1, dtype=np.int64)
    for j in range(radices.shape[0]):
        radices[j] = soff[j + 1] - soff[j]
    return radices


@numba.njit(cache=True, nogil=True)
def _sorted_pairs(
    cl,
    cr,
    l_counts,
    l_starts,
    l_keys,
    l_soff,
    r_counts,
    r_starts,
    r_keys,
    r_soff,
    num_out,
):
    """Get the pair, left row and right row of each output block, in sorted
    output order. Assumes each left charge (or right charge, if there is no
    left side) is in at most one pair. Then the output is the left rows in
    order, each with the right rows of its pair in order. Returns ``ok=False``
    if the assumption fails.
    """
    e_pair = np.empty(num_out, dtype=np.int64)
    e_l = np.empty(num_out, dtype=np.int64)
    e_r = np.empty(num_out, dtype=np.int64)
    has_left = l_keys.shape[1] > 0

    r_lex = _argsort_rows(r_keys, _soff_radices(r_soff))
    r_owner = _row_charges(r_counts, r_starts)
    e = 0
    if not has_left:
        pair_of = np.full(r_counts.shape[0], -1, dtype=np.int64)
        for p in range(cr.shape[0]):
            if pair_of[cr[p]] >= 0:
                return e_pair, e_l, e_r, False
            pair_of[cr[p]] = p
        for idx in range(r_lex.shape[0]):
            rr = r_lex[idx]
            p = pair_of[r_owner[rr]]
            if p >= 0:
                e_pair[e] = p
                e_l[e] = 0
                e_r[e] = rr
                e += 1
        return e_pair, e_l, e_r, True

    pair_of = np.full(l_counts.shape[0], -1, dtype=np.int64)
    for p in range(cl.shape[0]):
        if pair_of[cl[p]] >= 0:
            return e_pair, e_l, e_r, False
        pair_of[cl[p]] = p

    # right rows sorted within each charge
    r_seq = np.empty(r_lex.shape[0], dtype=np.int64)
    pos = r_starts.copy()
    for idx in range(r_lex.shape[0]):
        rr = r_lex[idx]
        r_seq[pos[r_owner[rr]]] = rr
        pos[r_owner[rr]] += 1

    l_lex = _argsort_rows(l_keys, _soff_radices(l_soff))
    l_owner = _row_charges(l_counts, l_starts)
    for idx in range(l_lex.shape[0]):
        rl = l_lex[idx]
        p = pair_of[l_owner[rl]]
        if p < 0:
            continue
        c = cr[p]
        for k in range(r_starts[c], r_starts[c] + r_counts[c]):
            e_pair[e] = p
            e_l[e] = rl
            e_r[e] = r_seq[k]
            e += 1
    return e_pair, e_l, e_r, True


@numba.njit(cache=True, nogil=True)
def _tensordot_entries(
    cl,
    a_off,
    inner,
    cr,
    b_off,
    l_counts,
    l_starts,
    l_keys,
    l_offsets,
    l_sizes,
    l_subsizes,
    l_soff,
    r_counts,
    r_starts,
    r_keys,
    r_offsets,
    r_sizes,
    r_subsizes,
    r_soff,
    radices,
):
    num_pairs = cl.shape[0]
    ml = l_keys.shape[1]
    mr = r_keys.shape[1]
    m = ml + mr
    num_out = 0
    for p in range(num_pairs):
        num_out += l_counts[cl[p]] * r_counts[cr[p]]

    e_pair, e_l, e_r, ok = _sorted_pairs(
        cl,
        cr,
        l_counts,
        l_starts,
        l_keys,
        l_soff,
        r_counts,
        r_starts,
        r_keys,
        r_soff,
        num_out,
    )
    if not ok:
        # every combination of each pair, then sort
        e = 0
        for p in range(num_pairs):
            for il in range(l_counts[cl[p]]):
                for ir in range(r_counts[cr[p]]):
                    e_pair[e] = p
                    e_l[e] = l_starts[cl[p]] + il
                    e_r[e] = r_starts[cr[p]] + ir
                    e += 1
        keys = np.empty((num_out, m), dtype=np.int64)
        for e in range(num_out):
            for j in range(ml):
                keys[e, j] = l_keys[e_l[e], j]
            for j in range(mr):
                keys[e, ml + j] = r_keys[e_r[e], j]
        order = _argsort_rows(keys, radices)
        e_pair = e_pair[order]
        e_l = e_l[order]
        e_r = e_r[order]

    out_sectors = np.empty((num_out, m), dtype=np.int64)
    out_shapes = np.empty((num_out, m), dtype=np.int64)
    out_offsets = np.zeros(num_out + 1, dtype=np.int64)
    nl = l_soff[ml]
    used = np.zeros(nl + r_soff[mr], dtype=np.bool_)
    g_a_off = np.empty(num_out, dtype=np.int64)
    g_rows = np.empty(num_out, dtype=np.int64)
    g_inner = np.empty(num_out, dtype=np.int64)
    g_b_off = np.empty(num_out, dtype=np.int64)
    g_cols = np.empty(num_out, dtype=np.int64)
    for e in range(num_out):
        p = e_pair[e]
        rl = e_l[e]
        rr = e_r[e]
        k = inner[p]
        size = 1
        for j in range(ml):
            c = l_keys[rl, j]
            out_sectors[e, j] = c
            out_shapes[e, j] = l_subsizes[l_soff[j] + c]
            used[l_soff[j] + c] = True
            size *= out_shapes[e, j]
        for j in range(mr):
            c = r_keys[rr, j]
            out_sectors[e, ml + j] = c
            out_shapes[e, ml + j] = r_subsizes[r_soff[j] + c]
            used[nl + r_soff[j] + c] = True
            size *= out_shapes[e, ml + j]
        out_offsets[e + 1] = out_offsets[e] + size
        g_a_off[e] = a_off[p] + l_offsets[rl] * k
        g_rows[e] = l_sizes[rl]
        g_inner[e] = k
        g_b_off[e] = b_off[p] + r_offsets[rr] * k
        g_cols[e] = r_sizes[rr]

    return (
        out_sectors,
        out_shapes,
        out_offsets,
        used,
        g_a_off,
        g_rows,
        g_inner,
        g_b_off,
        g_cols,
        out_offsets[:-1].copy(),
    )


def tensordot_entries(cl, a_off, inner, cr, b_off, left, right, radices):
    c = np.ascontiguousarray
    return _tensordot_entries(
        c(cl, dtype=np.int64),
        c(a_off, dtype=np.int64),
        c(inner, dtype=np.int64),
        c(cr, dtype=np.int64),
        c(b_off, dtype=np.int64),
        *(c(x, dtype=np.int64) for x in left),
        *(c(x, dtype=np.int64) for x in right),
        c(radices, dtype=np.int64),
    )


@numba.njit(cache=True, nogil=True)
def _scale_blocks(src, dst, offsets, scales):
    for b in range(scales.shape[0]):
        f = scales[b]
        for i in range(offsets[b], offsets[b + 1]):
            dst[i] = src[i] * f


def run_scale(src, offsets, scales):
    dst = np.empty_like(src)
    _scale_blocks(src, dst, offsets, np.asarray(scales, dtype=src.dtype))
    return dst


@numba.njit(cache=True, nogil=True)
def _pattern_phase(m, axes, flips):
    # bit ax of m is the parity of axis ax. Count swapped pairs of odd axes,
    # plus odd flipped axes

    count = 0
    ndim = axes.shape[0]
    for a in range(ndim):
        if (m >> axes[a]) & 1:
            for b in range(a + 1, ndim):
                if axes[a] > axes[b]:
                    count += (m >> axes[b]) & 1
    for ax in flips:
        count += (m >> ax) & 1
    return 1 - 2 * (count & 1)


@numba.njit(cache=True, nogil=True)
def _permutation_phases(parities, parity_offsets, sectors, axes, flips):
    nb, ndim = sectors.shape
    out = np.empty(nb, dtype=np.int8)
    # tabulate the phase of every parity pattern if there are few enough
    use_table = ndim <= 16 and (1 << ndim) <= nb
    table = np.empty((1 << ndim) if use_table else 0, dtype=np.int8)
    for m in range(table.shape[0]):
        table[m] = _pattern_phase(m, axes, flips)
    trivial = True
    for i in range(nb):
        m = 0
        for ax in range(ndim):
            m |= np.int64(parities[parity_offsets[ax] + sectors[i, ax]]) << ax
        s = table[m] if use_table else _pattern_phase(m, axes, flips)
        out[i] = s
        if s < 0:
            trivial = False
    return out, trivial


def permutation_phases(parities, parity_offsets, sectors, axes, flips):
    c = np.ascontiguousarray
    out, trivial = _permutation_phases(
        c(parities, dtype=np.int8),
        c(parity_offsets, dtype=np.int64),
        c(sectors, dtype=np.int64),
        c(axes, dtype=np.int64),
        c(flips, dtype=np.int64),
    )
    return None if trivial else out
