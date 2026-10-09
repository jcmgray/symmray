"""Reference packed kernels in plain numpy. Plans are index arrays per element,
so running one is a single gather. Building one costs time and memory
proportional to the number of elements.
"""

import numpy as np

NAME = "numpy"


def _element_entries(shapes):
    """For each element of each entry, in C order, get its entry and its multi
    index. Returns the entry of each element, the multi index as a list of
    arrays (one per dim), and the total number of elements.
    """
    n, nd = shapes.shape
    sizes = shapes.prod(axis=1) if nd else np.ones(n, dtype=np.int64)
    total = int(sizes.sum())
    ent = np.repeat(np.arange(n), sizes)
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(sizes[:-1], out=starts[1:])
    local = np.arange(total) - starts[ent]
    idx = []
    for d in range(nd - 1, -1, -1):
        dim = shapes[ent, d]
        idx.append(local % dim)
        local //= dim
    idx.reverse()
    return ent, idx, total


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
    ent, idx, _ = _element_entries(shapes)
    src_idx = src_base[ent]
    dst_idx = dst_base[ent]
    for d, i in enumerate(idx):
        src_idx = src_idx + i * src_strides[ent, d]
        dst_idx = dst_idx + i * dst_strides[ent, d]

    # invert into a gather. Unwritten positions get index -1, i.e. an appended
    # zero
    gather = np.full(dst_size, -1, dtype=np.int64)
    gather[dst_idx] = src_idx
    # which entry each destination element came from, for scaling
    gather_ent = np.full(dst_size, len(shapes), dtype=np.int64)
    gather_ent[dst_idx] = ent
    has_missing = bool(fill_zero) and bool((gather < 0).any())
    return gather, gather_ent, has_missing


def run_copy(plan, src, scales=None):
    gather, gather_ent, has_missing = plan
    if has_missing:
        src = np.append(src, src.dtype.type(0))
    out = src.take(gather)
    if scales is not None:
        out *= np.append(scales, 1).take(gather_ent)
    return out


def plan_muldiag(offsets, shapes, axis, vbase):
    n = shapes.shape[0]
    sizes = shapes.prod(axis=1)
    ent = np.repeat(np.arange(n), sizes)
    local = np.arange(int(sizes.sum())) - offsets[:-1][ent]
    post = shapes[:, axis + 1 :].prod(axis=1)
    return vbase[ent] + (local // post[ent]) % shapes[ent, axis]


def run_muldiag(plan, src, v, power):
    if power == 1:
        return src * v.take(plan)
    return src / v.take(plan)


def plan_gemm(a_off, rows, inner, b_off, cols, c_off):
    return list(
        zip(
            a_off.tolist(),
            rows.tolist(),
            inner.tolist(),
            b_off.tolist(),
            cols.tolist(),
            c_off.tolist(),
        )
    )


def run_gemm(plan, a, b, out_size):
    out = np.empty(out_size, dtype=np.result_type(a, b))
    for ao, r, k, bo, c, co in plan:
        np.matmul(
            a[ao : ao + r * k].reshape(r, k),
            b[bo : bo + k * c].reshape(k, c),
            out=out[co : co + r * c].reshape(r, c),
        )
    return out


def fuse_group(sub, radices, signed, voff, modulus, sizes):
    """Find the subsectors and new charges of one fused group of axes.

    Parameters
    ----------
    sub : array
        ``(num_blocks, m)`` charge ids of each block on the group's axes.
    radices : array
        ``(m,)`` number of charges of each axis.
    signed : array
        ``(sum(radices), num_components)`` charge values of every axis
        concatenated, signed to match the group dualness.
    voff : array
        ``(m + 1,)`` where each axis starts in ``signed`` and ``sizes``.
    modulus : int
        Combine charges by addition modulo this, or plain addition if 0.
    sizes : array
        ``(sum(radices),)`` sizes of every charge of every axis.

    Returns
    -------
    block_charge, block_offset : array
        For each block, the id of its new fused charge and where its subsector
        starts within that charge.
    charge_values, totals : array
        The value and size of each new charge, sorted.
    subkeys, charge_starts, suboffsets, subsizes : array
        Each subsector, grouped by new charge then sorted, as for
        `PackedSubIndexInfo`.
    """
    from ..packed_array_common import unique_charge_rows, unique_rows

    usub, inv = unique_rows(sub, radices.tolist())
    m = sub.shape[1]
    values = signed[voff[0] + usub[:, 0]]
    sub_sizes = sizes.take(voff[0] + usub[:, 0])
    for j in range(1, m):
        values = values + signed[voff[j] + usub[:, j]]
        sub_sizes = sub_sizes * sizes.take(voff[j] + usub[:, j])
    if modulus:
        values = values % modulus

    charge_values, charge_of_sub = unique_charge_rows(values)
    order = np.argsort(charge_of_sub, kind="stable")
    owner = charge_of_sub[order]
    sorted_sizes = sub_sizes[order]
    totals = np.bincount(
        owner, weights=sorted_sizes, minlength=charge_values.shape[0]
    ).astype(np.int64)
    starts = np.cumsum(totals) - totals
    sorted_offsets = np.cumsum(sorted_sizes) - sorted_sizes - starts[owner]
    sub_offsets = np.empty(len(order), dtype=np.int64)
    sub_offsets[order] = sorted_offsets
    charge_starts = np.zeros(charge_values.shape[0] + 1, dtype=np.int64)
    np.cumsum(
        np.bincount(owner, minlength=charge_values.shape[0]),
        out=charge_starts[1:],
    )
    return (
        charge_of_sub.take(inv),
        sub_offsets.take(inv),
        charge_values,
        totals,
        usub[order],
        charge_starts,
        sorted_offsets,
        sorted_sizes,
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
    """Get the fused structure and the copy into it.

    Parameters
    ----------
    cols : array
        ``(num_blocks, new_ndim)`` new charge id of each block on each new
        axis.
    suboffs : array
        ``(num_blocks, new_ndim)`` where each block's subsector starts within
        its new charge, zero for unfused axes.
    radices : array
        ``(new_ndim,)`` number of charges of each new axis.
    new_sizes, noff : array
        Sizes of the charges of every new axis concatenated, and where each
        axis starts.
    shapes : array
        ``(num_blocks, ndim)`` source block shapes.
    perm : array
        The source axes in fused order.
    glens : array
        How many source axes make up each new axis.
    src_base : array
        Where each source block starts.
    panel_last : bool
        Lay out each fused block as panels, one per subsector of the last axis,
        see `calc_plan_fuse`.

    Returns
    -------
    new_sectors, new_shapes, new_offsets : array
        The fused structure.
    src_base, src_strides, dst_base, dst_strides, iter_shapes : array
        The copy, entry ``i`` being source block ``i``.
    fill_zero : bool
        Whether some fused blocks are not fully covered by source blocks.
    """
    from ..packed_array_common import c_strides, calc_offsets, unique_rows

    nb = cols.shape[0]
    new_sectors, block_of = unique_rows(cols, radices.tolist())
    new_shapes = np.empty(new_sectors.shape, dtype=np.int64)
    for a in range(new_shapes.shape[1]):
        new_shapes[:, a] = new_sizes.take(noff[a] + new_sectors[:, a])
    new_offsets = calc_offsets(new_shapes)

    iter_shapes = shapes[:, perm]
    dst_base = new_offsets.take(block_of)
    dst_strides = np.empty((nb, len(perm)), dtype=np.int64)
    axis_strides = c_strides(new_shapes).take(block_of, axis=0)
    suboffs = suboffs.copy()
    if panel_last:
        # in a panel the last axis has the subsector's size, and the panel
        # starts after earlier panels of the block
        last_extent = new_shapes[block_of, -1]
        sub_size = iter_shapes[:, -glens[-1] :].prod(axis=1)
        rest = new_shapes[block_of, :-1].prod(axis=1)
        axis_strides[:, :-1] = axis_strides[:, :-1] // last_extent[:, None]
        axis_strides[:, :-1] *= sub_size[:, None]
        axis_strides[:, -1] = 1
        dst_base = dst_base + suboffs[:, -1] * rest
        suboffs[:, -1] = 0
    dst_base = dst_base + (suboffs * axis_strides).sum(axis=1)
    d = 0
    for k, glen in enumerate(glens.tolist()):
        stride = axis_strides[:, k]
        d += glen
        for dd in range(d - 1, d - glen - 1, -1):
            dst_strides[:, dd] = stride
            stride = stride * iter_shapes[:, dd]

    fill_zero = int(shapes.prod(axis=1).sum()) != int(new_offsets[-1])
    return (
        new_sectors,
        new_shapes,
        new_offsets,
        src_base,
        c_strides(shapes)[:, perm],
        dst_base,
        dst_strides,
        iter_shapes,
        fill_zero,
    )


def tensordot_align(sub_a, sub_b, radices):
    """Find which blocks of each side have a contracted subsector also on the
    other side. ``sub_a`` and ``sub_b`` are the contracted charge ids, both as
    a's ids. -1 marks a charge that a doesn't have.
    """
    from ..packed_array_common import row_keys

    valid_b = (sub_b >= 0).all(axis=1)
    keys_a = row_keys(sub_a, radices.tolist())
    keys_b = row_keys(np.where(sub_b >= 0, sub_b, 0), radices.tolist())
    if keys_a is None:
        # too many combinations for integer keys, compare rows as tuples
        rows_a = set(map(tuple, sub_a.tolist()))
        rows_b = set(map(tuple, sub_b[valid_b].tolist()))
        keep_a = np.array([r in rows_b for r in map(tuple, sub_a.tolist())])
        keep_b = np.array([r in rows_a for r in map(tuple, sub_b.tolist())])
        return keep_a.astype(bool), keep_b.astype(bool) & valid_b
    keep_a = np.isin(keys_a, keys_b[valid_b])
    keep_b = valid_b & np.isin(keys_b, keys_a)
    return keep_a, keep_b


def tensordot_entries(
    cl,
    a_off,
    inner,
    cr,
    b_off,
    left,
    right,
    radices,
):
    """Get the output blocks of a fused tensordot. Each is a row range of a
    fused block of a, times a column panel of a fused block of b.

    Parameters
    ----------
    cl, a_off, inner : array
        For each aligned pair of blocks, the left charge id, the start of the a
        block, and the contracted size.
    cr, b_off : array
        For each pair, the right charge id and the start of the b block.
    left, right : tuple
        The subsectors of each side, as ``(counts, starts, keys, offsets,
        sizes, sub_sizes, soff)``:

        - ``counts``, ``starts``: number and first row of the subsectors of
          each charge.
        - ``keys``, ``offsets``, ``sizes``: sub charge ids, offset and size of
          each row.
        - ``sub_sizes``, ``soff``: sizes of the charges of each sub index
          concatenated, and where each sub index starts.

    radices : array
        The number of charges of each output axis.

    Returns
    -------
    out_sectors, out_shapes, out_offsets : array
        The output structure, sorted.
    used : array
        Whether each charge of each output axis, concatenated, has a block.
    g_a_off, g_rows, g_inner, g_b_off, g_cols, g_c_off : array
        The matrix products, as for `plan_gemm`.
    """
    from ..packed_array_common import calc_offsets, row_order

    l_counts, l_starts, l_keys, l_offsets, l_sizes, l_subsizes, l_soff = left
    r_counts, r_starts, r_keys, r_offsets, r_sizes, r_subsizes, r_soff = right
    num_pairs = len(cl)

    n_l = l_counts.take(cl)
    n_r = r_counts.take(cr)
    per_pair = n_l * n_r
    num_out = int(per_pair.sum())
    e_pair = np.repeat(np.arange(num_pairs), per_pair)
    local = np.arange(num_out) - (np.cumsum(per_pair) - per_pair).take(e_pair)
    n_r_e = n_r.take(e_pair)
    rows_l = l_starts.take(cl).take(e_pair) + local // n_r_e
    rows_r = r_starts.take(cr).take(e_pair) + local % n_r_e

    def info(rows, keys, subsizes, soff):
        keys = keys.take(rows, axis=0)
        shape = np.empty(keys.shape, dtype=np.int64)
        for j in range(keys.shape[1]):
            shape[:, j] = subsizes.take(soff[j] + keys[:, j])
        return keys, shape

    keys_l, shape_l = info(rows_l, l_keys, l_subsizes, l_soff)
    keys_r, shape_r = info(rows_r, r_keys, r_subsizes, r_soff)
    out_sectors = np.concatenate([keys_l, keys_r], axis=1)
    out_shapes = np.concatenate([shape_l, shape_r], axis=1)
    order = row_order(out_sectors, radices.tolist())
    out_sectors = out_sectors[order]
    out_shapes = out_shapes[order]
    out_offsets = calc_offsets(out_shapes)
    c_off = np.empty(num_out, dtype=np.int64)
    c_off[order] = out_offsets[:-1]

    used = np.zeros(l_soff[-1] + r_soff[-1], dtype=bool)
    soff = np.concatenate([l_soff[:-1], l_soff[-1] + r_soff[:-1]])
    for j in range(out_sectors.shape[1]):
        used[soff[j] + out_sectors[:, j]] = True

    inner_e = inner.take(e_pair)
    return (
        out_sectors,
        out_shapes,
        out_offsets,
        used,
        a_off.take(e_pair) + l_offsets.take(rows_l) * inner_e,
        l_sizes.take(rows_l),
        inner_e,
        b_off.take(e_pair) + r_offsets.take(rows_r) * inner_e,
        r_sizes.take(rows_r),
        c_off,
    )


def run_scale(src, offsets, scales):
    return src * np.repeat(scales, np.diff(offsets))


def permutation_phases(parities, parity_offsets, sectors, axes, flips):
    nb, ndim = sectors.shape
    p = parities[np.asarray(parity_offsets)[:ndim] + sectors]
    position = np.empty(ndim, dtype=np.int64)
    position[list(axes)] = np.arange(ndim)
    count = np.zeros(nb, dtype=np.int64)
    for a in range(ndim):
        for b in range(a + 1, ndim):
            if position[a] > position[b]:
                count += p[:, a] & p[:, b]
    for ax in flips:
        count += p[:, ax]
    if not (count & 1).any():
        return None
    return (1 - 2 * (count & 1)).astype(np.int8)
