"""Methods for abelian arrays with packed storage, both fermionic and bosonic.

Most operations have two parts. A 'plan' is computed only from the structure of
the inputs (indices and sector ids), and is cached. A cheap 'run' then acts on
the data buffers via the current kernel set, see `symmray.packed.kernels`.
"""

import collections
import math
import os

from ..array_common import maybe_keep_label, parse_tensordot_axes
from ..utils import DEBUG, get_array_cls
from .kernels import get_kernels
from .packed_index import PackedIndex, PackedSubIndexInfo
from .packed_vector import PackedVector

# --------------------------- structure helpers ----------------------------- #


def c_strides(shapes):
    """The C order element strides of each block, ``(num_blocks, ndim)``."""
    import numpy as np

    n, nd = shapes.shape
    strides = np.ones((n, nd), dtype=np.int64)
    if nd > 1:
        strides[:, :-1] = np.cumprod(shapes[:, :0:-1], axis=1)[:, ::-1]
    return strides


def calc_offsets(shapes):
    """Where each block of ``shapes`` starts in a packed buffer, plus the total
    size at the end.
    """
    import numpy as np

    offsets = np.zeros(shapes.shape[0] + 1, dtype=np.int64)
    np.cumsum(shapes.prod(axis=1), out=offsets[1:])
    return offsets


def calc_shapes(indices, sectors):
    """The shape of each block given its charge ids ``sectors``."""
    import numpy as np

    shapes = np.empty((sectors.shape[0], len(indices)), dtype=np.int64)
    for ax, ix in enumerate(indices):
        shapes[:, ax] = ix.sizes.take(sectors[:, ax])
    return shapes


def row_keys(sectors, radices):
    """Combine each row of ``sectors`` into a single integer that sorts the
    same way, or None if that might overflow.
    """
    import numpy as np

    total = 1
    for r in radices:
        total *= max(r, 1)
    if total >= 2**62:
        return None
    keys = np.zeros(sectors.shape[0], dtype=np.int64)
    for ax, r in enumerate(radices):
        keys *= max(r, 1)
        keys += sectors[:, ax]
    return keys


def row_order(sectors, radices):
    """The order that lexicographically sorts the rows of ``sectors``."""
    import numpy as np

    keys = row_keys(sectors, radices)
    if keys is None:
        return np.lexsort(sectors.T[::-1])
    return np.argsort(keys, kind="stable")


def unique_rows(sectors, radices):
    """The sorted unique rows of ``sectors`` and the position of each original
    row among them.
    """
    import numpy as np

    keys = row_keys(sectors, radices)
    if keys is None:
        urows, inverse = np.unique(sectors, axis=0, return_inverse=True)
        return urows, inverse.reshape(-1)
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    return sectors[first], inverse


def unique_charge_rows(values):
    """The unique rows of an integer array of charge values,
    ``(n, num_components)``, sorted as the charges themselves would be, and the
    position of each original row among them.
    """
    import numpy as np

    if values.shape[0] == 0:
        return values, np.zeros(0, dtype=np.int64)
    lo = values.min(axis=0)
    shifted = values - lo
    radices = (shifted.max(axis=0) + 1).tolist()
    urows, inverse = unique_rows(shifted, radices)
    return urows + lo, inverse


def is_sorted_rows(sectors, radices):
    """Whether the rows of ``sectors`` are strictly increasing."""
    import numpy as np

    if sectors.shape[0] < 2:
        return True
    keys = row_keys(sectors, radices)
    if keys is None:
        order = np.lexsort(sectors.T[::-1])
        diffs = (sectors[1:] != sectors[:-1]).any(axis=1)
        return bool((order == np.arange(len(order))).all() and diffs.all())
    return bool((keys[1:] > keys[:-1]).all())


def radices_of(indices):
    """The number of charges of each index."""
    return [ix.num_charges for ix in indices]


def select_used_charges(indices, sectors):
    """Drop index charges that no block uses. Returns new indices and sectors,
    with charge ids renumbered. Row order is unchanged.
    """
    import numpy as np

    if sectors.shape[0] == 0:
        # keep the indices as they are rather than leave them empty
        return tuple(indices), sectors

    new_indices = list(indices)
    new_sectors = None
    for ax, ix in enumerate(indices):
        counts = np.bincount(sectors[:, ax], minlength=ix.num_charges)
        if counts.all():
            continue
        used = np.flatnonzero(counts)
        if new_sectors is None:
            new_sectors = sectors.copy()
        remap = np.full(ix.num_charges, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        new_sectors[:, ax] = remap.take(sectors[:, ax])
        new_indices[ax] = ix.select_charge_ids(used)
    if new_sectors is None:
        new_sectors = sectors
    return tuple(new_indices), new_sectors


# ------------------------------- plan cache -------------------------------- #

_plans = collections.OrderedDict()

try:
    _plan_cache_maxsize = int(os.environ["SYMMRAY_PACKED_PLAN_CACHE_MAXSIZE"])
except KeyError:
    _plan_cache_maxsize = 8192

try:
    _plan_cache_maxbytes = int(
        os.environ["SYMMRAY_PACKED_PLAN_CACHE_MAXBYTES"]
    )
except KeyError:
    _plan_cache_maxbytes = 2**30

_plan_cache_stats = collections.Counter()

# hashes of keys seen once, plans are only stored when seen a second time
_seen_once = collections.OrderedDict()


def _nbytes(obj):
    """Estimate the memory held by the arrays in a plan or key."""
    nbytes = getattr(obj, "nbytes", None)
    if nbytes is not None:
        return nbytes
    if isinstance(obj, bytes):
        return len(obj)
    if isinstance(obj, (tuple, list)):
        return 64 + sum(map(_nbytes, obj))
    if isinstance(obj, dict):
        return 64 + sum(map(_nbytes, obj.values()))
    return 32


def cached_plan(key, build, *args):
    """Get the plan for ``key``, building it with ``build(*args)`` on a miss.
    The cache is LRU and roughly bounded in memory, counting both plans and
    keys. Plans are only stored the second time their key is seen, so
    structures that never repeat (common in boundary contractions) don't fill
    the cache.
    """
    try:
        plan, nbytes = _plans[key]
    except KeyError:
        pass
    else:
        _plans.move_to_end(key)
        _plan_cache_stats["hit"] += 1
        return plan

    _plan_cache_stats["miss"] += 1
    plan = build(*args)
    if _plan_cache_maxsize == 0:
        return plan

    h = hash(key)
    if _seen_once.pop(h, None) is None:
        # first time seen, only remember the key's hash
        _seen_once[h] = True
        if len(_seen_once) > _plan_cache_maxsize:
            _seen_once.popitem(last=False)
        return plan

    nbytes = _nbytes(plan) + _nbytes(key)
    _plans[key] = plan, nbytes
    _plan_cache_stats["nbytes"] += nbytes
    while (len(_plans) > _plan_cache_maxsize) or (
        _plan_cache_stats["nbytes"] > _plan_cache_maxbytes
    ):
        _, (_, old_nbytes) = _plans.popitem(last=False)
        _plan_cache_stats["nbytes"] -= old_nbytes
    return plan


def clear_plan_cache():
    """Clear all cached packed plans."""
    _plans.clear()
    _seen_once.clear()
    _plan_cache_stats["nbytes"] = 0


def print_plan_cache_stats():
    """Print the size and hit rate of the plan cache."""
    hit, miss = _plan_cache_stats["hit"], _plan_cache_stats["miss"]
    ratio = miss / (hit + miss) if (hit + miss) else 0.0
    print(
        f"Plan cache: {len(_plans)} entries, "
        f"~{_plan_cache_stats['nbytes'] / 2**20:.1f} / "
        f"{_plan_cache_maxbytes / 2**20:.1f} MiB\n"
        f"missed: {miss}, hit: {hit}, miss ratio: {ratio:.2f}"
    )


# counts of methods run by converting to the sparse backend and back
FALLBACK_COUNTS = collections.Counter()


def packed_from_any(obj):
    """Convert block sparse arrays and vectors, possibly nested in tuples or
    lists, to their packed equivalents.
    """
    from ..sparse.sparse_array_common import SparseArrayCommon
    from ..sparse.sparse_vector import BlockVector

    if isinstance(obj, SparseArrayCommon):
        cls = get_array_cls(obj.symmetry, obj.fermionic, layout="packed")
        return cls.from_blocksparse(obj, symmetry=obj.symmetry)
    if isinstance(obj, BlockVector):
        return PackedVector.from_blockvector(obj)
    if isinstance(obj, (tuple, list)):
        return type(obj)(packed_from_any(x) for x in obj)
    return obj


def blocksparse_from_any(obj):
    """Convert packed arrays and vectors, possibly nested in tuples or lists,
    to their block sparse equivalents.
    """
    if isinstance(obj, PackedArrayCommon):
        return obj.to_blocksparse()
    if isinstance(obj, PackedVector):
        return obj.to_blockvector()
    if isinstance(obj, (tuple, list)):
        return type(obj)(blocksparse_from_any(x) for x in obj)
    return obj


# ---------------------------------- plans ---------------------------------- #


def calc_plan_select(indices, sectors, shapes, offsets, keep):
    """Plan keeping only the blocks where ``keep`` is True."""
    kernels = get_kernels()
    new_sectors = sectors[keep]
    new_shapes = shapes[keep]
    new_offsets = calc_offsets(new_shapes)
    plan = kernels.plan_copy(
        offsets[:-1][keep],
        c_strides(new_shapes),
        new_offsets[:-1],
        c_strides(new_shapes),
        new_shapes,
        int(new_offsets[-1]),
        False,
    )
    return indices, new_sectors, new_shapes, new_offsets, plan


def calc_plan_transpose(indices, sectors, shapes, offsets, perm):
    """Plan permuting the axes of every block by ``perm``. Returns the new
    structure, the copy plan, and the source block of each new block.
    """
    kernels = get_kernels()
    perm = list(perm)
    new_indices = tuple(indices[ax] for ax in perm)
    new_sectors = sectors[:, perm]
    order = row_order(new_sectors, radices_of(new_indices))
    new_sectors = new_sectors[order]
    new_shapes = shapes[order][:, perm]
    new_offsets = calc_offsets(new_shapes)
    plan = kernels.plan_copy(
        offsets[:-1][order],
        c_strides(shapes)[order][:, perm],
        new_offsets[:-1],
        c_strides(new_shapes),
        new_shapes,
        int(new_offsets[-1]),
        False,
    )
    return new_indices, new_sectors, new_shapes, new_offsets, plan, order


def _kernels_for_keys(radices):
    """The current kernels, or the numpy ones if combining charge ids with
    ``radices`` into single keys could overflow.
    """
    from .kernels import load_kernels

    total = 1
    for r in radices:
        total *= max(int(r), 1)
    if total >= 2**62:
        return load_kernels("numpy")
    return get_kernels()


def fuse_group_arrays(indices, sub, symmetry, dual):
    """Run `fuse_group` on the charge id rows ``sub`` of ``indices``, fused
    into a single index with dualness ``dual``.
    """
    import numpy as np

    modulus = symmetry.charge_modulus
    if modulus is None:
        raise NotImplementedError(
            f"Packed fusing needs `charge_modulus` for symmetry {symmetry}."
        )
    radices = np.array(radices_of(indices), dtype=np.int64)
    voff = np.zeros(len(indices) + 1, dtype=np.int64)
    np.cumsum(radices, out=voff[1:])
    return _kernels_for_keys(radices).fuse_group(
        sub,
        radices,
        np.concatenate(
            [
                ix.signed_charge_array(symmetry, ix.dual != dual)
                for ix in indices
            ]
        ),
        voff,
        modulus,
        np.concatenate([ix.sizes for ix in indices]),
    )


def assemble_fused(
    cols, suboffs, new_sizes, shapes, src_base, perm, glens, panel_last
):
    """Build the fused structure and copy plan.

    ``cols`` gives each block's new charge ids, ``suboffs`` its subsector
    offsets within them, and ``new_sizes`` the charge sizes of each new axis.
    Block axes are permuted by ``perm``, then grouped in runs of ``glens``. If
    ``panel_last``, each fused block is instead laid out as consecutive C
    contiguous panels, one per subsector of the last axis. Tensordot uses this
    so every column panel is a contiguous matrix.
    """
    import numpy as np

    radices = np.array([len(x) for x in new_sizes], dtype=np.int64)
    noff = np.zeros(len(new_sizes) + 1, dtype=np.int64)
    np.cumsum(radices, out=noff[1:])
    glens = np.array(glens, dtype=np.int64)
    kernels = _kernels_for_keys(radices)
    (
        new_sectors,
        new_shapes,
        new_offsets,
        src_base,
        src_strides,
        dst_base,
        dst_strides,
        iter_shapes,
        fill_zero,
    ) = kernels.fuse_assemble(
        cols,
        suboffs,
        radices,
        np.concatenate(new_sizes),
        noff,
        shapes,
        np.array(perm, dtype=np.int64),
        glens,
        src_base,
        bool(panel_last and glens[-1] > 1),
    )
    plan = get_kernels().plan_copy(
        src_base,
        src_strides,
        dst_base,
        dst_strides,
        iter_shapes,
        int(new_offsets[-1]),
        bool(fill_zero),
        coalesced=kernels is get_kernels(),
    )
    return new_sectors, new_shapes, new_offsets, plan


def calc_plan_fuse(
    indices,
    sectors,
    shapes,
    src_base,
    duals,
    symmetry,
    axes_groups,
):
    """Plan fusing ``axes_groups``, as `fuse`. ``src_base`` gives where each
    block starts in the source buffer, so a subset of blocks can be fused
    directly. Returns the new structure and the copy plan, whose entries are
    the source blocks in order.
    """
    import numpy as np

    from ..sparse.sparse_array_common import calc_fuse_group_info

    info = calc_fuse_group_info(axes_groups, duals)
    perm, axes_before, axes_after, group_duals = (
        info[3],
        info[5],
        info[6],
        info[8],
    )
    new_axes = (
        [(ax,) for ax in axes_before]
        + [tuple(g) for g in axes_groups]
        + [(ax,) for ax in axes_after]
    )
    new_duals = (
        [indices[ax].dual for ax in axes_before]
        + list(group_duals)
        + [indices[ax].dual for ax in axes_after]
    )

    nb = sectors.shape[0]
    cols = np.empty((nb, len(new_axes)), dtype=np.int64)
    suboffs = np.zeros((nb, len(new_axes)), dtype=np.int64)
    new_indices = []
    for k, (gaxes, gdual) in enumerate(zip(new_axes, new_duals)):
        if len(gaxes) == 1:
            (ax,) = gaxes
            cols[:, k] = sectors[:, ax]
            new_indices.append(indices[ax])
            continue

        gindices = [indices[ax] for ax in gaxes]
        (
            cols[:, k],
            suboffs[:, k],
            charge_values,
            totals,
            subkeys,
            charge_starts,
            suboffsets,
            subsizes,
        ) = fuse_group_arrays(
            gindices, sectors[:, list(gaxes)], symmetry, gdual
        )
        new_indices.append(
            PackedIndex(
                charges=symmetry.array_to_charges(charge_values),
                sizes=totals,
                dual=gdual,
                subinfo=PackedSubIndexInfo(
                    indices=gindices,
                    subkeys=subkeys,
                    charge_starts=charge_starts,
                    suboffsets=suboffsets,
                    subsizes=subsizes,
                ),
            )
        )

    new_indices = tuple(new_indices)
    return new_indices, *assemble_fused(
        cols,
        suboffs,
        [ix.sizes for ix in new_indices],
        shapes,
        src_base,
        perm,
        [len(g) for g in new_axes],
        False,
    )


def calc_plan_unfuse(indices, sectors, shapes, offsets, axis):
    """Plan unfusing ``axis``. Returns the new structure, the copy plan, and
    the source block of each new block.
    """
    import numpy as np

    kernels = get_kernels()
    ix = indices[axis]
    subinfo = ix.subinfo
    if subinfo is None:
        raise ValueError(f"Axis {axis} is not fused in this array.")

    nb = sectors.shape[0]
    starts = subinfo.charge_starts[:-1]
    counts = subinfo.charge_starts[1:] - starts
    fused = sectors[:, axis]
    reps = counts.take(fused)
    num_out = int(reps.sum())
    blk = np.repeat(np.arange(nb), reps)
    rep_starts = np.cumsum(reps) - reps
    row = starts.take(fused).take(blk) + (
        np.arange(num_out) - rep_starts.take(blk)
    )

    subkeys = subinfo.subkeys.take(row, axis=0)
    subshape = np.empty(subkeys.shape, dtype=np.int64)
    for j, six in enumerate(subinfo.indices):
        subshape[:, j] = six.sizes.take(subkeys[:, j])

    new_indices = (*indices[:axis], *subinfo.indices, *indices[axis + 1 :])
    new_sectors = np.concatenate(
        [sectors[blk, :axis], subkeys, sectors[blk, axis + 1 :]], axis=1
    )
    order = row_order(new_sectors, radices_of(new_indices))
    new_sectors = new_sectors[order]
    blk = blk[order]
    row = row[order]
    subshape = subshape[order]
    new_shapes = np.concatenate(
        [shapes[blk, :axis], subshape, shapes[blk, axis + 1 :]], axis=1
    )
    new_offsets = calc_offsets(new_shapes)

    src_strides = c_strides(shapes).take(blk, axis=0)
    axis_stride = src_strides[:, axis : axis + 1]
    src_base = (
        offsets.take(blk) + subinfo.suboffsets.take(row) * axis_stride[:, 0]
    )
    iter_src_strides = np.concatenate(
        [
            src_strides[:, :axis],
            axis_stride * c_strides(subshape),
            src_strides[:, axis + 1 :],
        ],
        axis=1,
    )
    plan = kernels.plan_copy(
        src_base,
        iter_src_strides,
        new_offsets[:-1],
        c_strides(new_shapes),
        new_shapes,
        int(new_offsets[-1]),
        False,
    )
    return new_indices, new_sectors, new_shapes, new_offsets, plan, blk


def _missing_side():
    """A tensordot side with no axes, a single empty subsector."""
    import numpy as np

    one = np.ones(1, dtype=np.int64)
    zero = np.zeros(1, dtype=np.int64)
    return (
        one,
        zero,
        np.zeros((1, 0), dtype=np.int64),
        zero,
        one,
        np.zeros(0, dtype=np.int64),
        zero,
    )


def _fuse_side(indices, sub, symmetry, dual):
    """Fuse the charge id rows ``sub`` of ``indices`` into one tensordot axis.
    Returns the new charge id and subsector offset of each row, the size of
    each new charge, and the subsectors in the form `tensordot_entries` takes.
    A single axis gives one subsector per charge.
    """
    import numpy as np

    if len(indices) == 1:
        (ix,) = indices
        n = ix.num_charges
        side = (
            np.ones(n, dtype=np.int64),
            np.arange(n, dtype=np.int64),
            np.arange(n, dtype=np.int64)[:, None],
            np.zeros(n, dtype=np.int64),
            ix.sizes,
            ix.sizes,
            np.array([0, n], dtype=np.int64),
        )
        return (
            sub[:, 0],
            np.zeros(sub.shape[0], dtype=np.int64),
            ix.sizes,
            side,
        )

    (
        block_charge,
        block_offset,
        _,
        totals,
        subkeys,
        charge_starts,
        suboffsets,
        subsizes,
    ) = fuse_group_arrays(indices, sub, symmetry, dual)
    soff = np.zeros(len(indices) + 1, dtype=np.int64)
    np.cumsum(radices_of(indices), out=soff[1:])
    side = (
        np.diff(charge_starts),
        charge_starts[:-1],
        subkeys,
        suboffsets,
        subsizes,
        np.concatenate([ix.sizes for ix in indices]),
        soff,
    )
    return block_charge, block_offset, totals, side


def calc_plan_tensordot(a, b, left_axes, axes_a, axes_b, right_axes):
    """Plan a fused tensordot between packed arrays ``a`` and ``b``.

    Both are aligned and fused into matrices. Each output block is then one
    matrix product, of a row range of a and a column panel of b, so the
    products write the unfused result directly. Fused indices are only built as
    arrays. The contracted axes of both are fused together, in a's charge ids.
    """
    import numpy as np

    if a._symmetry != b._symmetry:
        raise ValueError(
            f"Cannot contract {a._symmetry} and {b._symmetry} arrays."
        )
    symmetry = a._symmetry

    # find the contracted subsectors present in both, using a's charge ids
    radices = np.array(
        [a._indices[ax].num_charges for ax in axes_a], dtype=np.int64
    )
    sub_b = np.empty((b.num_blocks, len(axes_b)), dtype=np.int64)
    for j, (axa, axb) in enumerate(zip(axes_a, axes_b)):
        ixa, ixb = a._indices[axa], b._indices[axb]
        if ixa.dual == ixb.dual:
            raise ValueError(
                f"Contracted axes {axa} and {axb} must have opposite duals."
            )
        ids = ixa.charge_ids()
        remap = np.array([ids.get(c, -1) for c in ixb.charges], dtype=np.int64)
        shared = remap >= 0
        if (ixa.sizes.take(remap[shared]) != ixb.sizes[shared]).any():
            raise ValueError(
                f"Contracted axes {axa} and {axb} have different sizes for "
                "the same charge."
            )
        sub_b[:, j] = remap.take(b._sectors[:, axb])
    keep_a, keep_b = _kernels_for_keys(radices).tensordot_align(
        a._sectors[:, list(axes_a)], sub_b, radices
    )
    kept_a = None if keep_a.all() else np.flatnonzero(keep_a)
    kept_b = None if keep_b.all() else np.flatnonzero(keep_b)
    sec_a = a._sectors if kept_a is None else a._sectors[kept_a]
    sec_b = b._sectors if kept_b is None else b._sectors[kept_b]
    if kept_b is not None:
        sub_b = sub_b[kept_b]
    has_left = bool(left_axes)
    has_right = bool(right_axes)

    # fuse contracted axes of both together, so they share charge ids and
    # shared charges are in the same order
    na = sec_a.shape[0]
    ck, ck_off, k_sizes, _ = _fuse_side(
        [a._indices[ax] for ax in axes_a],
        np.concatenate([sec_a[:, list(axes_a)], sub_b]),
        symmetry,
        a._indices[axes_a[0]].dual,
    )

    if has_left:
        indices_l = tuple(a._indices[ax] for ax in left_axes)
        cl, cl_off, l_sizes, left = _fuse_side(
            indices_l,
            sec_a[:, list(left_axes)],
            symmetry,
            indices_l[0].dual,
        )
        cols_a = (cl, ck[:na])
        offs_a = (cl_off, ck_off[:na])
        sizes_a = (l_sizes, k_sizes)
        glens_a = (len(left_axes), len(axes_a))
    else:
        indices_l = ()
        left = _missing_side()
        cols_a, offs_a, sizes_a = (ck[:na],), (ck_off[:na],), (k_sizes,)
        glens_a = (len(axes_a),)

    if has_right:
        indices_r = tuple(b._indices[ax] for ax in right_axes)
        cr, cr_off, r_sizes, right = _fuse_side(
            indices_r,
            sec_b[:, list(right_axes)],
            symmetry,
            indices_r[0].dual,
        )
        cols_b = (ck[na:], cr)
        offs_b = (ck_off[na:], cr_off)
        sizes_b = (k_sizes, r_sizes)
        glens_b = (len(axes_b), len(right_axes))
    else:
        indices_r = ()
        right = _missing_side()
        cols_b, offs_b, sizes_b = (ck[na:],), (ck_off[na:],), (k_sizes,)
        glens_b = (len(axes_b),)

    def assemble(x, kept, cols, offs, sizes, perm, glens, panel_last):
        cols = np.stack(cols, axis=1)
        if kept is None and perm == tuple(range(x.ndim)) and max(glens) == 1:
            # already a matrix in the right layout
            return cols, x._shapes, x._offsets, None
        return assemble_fused(
            cols,
            np.stack(offs, axis=1),
            sizes,
            x._shapes if kept is None else x._shapes[kept],
            x._offsets[:-1] if kept is None else x._offsets[:-1][kept],
            perm,
            glens,
            panel_last,
        )

    sa, sha, oa, plan_a = assemble(
        a,
        kept_a,
        cols_a,
        offs_a,
        sizes_a,
        (*left_axes, *axes_a),
        glens_a,
        False,
    )
    sb, _, ob, plan_b = assemble(
        b,
        kept_b,
        cols_b,
        offs_b,
        sizes_b,
        (*axes_b, *right_axes),
        glens_b,
        True,
    )

    # pair each block of a with the blocks of b sharing its contracted charge
    target = sa[:, -1]
    lo = np.searchsorted(sb[:, 0], target, "left")
    counts = np.searchsorted(sb[:, 0], target, "right") - lo
    num_pairs = int(counts.sum())
    pa = np.repeat(np.arange(sa.shape[0]), counts)
    pair_starts = np.cumsum(counts) - counts
    pb = lo.take(pa) + np.arange(num_pairs) - pair_starts.take(pa)
    inner = sha[pa, -1]

    if not (has_left or has_right):
        # a scalar, several aligned charges may contribute
        gemms = list(
            zip(oa.take(pa).tolist(), inner.tolist(), ob.take(pb).tolist())
        )
        out_sectors = np.zeros((min(num_pairs, 1), 0), dtype=np.int64)
        out_shapes = np.zeros((min(num_pairs, 1), 0), dtype=np.int64)
        return (
            plan_a,
            plan_b,
            kept_a,
            kept_b,
            gemms,
            True,
            ((), out_sectors, out_shapes, calc_offsets(out_shapes)),
        )

    out_indices = (*indices_l, *indices_r)
    out_radices = np.array(radices_of(out_indices), dtype=np.int64)
    zeros = np.zeros(num_pairs, dtype=np.int64)
    (
        out_sectors,
        out_shapes,
        out_offsets,
        used,
        *gemm_args,
    ) = _kernels_for_keys(out_radices).tensordot_entries(
        sa[pa, 0] if has_left else zeros,
        oa.take(pa),
        inner,
        sb[pb, -1] if has_right else zeros,
        ob.take(pb),
        left,
        right,
        out_radices,
    )
    gemms = get_kernels().plan_gemm(*gemm_args)
    if not used.all():
        out_indices, out_sectors = select_used_charges(
            out_indices, out_sectors
        )

    return (
        plan_a,
        plan_b,
        kept_a,
        kept_b,
        gemms,
        False,
        (out_indices, out_sectors, out_shapes, out_offsets),
    )


def calc_plan_outer(a, b):
    """Plan a tensordot with no contracted axes, i.e. every pair of blocks."""
    import numpy as np

    na, nb = a.num_blocks, b.num_blocks
    pa = np.repeat(np.arange(na), nb)
    pb = np.tile(np.arange(nb), na)
    sectors = np.concatenate([a._sectors[pa], b._sectors[pb]], axis=1)
    shapes = np.concatenate([a._shapes[pa], b._shapes[pb]], axis=1)
    offsets = calc_offsets(shapes)
    pairs = list(
        zip(
            a._offsets.take(pa).tolist(),
            np.diff(a._offsets).take(pa).tolist(),
            b._offsets.take(pb).tolist(),
            np.diff(b._offsets).take(pb).tolist(),
            offsets[:-1].tolist(),
        )
    )
    indices, sectors = select_used_charges((*a._indices, *b._indices), sectors)
    return pairs, (indices, sectors, shapes, offsets)


def calc_plan_muldiag(x, v, axis):
    """Plan multiplying ``x`` by the vector ``v`` along ``axis``. Blocks with
    no matching vector block are dropped first.
    """
    import numpy as np

    kernels = get_kernels()
    ids = {c: i for i, c in enumerate(v._charges)}
    vstart = v._offsets[:-1]
    vsizes = np.diff(v._offsets)
    ix = x._indices[axis]
    per_charge = np.full(ix.num_charges, -1, dtype=np.int64)
    for i, (c, d) in enumerate(zip(ix.charges, ix.sizes.tolist())):
        j = ids.get(c, -1)
        if j < 0:
            continue
        if vsizes[j] != d:
            raise ValueError(
                f"Vector block for charge {c} has size {vsizes[j]}, "
                f"but axis {axis} has size {d} there."
            )
        per_charge[i] = vstart[j]
    vbase = per_charge.take(x._sectors[:, axis])
    keep = vbase >= 0
    if keep.all():
        select = keep = None
        offsets, shapes = x._offsets, x._shapes
    else:
        select = calc_plan_select(
            x._indices, x._sectors, x._shapes, x._offsets, keep
        )
        _, _, shapes, offsets, _ = select
        vbase = vbase[keep]
    plan = kernels.plan_muldiag(offsets, shapes, axis, vbase)
    return select, keep, plan


def calc_plan_trim(indices, sectors, shapes, offsets, axis, keep):
    """Plan keeping only the first ``keep[c]`` entries of each charge id ``c``
    along ``axis``, dropping charges with zero entries.
    """
    import numpy as np

    kernels = get_kernels()
    ids = sectors[:, axis]
    new_sizes = keep.take(ids)
    mask = new_sizes > 0
    new_shapes = shapes[mask].copy()
    new_shapes[:, axis] = new_sizes[mask]
    kept_ids = np.flatnonzero(keep > 0)
    remap = np.cumsum(keep > 0) - 1
    new_sectors = sectors[mask].copy()
    new_sectors[:, axis] = remap.take(ids[mask])
    new_offsets = calc_offsets(new_shapes)
    ix = indices[axis]
    new_ix = ix.select_charge_ids(kept_ids).copy_with(
        sizes=keep.take(kept_ids), subinfo=None
    )
    new_indices = (*indices[:axis], new_ix, *indices[axis + 1 :])
    plan = kernels.plan_copy(
        offsets[:-1][mask],
        c_strides(shapes)[mask],
        new_offsets[:-1],
        c_strides(new_shapes),
        new_shapes,
        int(new_offsets[-1]),
        False,
    )
    return new_indices, new_sectors, new_shapes, new_offsets, plan, mask


def trim_bond(x, axis, keep):
    """Keep only the first ``keep[c]`` entries of each charge id ``c`` of ``x``
    along ``axis``, inplace.
    """
    indices, sectors, shapes, offsets, plan, mask = cached_plan(
        ("trim", x.structure_key(), axis, keep.tobytes()),
        calc_plan_trim,
        x._indices,
        x._sectors,
        x._shapes,
        x._offsets,
        axis,
        keep,
    )
    phases = x._block_phases()
    if phases is not None:
        phases = phases[mask]
    data = get_kernels().run_copy(plan, x._data, phases)
    return x._set_structure(indices, sectors, shapes, offsets, data)


def truncate_svd_result_packed(
    U,
    s,
    VH,
    cutoff,
    cutoff_mode,
    max_bond,
    absorb,
    renorm,
    use_abs=False,
):
    """Truncate a full packed svd-like result, using the singular values of
    every sector, then absorb them as requested. See
    `truncate_svd_result_blocksparse`.
    """
    import numpy as np

    from ..linalg_common import (
        Absorb,
        absorb_svd_result,
        calc_truncation_cutoff,
    )

    absorb = Absorb.parse(absorb)
    if renorm:
        raise NotImplementedError("renorm not implemented yet.")

    s_data = np.abs(s._data) if use_abs else s._data
    sizes = np.diff(s._offsets)
    keep = sizes.copy()

    if ((cutoff > 0.0) or (max_bond > 0)) and s_data.size:
        abs_cutoff = calc_truncation_cutoff(
            np.sort(s_data), cutoff, cutoff_mode, max_bond, np
        )
        if abs_cutoff is not None:
            # number of values to keep in each sector
            keep = np.add.reduceat(
                (s_data >= abs_cutoff).astype(np.int64), s._offsets[:-1]
            )

    if (keep != sizes).any():
        trim_bond(U, 1, keep)
        trim_bond(VH, 0, keep)
        kept = np.flatnonzero(keep > 0)
        new_offsets = np.zeros(len(kept) + 1, dtype=np.int64)
        np.cumsum(keep.take(kept), out=new_offsets[1:])
        starts = s._offsets.take(kept)
        if len(kept):
            data = np.concatenate(
                [
                    s._data[a : a + n]
                    for a, n in zip(starts.tolist(), keep.take(kept).tolist())
                ]
            )
        else:
            data = s._data[:0]
        s = PackedVector.from_parts(
            [s._charges[i] for i in kept.tolist()], data, new_offsets
        )
    else:
        # inner fusing info is no longer valid either way
        for x, axis in ((U, 1), (VH, 0)):
            ix = x._indices[axis]
            if ix.subinfo is not None:
                x._indices = (
                    *x._indices[:axis],
                    ix.copy_with(subinfo=None),
                    *x._indices[axis + 1 :],
                )
                x._skey = None

    if DEBUG:
        U.check()
        VH.check()
        s.check()

    return absorb_svd_result(U, s, VH, absorb)


# ------------------------------- the mixin --------------------------------- #


class PackedArrayCommon:
    """Mixin for abelian arrays with packed storage."""

    def _init_abelian(
        self,
        sectors,
        data,
        indices,
        charge=None,
        symmetry=None,
        label=None,
    ):
        import numpy as np

        self._indices = tuple(indices)
        self._symmetry = self.get_class_symmetry(symmetry)
        self._label = label
        self._sectors = np.asarray(sectors, dtype=np.int64).reshape(
            len(sectors), len(self._indices)
        )
        self._shapes = calc_shapes(self._indices, self._sectors)
        self._offsets = calc_offsets(self._shapes)
        self._data = np.asarray(data).reshape(-1)
        self._skey = None

        if charge is None:
            if self.num_blocks > 0:
                sector = self.sectors[0]
                self._charge = self._symmetry.combine(
                    *(
                        self._symmetry.sign(c, ix.dual)
                        for c, ix in zip(sector, self._indices)
                    )
                )
            else:
                self._charge = self._symmetry.combine()
        else:
            self._charge = charge

    def _new_with_abelian(
        self, indices, charge, data, sectors, shapes, offsets, label=None
    ):
        new = self._new_with_packedcommon(data, sectors, shapes, offsets)
        new._indices = indices
        new._charge = charge
        new._symmetry = self._symmetry
        new._label = label
        return new

    def _copy_abelian(self):
        new = self._copy_packedcommon()
        new._indices = self._indices
        new._charge = self._charge
        new._symmetry = self._symmetry
        new._label = self._label
        return new

    def _copy_with_abelian(
        self,
        indices=None,
        charge=None,
        data=None,
        sectors=None,
        shapes=None,
        offsets=None,
    ):
        new = self._copy_abelian()
        new._modify_abelian(
            indices=indices,
            charge=charge,
            data=data,
            sectors=sectors,
            shapes=shapes,
            offsets=offsets,
        )
        return new

    def _modify_abelian(
        self,
        indices=None,
        charge=None,
        data=None,
        sectors=None,
        shapes=None,
        offsets=None,
    ):
        self._modify_packedcommon(
            data=data, sectors=sectors, shapes=shapes, offsets=offsets
        )
        if indices is not None:
            self._indices = indices
            self._skey = None
        if charge is not None:
            self._charge = charge
        return self

    def _set_structure(
        self, indices, sectors, shapes, offsets, data, phases=None
    ):
        """Replace the whole structure and data inplace, and for fermionic
        arrays the lazy block phases.
        """
        self._indices = indices
        self._sectors = sectors
        self._shapes = shapes
        self._offsets = offsets
        self._data = data
        self._skey = None
        if self.fermionic:
            self._phases = phases
        return self

    def _block_phases(self):
        """The lazy phase of each block for fermionic arrays, or None if all
        +1. Operations that copy the data apply these.
        """
        return getattr(self, "_phases", None)

    def _phased_blocks(self):
        """A view of each block, with any lazy phases applied."""
        data = self._data
        phases = self._block_phases()
        if phases is not None:
            data = get_kernels().run_scale(data, self._offsets, phases)
        offsets = self._offsets.tolist()
        return [
            data[offsets[i] : offsets[i + 1]].reshape(shape)
            for i, shape in enumerate(self._shapes.tolist())
        ]

    def structure_key(self):
        """A hashable key for the block structure of this array. Arrays with
        equal keys have identical layouts.
        """
        if self._skey is None:
            self._skey = (
                self._symmetry,
                tuple(ix.hashkey() for ix in self._indices),
                self._sectors.shape[0],
                self._sectors.tobytes(),
            )
        return self._skey

    # ----------------------------- properties ------------------------------ #

    @property
    def label(self):
        """The label of the array, possibly used for ordering odd parity
        fermionic modes.
        """
        return self._label

    @property
    def charge(self):
        """The total charge of the array."""
        return self._charge

    @property
    def sizes(self):
        """The sizes of each index."""
        return tuple(tuple(ix.sizes.tolist()) for ix in self._indices)

    @property
    def charges(self):
        """The possible charges of each index."""
        return tuple(ix.charges for ix in self._indices)

    @property
    def shape(self):
        """The shape of the array, i.e. the total size of each index."""
        return tuple(ix.size_total for ix in self._indices)

    @property
    def size(self):
        """The number of possible elements in this array, if it was dense."""
        return math.prod(self.shape)

    @property
    def ndim(self):
        """The number of dimensions/indices."""
        return len(self._indices)

    @property
    def sectors(self):
        """The sectors, as tuples of charges, of each block in order."""
        charges = [ix.charges for ix in self._indices]
        return tuple(
            tuple(cs[i] for cs, i in zip(charges, row))
            for row in self._sectors.tolist()
        )

    def get_sector_block_pairs(self):
        """Iterate over ``(sector, block)`` pairs, blocks being views."""
        offsets = self._offsets.tolist()
        for i, (sector, shape) in enumerate(
            zip(self.sectors, self._shapes.tolist())
        ):
            yield (
                sector,
                self._data[offsets[i] : offsets[i + 1]].reshape(shape),
            )

    @property
    def blocks(self):
        """A mapping of each sector to a view of its block."""
        return dict(self.get_sector_block_pairs())

    def get_all_blocks(self):
        return self.blocks.values()

    def get_any_sector(self):
        return next(iter(self.sectors), ())

    def _find_block(self, sector):
        import numpy as np

        try:
            row = [ix.charge_ids()[c] for ix, c in zip(self._indices, sector)]
        except KeyError:
            return -1
        matches = (self._sectors == np.array(row, dtype=np.int64)).all(axis=1)
        hits = np.flatnonzero(matches)
        return int(hits[0]) if len(hits) else -1

    def has_sector(self, sector):
        return self._find_block(sector) >= 0

    def get_block(self, sector):
        """Get a view of the block for ``sector``."""
        i = self._find_block(sector)
        if i < 0:
            raise KeyError(sector)
        shape = self._shapes[i].tolist()
        return self._data[self._offsets[i] : self._offsets[i + 1]].reshape(
            shape
        )

    def get_block_shape(self, sector):
        """The shape of the block for ``sector``."""
        return tuple(ix.size_of(c) for ix, c in zip(self._indices, sector))

    def _get_scalar_element_blockwise(self):
        if self.ndim != 0:
            raise ValueError("Array does not seem to be a scalar.")
        if self.num_blocks == 0:
            return 0.0
        return self._data[0]

    # ------------------------- checks and conversion ----------------------- #

    def _check_abelian(self):
        import numpy as np

        for ix in self._indices:
            ix.check()
        S = self._sectors
        assert S.dtype == np.int64
        assert S.shape == (S.shape[0], self.ndim)
        for ax, ix in enumerate(self._indices):
            assert (S[:, ax] >= 0).all() and (S[:, ax] < ix.num_charges).all()
        if not is_sorted_rows(S, radices_of(self._indices)):
            raise ValueError("Sectors must be unique and sorted.")
        assert (self._shapes == calc_shapes(self._indices, S)).all()
        assert (self._offsets == calc_offsets(self._shapes)).all()
        assert self._data.ndim == 1
        assert self._data.shape[0] == self._offsets[-1]
        if self.num_blocks and self.ndim:
            sym = self._symmetry
            signed = np.stack(
                [
                    sym.sign_array(
                        ix.charge_array(sym).take(S[:, ax], axis=0),
                        np.full(S.shape[0], ix.dual),
                    )
                    for ax, ix in enumerate(self._indices)
                ],
                axis=1,
            )
            totals = sym.array_to_charges(sym.combine_array(signed, axis=1))
            for total, sector in zip(totals, self.sectors):
                if total != self._charge:
                    raise ValueError(
                        f"Invalid sector {sector} for array with "
                        f"{self.duals} and charge total {self._charge}."
                    )
        if not np.isfinite(self._data).all():
            raise ValueError("Array contains non-finite values.")

    def check_chargemaps_aligned(self):
        """Check that every charge of every index has at least one block."""
        import numpy as np

        if self.num_blocks == 0:
            return
        for ax, ix in enumerate(self._indices):
            used = np.unique(self._sectors[:, ax])
            if len(used) != ix.num_charges:
                raise ValueError(
                    f"Charges for index {ix} are inconsistent with blocks."
                )

    @classmethod
    def _parts_from_blocksparse(cls, x):
        """Get the packed indices, sector ids, data and phases of a block
        sparse array ``x``.
        """
        import numpy as np

        indices = tuple(PackedIndex.from_blockindex(ix) for ix in x.indices)
        id_maps = [ix.charge_ids() for ix in indices]
        sectors = list(x.blocks)
        rows = np.array(
            [[m[c] for m, c in zip(id_maps, s)] for s in sectors],
            dtype=np.int64,
        ).reshape(len(sectors), len(indices))
        order = row_order(rows, radices_of(indices)).tolist()
        arrays = [ar_ravel(x.blocks[sectors[i]]) for i in order]
        if arrays:
            data = np.concatenate(arrays)
        else:
            data = np.zeros(0, dtype=getattr(x, "dtype", "float64"))

        x_phases = getattr(x, "phases", None)
        if x_phases and any(p == -1 for p in x_phases.values()):
            phases = np.array(
                [x_phases.get(sectors[i], 1) for i in order], dtype=np.int8
            )
        else:
            phases = None
        return indices, rows[order], data, phases

    def _to_blocksparse_parts(self):
        """Get the block sparse indices, blocks and lazy phases."""
        indices = tuple(ix.to_blockindex() for ix in self._indices)
        phases = self._block_phases()
        if phases is None:
            phase_dict = {}
        else:
            phase_dict = {
                sector: -1
                for sector, p in zip(self.sectors, phases.tolist())
                if p == -1
            }
        return indices, self.blocks, phase_dict

    def _via_blocksparse(self, method, /, *args, inplace=False, **kwargs):
        """Run ``method`` on the block sparse version of this array, with any
        packed arguments converted too, and convert the results back. This
        covers methods without a native packed implementation.
        """
        FALLBACK_COUNTS[method] += 1
        x = self.to_blocksparse()
        args = blocksparse_from_any(args)
        kwargs = {k: blocksparse_from_any(v) for k, v in kwargs.items()}
        out = packed_from_any(getattr(x, method)(*args, **kwargs))
        if not inplace:
            return out
        return self._become(out)

    def to_packed(self):
        """Return this array, which is already packed."""
        return self

    @classmethod
    def _sparse_cls(cls):
        """The block sparse class matching this packed class."""
        return get_array_cls(
            cls.static_symmetry, cls.fermionic, layout="sparse"
        )

    @classmethod
    def from_blocks(cls, blocks, duals, charge=None, symmetry=None, **kwargs):
        """Create from a mapping of sectors to blocks, see the sparse
        `from_blocks`.
        """
        x = cls._sparse_cls().from_blocks(
            blocks, duals, charge=charge, symmetry=symmetry, **kwargs
        )
        return cls.from_blocksparse(x)

    @classmethod
    def from_dense(cls, array, index_maps, duals, **kwargs):
        """Create from a dense array, see the sparse `from_dense`."""
        x = cls._sparse_cls().from_dense(array, index_maps, duals, **kwargs)
        return cls.from_blocksparse(x)

    @classmethod
    def from_fill_fn(cls, fill_fn, indices, **kwargs):
        """Create by filling every valid sector, see the sparse `from_fill_fn`."""
        indices = [
            ix.to_blockindex() if isinstance(ix, PackedIndex) else ix
            for ix in indices
        ]
        x = cls._sparse_cls().from_fill_fn(fill_fn, indices, **kwargs)
        return cls.from_blocksparse(x)

    @classmethod
    def random(cls, indices, **kwargs):
        """Create with random values, see the sparse `random`."""
        indices = [
            ix.to_blockindex() if isinstance(ix, PackedIndex) else ix
            for ix in indices
        ]
        x = cls._sparse_cls().random(indices, **kwargs)
        return cls.from_blocksparse(x)

    @classmethod
    def from_pytree(cls, pytree):
        """Create from a pytree, as made by `to_pytree`."""
        return cls.from_blocksparse(cls._sparse_cls().from_pytree(pytree))

    def _map_blocks(self, fn_block=None, fn_sector=None, fn_filter=None):
        """Map the blocks and their keys (sectors) of the array inplace."""
        x = self.to_blocksparse()
        x._map_blocks(fn_block, fn_sector, fn_filter)
        return self._become(type(self).from_blocksparse(x))

    # ------------------------- structural operations ----------------------- #

    def _transpose_abelian(self, axes=None, inplace=False):
        """Transpose this array, with no fermionic phases."""
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        else:
            axes = tuple(axes)
        new = self if inplace else self.copy()
        if axes == tuple(range(self.ndim)):
            return new

        indices, sectors, shapes, offsets, plan, order = cached_plan(
            ("transpose", self.structure_key(), axes),
            calc_plan_transpose,
            self._indices,
            self._sectors,
            self._shapes,
            self._offsets,
            axes,
        )
        phases = self._block_phases()
        if phases is not None:
            phases = phases.take(order)
        data = get_kernels().run_copy(plan, self._data, phases)
        return new._set_structure(indices, sectors, shapes, offsets, data)

    def _conj_abelian(self, inplace=False):
        """Complex conjugate, including the indices."""
        import numpy as np

        new = self if inplace else self.copy()
        if np.iscomplexobj(new._data):
            new._data = new._data.conj()
        return new._modify_abelian(
            indices=tuple(ix.conj() for ix in self._indices),
            charge=self._symmetry.sign(self._charge),
        )

    def _fuse_core_abelian(self, *axes_groups, mode="auto", inplace=False):
        axes_groups = tuple(tuple(g) for g in axes_groups if g)
        new = self if inplace else self.copy()
        indices, sectors, shapes, offsets, plan = cached_plan(
            ("fuse", self.structure_key(), axes_groups, self._symmetry),
            calc_plan_fuse,
            self._indices,
            self._sectors,
            self._shapes,
            self._offsets[:-1],
            self.duals,
            self._symmetry,
            axes_groups,
        )
        data = get_kernels().run_copy(plan, self._data, self._block_phases())
        return new._set_structure(indices, sectors, shapes, offsets, data)

    def _unfuse_abelian(self, axis, inplace=False):
        if axis < 0:
            axis += self.ndim
        new = self if inplace else self.copy()
        indices, sectors, shapes, offsets, plan, blk = cached_plan(
            ("unfuse", self.structure_key(), axis),
            calc_plan_unfuse,
            self._indices,
            self._sectors,
            self._shapes,
            self._offsets,
            axis,
        )
        phases = self._block_phases()
        if phases is not None:
            phases = phases.take(blk)
        data = get_kernels().run_copy(plan, self._data, phases)
        return new._set_structure(indices, sectors, shapes, offsets, data)

    def _mark_empty_fused(self, axis):
        """Mark the size-1 ``axis`` as the fusion of no indices at all, so that
        unfusing it removes the axis again.
        """
        ix = self._indices[axis]
        subinfo = PackedSubIndexInfo(
            indices=(),
            subkeys=[[]],
            charge_starts=[0, 1],
            suboffsets=[0],
            subsizes=[1],
        )
        self._indices = (
            *self._indices[:axis],
            ix.copy_with(subinfo=subinfo),
            *self._indices[axis + 1 :],
        )
        self._skey = None
        return self

    def expand_dims(self, axis, c=None, dual=None, inplace=False):
        """Expand the shape of a packed abelian array, see the sparse
        `expand_dims`.
        """
        import numpy as np

        x = self if inplace else self.copy()
        if axis < 0:
            axis += x.ndim + 1
        if dual is None:
            if axis > 0:
                dual = x._indices[axis - 1].dual
            elif axis < x.ndim:
                dual = x._indices[axis].dual
            else:
                dual = False
        if c is None:
            c = x._symmetry.combine()
            new_charge = x._charge
        else:
            new_charge = x._symmetry.combine(
                x._charge, x._symmetry.sign(c, dual)
            )
        sectors = np.insert(x._sectors, axis, 0, axis=1)
        shapes = np.insert(x._shapes, axis, 1, axis=1)
        indices = (
            *x._indices[:axis],
            PackedIndex(charges=(c,), sizes=(1,), dual=dual),
            *x._indices[axis:],
        )
        x._set_structure(
            indices, sectors, shapes, x._offsets, x._data, x._block_phases()
        )
        x._charge = new_charge
        return x

    def _squeeze_abelian(self, axis=None, inplace=False):
        """Remove axes of size 1, keeping the data as is."""
        import numpy as np

        x = self if inplace else self.copy()
        if axis is not None and np.ndim(axis) == 0:
            axis = (axis,)
        if axis is not None:
            axis = tuple(ax % x.ndim for ax in axis)

        keep = []
        new_charge = x._charge
        for ax, ix in enumerate(x._indices):
            if axis is None:
                remove = ix.size_total == 1
            else:
                remove = ax in axis
                if remove and ix.size_total > 1:
                    raise ValueError("Cannot squeeze d > 1 index")
            if remove:
                (charge,) = ix.charges
                new_charge = x._symmetry.combine(
                    new_charge, x._symmetry.sign(charge, not ix.dual)
                )
            else:
                keep.append(ax)

        x._set_structure(
            tuple(x._indices[ax] for ax in keep),
            x._sectors[:, keep],
            x._shapes[:, keep],
            x._offsets,
            x._data,
            x._block_phases(),
        )
        x._charge = new_charge
        return x

    def sync_charges(self, inplace=False):
        """Drop index charges that no block uses."""
        new = self if inplace else self.copy()
        indices, sectors = select_used_charges(new._indices, new._sectors)
        new._indices = indices
        new._sectors = sectors
        new._skey = None
        return new

    # ----------------------------- contraction ----------------------------- #

    def _tensordot_abelian(
        self, other, axes=2, mode="auto", preserve_array=False
    ):
        return tensordot_packed(
            self, other, axes=axes, preserve_array=preserve_array
        )

    def _matmul_abelian(self, other, preserve_array=False):
        if self.ndim > 2 or other.ndim > 2:
            raise ValueError("Only 1D and 2D arrays supported.")
        return tensordot_packed(
            self,
            other,
            axes=((self.ndim - 1,), (0,)),
            preserve_array=preserve_array,
        )

    def multiply_diagonal(self, v, axis, power=1, inplace=False):
        """Multiply this array by the vector ``v`` as if contracting a diagonal
        matrix along ``axis``.

        Parameters
        ----------
        v : PackedVector
            The vector to contract with.
        axis : int
            The axis along which to contract.
        power : {1, -1}, optional
            Whether to multiply or divide by the vector.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        PackedArrayCommon
        """
        if power not in (1, -1):
            raise ValueError(f"Unsupported power {power}, expected 1 or -1.")
        if not isinstance(v, PackedVector):
            v = PackedVector(v.blocks)
        if not -self.ndim <= axis < self.ndim:
            raise ValueError(f"Invalid axis {axis} for {self.ndim}D array.")
        if axis < 0:
            axis += self.ndim

        x = self if inplace else self.copy()
        kernels = get_kernels()
        select, keep, plan = cached_plan(
            ("muldiag", self.structure_key(), axis, v.structure_key()),
            calc_plan_muldiag,
            self,
            v,
            axis,
        )
        data = self._data
        if select is not None:
            if power == -1:
                raise ZeroDivisionError(
                    "Cannot divide by implicitly zero (missing) block."
                )
            indices, sectors, shapes, offsets, splan = select
            phases = self._block_phases()
            if phases is not None:
                phases = phases[keep]
            data = kernels.run_copy(splan, data, phases)
            x._set_structure(indices, sectors, shapes, offsets, data)
        x._data = kernels.run_muldiag(plan, data, v._data, power)
        return x

    def ldmul(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-2, inplace=inplace)

    def rdmul(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-1, inplace=inplace)

    def lddiv(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-2, power=-1, inplace=inplace)

    def rddiv(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-1, power=-1, inplace=inplace)

    # ------------------------ elementwise operations ----------------------- #

    def _binary_blockwise_op_abelian(
        self, other, fn, missing=None, inplace=False
    ):
        if self.structure_key() == other.structure_key():
            new = self if inplace else self.copy()
            new._data = fn(self._data, other._data)
            return new
        return self._via_blocksparse(
            "_binary_blockwise_op",
            other,
            fn=fn,
            missing=missing,
            inplace=inplace,
        )

    def _allclose_abelian(self, other, **allclose_opts):
        import numpy as np

        if (
            isinstance(other, PackedArrayCommon)
            and self.structure_key() == other.structure_key()
            and self._charge == other._charge
        ):
            return bool(np.allclose(self._data, other._data, **allclose_opts))
        return self.to_blocksparse().allclose(
            blocksparse_from_any(other), **allclose_opts
        )

    def _test_allclose_abelian(self, other, **allclose_opts):
        return self.to_blocksparse().test_allclose(
            blocksparse_from_any(other), **allclose_opts
        )

    # --------------- operations run via the block sparse backend ----------- #

    def _to_dense_abelian(self, index_maps=None):
        return self._via_blocksparse(
            "_to_dense_abelian", index_maps=index_maps
        )

    def _einsum_abelian(self, eq, preserve_array=False):
        return self._via_blocksparse(
            "_einsum_abelian", eq, preserve_array=preserve_array
        )

    def _trace_abelian(self):
        return self._via_blocksparse("_trace_abelian")

    def select_charge(self, axis, charge, subselect=None, inplace=False):
        """Restrict ``axis`` to a single charge, see the sparse
        `select_charge`.
        """
        return self._via_blocksparse(
            "select_charge", axis, charge, subselect=subselect, inplace=inplace
        )

    def isel(self, axis, idx, inplace=False):
        """Select a single linear index along ``axis``."""
        return self._via_blocksparse("isel", axis, idx, inplace=inplace)

    def align_axes(self, other, axes):
        """Drop blocks of this array and ``other`` whose subsectors on ``axes``
        are not on both. ``axes`` is given as for tensordot. The axes then fuse
        to matching indices.
        """
        import numpy as np

        axes_a, axes_b = axes
        radices = np.array(
            [self._indices[ax].num_charges for ax in axes_a], dtype=np.int64
        )
        sub_b = np.empty((other.num_blocks, len(axes_b)), dtype=np.int64)
        for j, (axa, axb) in enumerate(zip(axes_a, axes_b)):
            ids = self._indices[axa].charge_ids()
            remap = np.array(
                [ids.get(c, -1) for c in other._indices[axb].charges],
                dtype=np.int64,
            )
            sub_b[:, j] = remap.take(other._sectors[:, axb])
        keep_a, keep_b = _kernels_for_keys(radices).tensordot_align(
            self._sectors[:, list(axes_a)], sub_b, radices
        )
        return self._select_blocks(keep_a), other._select_blocks(keep_b)

    def _select_blocks(self, keep):
        """Copy keeping only the blocks where ``keep`` is True, and only the
        charges they use.
        """
        new = self.copy()
        if keep.all():
            return new.sync_charges(inplace=True)
        indices, sectors, shapes, offsets, plan = cached_plan(
            ("select", self.structure_key(), keep.tobytes()),
            calc_plan_select,
            self._indices,
            self._sectors,
            self._shapes,
            self._offsets,
            keep,
        )
        phases = self._block_phases()
        data = get_kernels().run_copy(
            plan, self._data, None if phases is None else phases[keep]
        )
        new._set_structure(indices, sectors, shapes, offsets, data)
        return new.sync_charges(inplace=True)

    def gen_valid_sectors(self):
        """Generate all valid sectors for this array."""
        return self._via_blocksparse("gen_valid_sectors")

    def get_sparsity(self):
        """The number of blocks over the number of possible blocks."""
        return self._via_blocksparse("get_sparsity")

    def fill_missing_blocks(self):
        """Insert zero blocks for all missing valid sectors, inplace."""
        x = self.to_blocksparse()
        x.fill_missing_blocks()
        return self._become(packed_from_any(x))

    def drop_missing_blocks(self):
        """Drop blocks that are all zero, inplace."""
        x = self.to_blocksparse()
        x.drop_missing_blocks()
        return self._become(packed_from_any(x))

    def check_with(self, other, *args):
        """Check compatibility with another array or vector, see the sparse
        `check_with`.
        """
        return self._via_blocksparse("check_with", other, *args)

    def _split_abelian(
        self,
        *,
        fn=None,
        charge_side="auto",
        max_bond_mode="global",
        **kwargs,
    ):
        """Decompose this 2D packed array block by block, see the sparse
        `_split_abelian`.
        """
        import numpy as np

        from ..linalg_common import Absorb, array_split
        from ..sparse.sparse_array_common import calc_sub_max_bonds

        if self.ndim != 2:
            raise NotImplementedError(
                "split only implemented for 2D arrays,"
                f" got {self.ndim}D. Consider fusing first."
            )
        if max_bond_mode not in ("global", "eager"):
            raise ValueError(
                "max_bond_mode must be 'global' or 'eager', "
                f"got {max_bond_mode!r}"
            )
        if ("max_bond" in kwargs) and (kwargs["max_bond"] is None):
            kwargs["max_bond"] = -1

        max_bond = kwargs.get("max_bond", -1)
        eager_max_bond = (max_bond_mode == "eager") and (max_bond > 0)
        if (
            (kwargs.get("cutoff", -1.0) > 0.0)
            or (kwargs.get("renorm", 0) > 0)
            or ((max_bond > 0) and (max_bond_mode == "global"))
        ):
            # compute the required spectrum first, *then* truncate / absorb
            return self._split_abelian_dynamic(
                fn=fn,
                charge_side=charge_side,
                max_bond_mode=max_bond_mode,
                **kwargs,
            )

        charge_side = Absorb.choose_charge_side(kwargs["absorb"], charge_side)
        if fn is None:
            fn = array_split

        blocks = self._phased_blocks()
        nb = len(blocks)

        if eager_max_bond and nb:
            sub_max_bonds = calc_sub_max_bonds(
                tuple(min(block.shape) for block in blocks), max_bond
            )
        else:
            sub_max_bonds = (None,) * nb

        # bond takes the right index charge if the array charge goes with the
        # left factor, and vice versa
        bond_axis = 1 if charge_side == "left" else 0
        bond_ids = self._sectors[:, bond_axis].tolist()
        lefts, ss, rights, done, bond_sizes = [], [], [], [], []
        for i, block in enumerate(blocks):
            if sub_max_bonds[i] == 0:
                continue
            if sub_max_bonds[i] is None:
                split_kwargs = kwargs
            else:
                split_kwargs = {**kwargs, "max_bond": sub_max_bonds[i]}
            left, s, right = fn(block, **split_kwargs)
            if left is not None:
                k = left.shape[-1]
                lefts.append(left)
            elif right is not None:
                k = right.shape[-2]
            else:
                k = s.shape[-1]
            if right is not None:
                rights.append(right)
            if s is not None:
                ss.append(s)
            done.append(i)
            bond_sizes.append(k)

        ixl, ixr = self._indices
        src = ixr if charge_side == "left" else ixl
        done = np.array(done, dtype=np.int64)
        src_ids = np.array(bond_ids, dtype=np.int64).take(done)
        order = np.argsort(src_ids, kind="stable")
        bond_sizes = np.array(bond_sizes, dtype=np.int64)
        ixb = PackedIndex(
            charges=[src.charges[i] for i in src_ids[order].tolist()],
            sizes=bond_sizes[order],
            dual=ixr.dual if charge_side == "left" else not ixl.dual,
        )
        # new bond charge id of each split block
        rank = np.empty(len(done), dtype=np.int64)
        rank[order] = np.arange(len(done))
        outer = self._sectors.take(done, axis=0)

        def build(arrays, factor_sectors, factor_indices, keep_charge):
            if not arrays and nb:
                return None
            perm = row_order(factor_sectors, radices_of(factor_indices))
            factor_sectors = factor_sectors[perm]
            perm = perm.tolist()
            factor_shapes = np.array(
                [arrays[j].shape for j in perm], dtype=np.int64
            ).reshape(-1, 2)
            if arrays:
                factor_data = np.concatenate(
                    [arrays[j].reshape(-1) for j in perm]
                )
            else:
                factor_data = self._data[:0]
            factor_offsets = calc_offsets(factor_shapes)
            if keep_charge:
                new = self.copy()
                return new._set_structure(
                    factor_indices,
                    factor_sectors,
                    factor_shapes,
                    factor_offsets,
                    factor_data,
                )
            return self.new_with(
                indices=factor_indices,
                charge=self._symmetry.combine(),
                data=factor_data,
                sectors=factor_sectors,
                shapes=factor_shapes,
                offsets=factor_offsets,
            )

        # left factor maps the left charge to the bond, right factor the bond
        # to the right charge, whichever side carries the array charge
        left_sectors = np.stack([outer[:, 0], rank], axis=1)
        right_sectors = np.stack([rank, outer[:, 1]], axis=1)

        left = build(lefts, left_sectors, (ixl, ixb), charge_side == "left")
        right = build(
            rights, right_sectors, (ixb.conj(), ixr), charge_side == "right"
        )

        if ss:
            s_order = order.tolist()
            s_data = np.concatenate([ss[j].reshape(-1) for j in s_order])
            s_offsets = np.zeros(len(ss) + 1, dtype=np.int64)
            np.cumsum([ss[j].size for j in s_order], out=s_offsets[1:])
            s = PackedVector.from_parts(ixb.charges, s_data, s_offsets)
        else:
            s = None

        if DEBUG:
            for x in (left, right, s):
                if x is not None:
                    x.check()

        return left, s, right

    def _split_abelian_dynamic(
        self,
        *,
        fn=None,
        charge_side="auto",
        max_bond_mode="global",
        **kwargs,
    ):
        """Split by computing the whole spectrum first, then truncating it
        globally, see the sparse `_split_abelian_dynamic`.
        """
        from ..linalg_common import CUTOFF_MODE_MAP

        cutoff = kwargs.pop("cutoff", 0.0)
        cutoff_mode = CUTOFF_MODE_MAP[kwargs.pop("cutoff_mode", "rel")]
        renorm = kwargs.pop("renorm", False)
        max_bond = kwargs.pop("max_bond", -1)
        absorb = kwargs.pop("absorb", "both")
        use_abs = kwargs.pop("use_abs", None)

        if max_bond_mode == "eager":
            if renorm:
                raise ValueError(
                    "renorm is not compatible with max_bond_mode='eager'"
                )
            if (cutoff > 0.0) and (cutoff_mode not in (1, 2)):
                raise ValueError(
                    "cumulative cutoff modes are not compatible with "
                    "max_bond_mode='eager'"
                )
            initial_max_bond = max_bond
            final_max_bond = -1
        else:
            initial_max_bond = -1
            final_max_bond = max_bond

        left, s, right = self._split_abelian(
            fn=fn,
            charge_side=charge_side,
            cutoff=0.0,
            renorm=False,
            max_bond=initial_max_bond,
            absorb=None,
            max_bond_mode=max_bond_mode,
            **kwargs,
        )

        if use_abs is None:
            positive = kwargs.get("positive", None)
            if positive is None:
                use_abs = False
            elif positive:
                use_abs = False
                # very small -ve eigenvalues can still cause problems
                s = s.clip(0.0, None)
            else:
                use_abs = True

        return truncate_svd_result_packed(
            left,
            s,
            right,
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=final_max_bond,
            absorb=absorb,
            renorm=renorm,
            use_abs=use_abs,
        )

    def _eigh_abelian(self):
        """Eigen decompose this 2D array block by block. The array should be
        Hermitian. See the sparse `_eigh_abelian`.
        """
        import numpy as np

        if self.ndim != 2:
            raise NotImplementedError(
                "eigh only implemented for 2D arrays,"
                f" got {self.ndim}D. Consider fusing first."
            )
        if self.charge != self._symmetry.combine():
            raise ValueError(
                "Total charge must be the identity (zero) element."
            )

        evals, evecs = [], []
        for block in self._phased_blocks():
            w, v = np.linalg.eigh(block)
            evals.append(w)
            evecs.append(v.reshape(-1))

        # eigenvalues take the charge of the right index
        ids = self._sectors[:, 1]
        order = np.argsort(ids, kind="stable").tolist()
        charges = self._indices[1].charges
        w_offsets = np.zeros(len(evals) + 1, dtype=np.int64)
        np.cumsum([evals[j].size for j in order], out=w_offsets[1:])
        if evals:
            w_data = np.concatenate([evals[j] for j in order])
        else:
            w_data = np.zeros(0)
        w = PackedVector.from_parts(
            [charges[i] for i in ids.take(order).tolist()], w_data, w_offsets
        )

        u = self.copy()
        u._set_structure(
            self._indices,
            self._sectors,
            self._shapes,
            self._offsets,
            np.concatenate(evecs) if evecs else self._data[:0],
        )

        if DEBUG:
            u.check()
            w.check()

        return w, u

    def _cholesky_abelian(self, upper=False, shift=True, **kwargs):
        """Cholesky decompose this 2D packed array block by block, see the
        sparse `_cholesky_abelian`.
        """
        import numpy as np

        from ..linalg_common import array_split

        if self.ndim != 2:
            raise NotImplementedError(
                "Cholesky decomposition only implemented for 2D "
                f"arrays, got {self.ndim}D. Consider fusing first."
            )
        if self.charge != self._symmetry.combine():
            raise ValueError(
                "Total charge must be the identity (zero) element."
            )

        factors = []
        for block in self._phased_blocks():
            left, _, right = array_split(
                block,
                absorb="rsqrt" if upper else "lsqrt",
                shift=shift,
                method="cholesky",
                **kwargs,
            )
            factor = left if right is None else right
            factors.append(factor.reshape(-1))

        new = self.copy()
        new._set_structure(
            self._indices,
            self._sectors,
            self._shapes,
            self._offsets,
            np.concatenate(factors) if factors else self._data[:0],
        )

        if DEBUG:
            new.check()

        return new

    def _solve_abelian(self, b):
        """Solve ``self @ x == b`` for ``x`` block by block, see the sparse
        `_solve_abelian`.
        """
        import numpy as np

        if self.ndim != 2 or b.ndim not in (1, 2):
            raise NotImplementedError(
                "solve only implemented for 2D and 1D or 2D and 2D "
                f"arrays, got {self.ndim}D and {b.ndim}D. "
                "Consider fusing first."
            )

        # b has a fixed charge, so each row charge has at most one block
        b_sectors = b._sectors.tolist()
        b_charges = b._indices[0].charges
        b_rows = {b_charges[row[0]]: j for j, row in enumerate(b_sectors)}
        b_blocks = b._phased_blocks()
        a_blocks = self._phased_blocks()
        a_charges = self._indices[0].charges

        x_sectors, x_blocks = [], []
        for i, (r, c) in enumerate(self._sectors.tolist()):
            j = b_rows.get(a_charges[r])
            if j is not None:
                x_sectors.append([c, *b_sectors[j][1:]])
                x_blocks.append(np.linalg.solve(a_blocks[i], b_blocks[j]))

        x_indices = (self._indices[1].conj(), *b._indices[1:])
        x_sectors = np.array(x_sectors, dtype=np.int64).reshape(-1, b.ndim)
        order = row_order(x_sectors, radices_of(x_indices))
        x_sectors = x_sectors[order]
        order = order.tolist()
        x_shapes = np.array(
            [x_blocks[j].shape for j in order], dtype=np.int64
        ).reshape(-1, b.ndim)
        if x_blocks:
            x_data = np.concatenate([x_blocks[j].reshape(-1) for j in order])
        else:
            x_data = b._data[:0]

        # c_x = c_b - c_A
        sym = self._symmetry
        x = b.copy()
        x._set_structure(
            x_indices, x_sectors, x_shapes, calc_offsets(x_shapes), x_data
        )
        x._charge = sym.combine(b.charge, sym.sign(self.charge))

        if DEBUG:
            x.check()

        return x

    def _become(self, other):
        """Take on all the attributes of ``other`` inplace."""
        for cls in type(self).__mro__:
            for name in getattr(cls, "__slots__", ()):
                if hasattr(other, name):
                    setattr(self, name, getattr(other, name))
        return self


def _run_or_scale(plan, data, phases, offsets, kept=None):
    """Run a copy plan, applying ``phases``. With no plan, just apply the
    phases. ``kept`` gives the blocks the plan copies, if not all.
    """

    if plan is not None:
        if phases is not None and kept is not None:
            phases = phases.take(kept)
        return get_kernels().run_copy(plan, data, phases)
    if phases is not None:
        return get_kernels().run_scale(data, offsets, phases)
    return data


def ar_ravel(x):
    import numpy as np

    return np.asarray(x).reshape(-1)


def tensordot_packed(a, b, axes=2, preserve_array=False):
    """Tensordot between two packed abelian arrays.

    Parameters
    ----------
    a, b : PackedArrayCommon
        The arrays to contract.
    axes : int or tuple[tuple[int]]
        The axes to contract, as for `numpy.tensordot`.
    preserve_array : bool, optional
        Whether to return a 0-dimensional array rather than a scalar for a full
        contraction.
    """
    import numpy as np

    if not isinstance(b, PackedArrayCommon):
        if getattr(b, "ndim", 0) == 0:
            return a * b
        raise TypeError(f"Expected PackedArrayCommon, got {type(b)}.")

    if DEBUG:
        a.check()
        b.check()

    left_axes, axes_a, axes_b, right_axes = parse_tensordot_axes(
        axes, a.ndim, b.ndim
    )
    kernels = get_kernels()
    charge = a._symmetry.combine(a._charge, b._charge)
    label = maybe_keep_label(a._label, b._label)
    dtype = np.result_type(a._data, b._data)

    if not axes_a:
        a = a.phase_sync() if a.fermionic else a
        b = b.phase_sync() if b.fermionic else b
        pairs, (indices, sectors, shapes, offsets) = cached_plan(
            ("outer", a.structure_key(), b.structure_key()),
            calc_plan_outer,
            a,
            b,
        )
        data = np.empty(int(offsets[-1]), dtype=dtype)
        for ao, na, bo, nb, co in pairs:
            np.multiply.outer(
                a._data[ao : ao + na],
                b._data[bo : bo + nb],
                out=data[co : co + na * nb].reshape(na, nb),
            )
    else:
        (
            plan_a,
            plan_b,
            kept_a,
            kept_b,
            gemms,
            is_scalar,
            (indices, sectors, shapes, offsets),
        ) = cached_plan(
            (
                "tensordot",
                a.structure_key(),
                b.structure_key(),
                left_axes,
                axes_a,
                axes_b,
                right_axes,
            ),
            calc_plan_tensordot,
            a,
            b,
            left_axes,
            axes_a,
            axes_b,
            right_axes,
        )
        da = _run_or_scale(
            plan_a, a._data, a._block_phases(), a._offsets, kept_a
        )
        db = _run_or_scale(
            plan_b, b._data, b._block_phases(), b._offsets, kept_b
        )

        if is_scalar:
            # sum the contributions of every aligned charge
            data = np.zeros(min(len(gemms), 1), dtype=dtype)
            for ao, k, bo in gemms:
                data += np.dot(da[ao : ao + k], db[bo : bo + k])
        else:
            data = kernels.run_gemm(gemms, da, db, int(offsets[-1]))

    c = a.new_with(
        indices=indices,
        charge=charge,
        data=data,
        sectors=sectors,
        shapes=shapes,
        offsets=offsets,
        label=label,
    )

    if DEBUG:
        c.check()

    if (c.ndim == 0) and (not preserve_array):
        if c.num_blocks == 0:
            return 0.0
        return c._data[0]

    return c
