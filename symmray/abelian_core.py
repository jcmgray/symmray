"""Blocks arrays with an abelian symmetry constraint."""

import contextlib
import functools
import hashlib
import itertools
import math
import operator
import os
import pickle
import warnings
from collections import OrderedDict, defaultdict

import autoray as ar
from autoray.lazy.core import find_full_reshape

from .block_core import BlockCommon, BlockVector
from .interface import tensordot
from .symmetries import Symmetry, get_symmetry
from .utils import DEBUG


def hasher(k):
    return hashlib.sha1(pickle.dumps(k)).hexdigest()


class BlockIndex:
    """An index of a block sparse, abelian symmetric tensor. This is intended
    to be used immutably.

    Parameters
    ----------
    chargemap : dict[int, int]
        A mapping from charge to size.
    dual : bool, optional
        Whether the index is 'dual' or not, i.e. whether the flow is
        'outwards' / (+ve) / ket-like = ``False`` or 'inwards' / (-ve) /
        bra-like = ``True``. The sign of charge contributions is then
        ``(-1) ** dual``.
    subinfo : SubIndexInfo, optional
        Information about the subindices of this index and their extents if
        this index was formed from fusing.
    """

    __slots__ = ("_chargemap", "_dual", "_subinfo", "_hashkey")

    def __init__(self, chargemap, dual=False, subinfo=None):
        # ensure always sorted
        if not isinstance(chargemap, dict):
            self._chargemap = dict(sorted(chargemap))
        else:
            self._chargemap = dict(sorted(chargemap.items()))
        self._dual = bool(dual)
        self._subinfo = subinfo
        self._hashkey = None

    @property
    def chargemap(self):
        """A mapping from charge to size."""
        return self._chargemap

    @property
    def dual(self):
        """Whether the index flows 'outwards' / (+ve) / ket-like = ``False`` or
        'inwards' / (-ve) / bra-like= ``True``. The charge sign is given by
        ``(-1) ** dual``.
        """
        return self._dual

    @property
    def subinfo(self):
        """Information about the subindices of this index and their extents if
        this index was formed from fusing.
        """
        return self._subinfo

    @property
    def subshape(self):
        if self._subinfo is None:
            return None
        return self._subinfo.subshape

    @property
    def charges(self):
        """The charges of this index."""
        return self._chargemap.keys()

    @property
    def sizes(self):
        """The sizes of the blocks of this index."""
        return self._chargemap.values()

    @property
    def size_total(self):
        """The total size of this index, i.e. the sum of the sizes of all
        blocks.
        """
        return sum(self._chargemap.values())

    @property
    def num_charges(self):
        """The number of charges."""
        return len(self._chargemap)

    def copy_with(self, chargemap=None, dual=None, subinfo=None):
        """A copy of this index with some attributes replaced. Note that checks
        are not performed on the new propoerties, this is intended for internal
        use.
        """
        new = self.__new__(self.__class__)
        new._chargemap = (
            self._chargemap.copy()
            if chargemap is None
            else (
                dict(sorted(chargemap.items()))
                if isinstance(chargemap, dict)
                else dict(sorted(chargemap))
            )
        )
        new._dual = self._dual if dual is None else dual
        new._subinfo = self._subinfo if subinfo is None else subinfo
        new._hashkey = None
        return new

    def conj(self):
        """A copy of this index with the dualness reversed."""
        dual = not self.dual
        subinfo = None if self.subinfo is None else self.subinfo.conj()
        return self.copy_with(dual=dual, subinfo=subinfo)

    def drop_charges(self, charges):
        """Return a new index with all charges in ``charges`` removed.

        Parameters
        ----------
        charges : Sequence[hashable]
            The charges to remove.

        Returns
        -------
        BlockIndex
            A new index with the charges removed.
        """
        return self.copy_with(
            chargemap={
                c: d for c, d in self._chargemap.items() if c not in charges
            },
            subinfo=(
                None
                if self.subinfo is None
                else self.subinfo.drop_charges(charges)
            ),
        )

    def select_charge(self, charge):
        """Drop all but the specified charge from this index."""
        drop = set(self._chargemap)
        drop.remove(charge)
        return self.drop_charges(drop)

    def size_of(self, c):
        """The size of the block with charge ``c``."""
        return self._chargemap[c]

    def check(self):
        """Check that the index is well-formed, i.e. all sizes are positive."""
        for c, d in self._chargemap.items():
            if d <= 0:
                raise ValueError(
                    f"Size of charge {c} is {d}, must be positive."
                )
            if not isinstance(d, int):
                raise ValueError(f"Size of charge {c} is {d}, must be an int.")

        assert sorted(self._chargemap) == list(self._chargemap)

        if self.subinfo:
            assert self.size_total == sum(
                d
                for extent in self.subinfo.extents.values()
                for d in extent.values()
            )

    def matches(self, other):
        """Whether this index matches ``other`` index, namely, whether the
        ``chargemap`` of each matches, their dualnesses are opposite, and also
        whether their subindices match, if they have any. For debugging.

        Parameters
        ----------
        other : BlockIndex
            The other index to compare to.
        """
        return (
            dicts_dont_conflict(self._chargemap, other._chargemap)
            and (self.dual ^ other.dual)
            and (
                (self.subinfo is other.subinfo is None)
                or (self.subinfo.matches(other.subinfo))
            )
        )

    def hashkey(self):
        """Get a hash key for this index."""
        if getattr(self, "_hashkey", None) is None:
            self._hashkey = hasher(
                (
                    tuple(self._chargemap.items()),
                    self._dual,
                    self._subinfo.hashkey() if self._subinfo else None,
                )
            )
        return self._hashkey

    def __hash__(self):
        # disable as hash(-1) == hash(-2) causes problems
        # XXX: use hashkey?
        raise NotImplementedError

    def __str__(self):
        lines = [
            f"({self.size_total} = "
            f"{'+'.join(map(str, self._chargemap.values()))} "
            f": {'-' if self.dual else '+'}"
            f"[{','.join(map(str, self._chargemap.keys()))}])"
        ]

        if self.subinfo:
            for charge, extent in sorted(self.subinfo.extents.items()):
                subcharges = extent.keys()
                subsizes = extent.values()
                lines.append(
                    f"    {charge} ; "
                    f"({'+'.join(map(str, subsizes))}) : "
                    f"[{','.join(map(str, subcharges))}]"
                )

        return "\n".join(lines)

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"chargemap={self._chargemap}, dual={self.dual}",
                (
                    f", subinfo={self.subinfo}"
                    if self.subinfo is not None
                    else ""
                ),
                ")",
            ]
        )


def dicts_dont_conflict(da, db):
    """Check that two dictionaries don't conflict, i.e. they have no keys in
    common with different values."""
    for k, va in da.items():
        vb = db.get(k, None)
        if vb is not None and va != vb:
            return False
    return True


class SubIndexInfo:
    """Holder class for storing the relevant information for unfusing.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices (ordered) that were fused to make this index.
    extents : dict[hashable, dict[hashable, int]]
        A mapping of each charge of the fused index to a mapping of each
        subsector (combination of sub charges) to the size of that subsector.
        This should not be mutated after creation.
    """

    __slots__ = ("_extents", "_indices", "_hashkey")

    def __init__(self, indices, extents):
        self._indices = indices
        self._extents = extents
        self._hashkey = None

    @property
    def indices(self):
        """The indices that were fused to make this index."""
        return self._indices

    @property
    def extents(self):
        """A mapping of each charge of the fused index to a mapping of each
        subsector (combination of sub charges) to the size of that subsector.
        """
        return self._extents

    @property
    def subshape(self):
        return tuple(ix.size_total for ix in self._indices)

    def copy_with(self, indices=None, extents=None):
        """A copy of this subindex information with some attributes replaced.
        Note that checks are not performed on the new properties, this is
        intended for internal use.
        """
        new = self.__new__(self.__class__)
        new._indices = self._indices if indices is None else indices
        new._extents = self._extents if extents is None else extents
        new._hashkey = None
        return new

    def conj(self):
        """A copy of this subindex information with the relevant dualnesses
        reversed.
        """
        return self.copy_with(indices=tuple(ix.conj() for ix in self._indices))

    def drop_charges(self, charges):
        """Get a copy of this subindex information with the charges in
        ``charges`` discarded.
        """
        return self.copy_with(
            extents={
                c: extent
                for c, extent in self._extents.items()
                if c not in charges
            },
        )

    def matches(self, other):
        """Whether this subindex information matches ``other`` subindex
        information, namely, whether the ``indices`` and ``extents`` match.
        For debugging.
        """
        return all(
            i.matches(j) for i, j in zip(self._indices, other._indices)
        ) and dicts_dont_conflict(self._extents, other._extents)

    def hashkey(self):
        """Get a string hash key for this subindex information. This is cached
        after the first call.
        """
        if getattr(self, "_hashkey", None) is None:
            self._hashkey = hasher(
                (
                    tuple(ix.hashkey for ix in self._indices),
                    tuple(
                        (c, tuple(extent.items()))
                        for c, extent in self._extents.items()
                    ),
                )
            )
        return self._hashkey

    def __hash__(self):
        # disable as hash(-1) == hash(-2) causes problems
        # XXX: use hashkey?
        raise NotImplementedError

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"indices={self._indices}, ",
                f"extents={self._extents}",
                ")",
            ]
        )


# --------------------------------------------------------------------------- #


def permuted(it, perm):
    """Return a tuple of the elements in ``it`` (which should be indexable),
    in the order given by ``perm``.

    Examples
    --------

        >>> permuted(['a', 'b', 'c', 'd'], [3, 1, 0, 2])
        ('d', 'b', 'a', 'c')

    """
    return tuple(it[p] for p in perm)


def without(it, remove):
    """Return a tuple of the elements in ``it`` with those at positions
    ``remove`` removed.
    """
    return tuple(el for i, el in enumerate(it) if i not in remove)


def replace_with_seq(it, index, seq):
    """Return a tuple, with the item at ``index`` in ``it`` replaced by the
    items in ``seq``.
    """
    return (*it[:index], *seq, *it[index + 1 :])


def accum_for_split(sizes):
    """Take a sequence of block sizes and return the sequence of linear
    partitions, suitable for use with the ``split`` function.
    """
    x = [0]
    s = []
    for size in sizes:
        x.append(x[-1] + size)
        s.append(slice(x[-2], x[-1]))
    return s


@functools.lru_cache(maxsize=2**15)
def calc_reshape_args(shape, newshape, subshapes):
    """Given a current block sparse shape ``shape`` a target shape ``newshape``
    and current sub index sizes ``subshapes`` (i.e. previously fused dimensions)
    compute the sequence of axes to unfuse, fuse and expand to reshape the
    array.

    Parameters
    ----------
    shape : tuple[int]
        The current shape of the array.
    newshape : tuple[int]
        The target shape of the array.
    subshapes : tuple[None or tuple[int]]
        The sizes of the subindices that were previously fused.

    Returns
    -------
    axs_unfuse : tuple[int]
        The axes to unfuse.
    axs_fuse : tuple[tuple[tuple[int]]]
        The axes (after unfusing) to fuse, grouped by contiguous groups.
    axs_expand : tuple[int]
        The axes (after unfusing and fusing) to expand.
    """
    # tracks position in input shape
    i = 0
    # tracks position in output shape
    j = 0
    # tracks position in post-fuse / pre-expand shape
    k = 0

    ndim_old = len(shape)
    ndim_new = len(newshape)

    term = []  # dnyamically updated labelled dimensions
    axs_squeeze = []
    unfuse_sizes = {}
    fuse_sizes = {}
    axs_expand = []
    any_singleton = False
    any_fused = False

    while i < ndim_old and j < ndim_new:
        di = shape[i]
        dj = newshape[j]

        if (subshapes[i] is not None) and (
            subshapes[i] == newshape[j : j + len(subshapes[i])]
        ):
            # unfuse, check first
            label = f"u{len(unfuse_sizes)}"
            s = 0
            for ds in subshapes[i]:
                dj = newshape[j]
                if ds != dj:
                    raise ValueError("Shape mismatch for unfuse.")
                s += 1
                j += 1
                k += 1
            unfuse_sizes[label] = s
            term.append(label)
            i += 1
        elif di == dj:
            # output dimension already
            term.append("o")
            i += 1
            j += 1
            k += 1
        elif di == 1:
            # have to handle squeezed dimensions after unfusing
            term.append("s")
            axs_squeeze.append(i)
            any_singleton = True
            i += 1
        elif dj == 1:
            # record expansion location relative to *post* fuse shape
            axs_expand.append(k)
            j += 1
        elif di < dj:
            # need to fuse
            label = f"g{len(fuse_sizes)}"
            term.append(label)
            s = 1
            i += 1
            while di < dj:
                di *= shape[i]
                term.append(label)
                i += 1
                s += 1
            if di != dj:
                raise ValueError("Shape mismatch for fuse.")
            fuse_sizes[label] = s
            any_fused = True
            j += 1
            k += 1
        else:
            raise ValueError("Shape mismatch.")

    # check trailing dimensions, which should be size 1
    for i in range(i, ndim_old):
        any_singleton = True
        term.append("s")
    for j in range(j, ndim_new):
        axs_expand.append(k)

    # first we handle unfusings
    axs_unfuse = []
    for label, s in unfuse_sizes.items():
        ax = term.index(label)
        axs_unfuse.append(ax)
        term = term[:ax] + ["o"] * s + term[ax + 1 :]

    # handle squeezes by converting them into fuse groups
    if any_singleton:
        i = 0
        label = term[i]
        if label == "s":
            # if we have squeeze axes on left, we have to group into right
            while label == "s":
                i += 1
                label = term[i]

            if label[0] == "g":
                # adjacent to existing group
                g = label
            elif label == "o":
                # or new group
                g = f"g{len(fuse_sizes)}"
                term[i] = g
                fuse_sizes[g] = 1

            # mark all axs up to this point
            for j in range(0, i):
                fuse_sizes[g] += 1
                term[j] = g

        # process rest of term, now preferring grouping into left
        i += 1
        while i < len(term):
            label = term[i]
            if label == "s":
                left = term[i - 1]
                if left[0] == "g":
                    g = left
                elif left == "o":
                    g = f"g{len(fuse_sizes)}"
                    term[i - 1] = g
                    fuse_sizes[g] = 1

                # update any right block of squeeze axs to g
                while label == "s":
                    term[i] = g
                    fuse_sizes[g] += 1
                    i += 1
                    if i == len(term):
                        break
                    label = term[i]
            i += 1

    # now we handle fusing
    axs_fuse = []
    if any_fused or any_singleton:
        # complexity here is we want to simulteneously fuse adjacent groups for
        # efficiency, but also need to handle non-adjacent groups
        current_groups = []
        i = 0
        while i < len(term):
            label = term[i]
            if label not in fuse_sizes:
                if current_groups:
                    # start of groups
                    i0 = i - sum(map(len, current_groups))
                    ng = len(current_groups)
                    term = term[:i0] + ["o"] * ng + term[i:]
                    axs_fuse.append(tuple(current_groups))
                    current_groups = []
                    # rewind to end of new group(s)
                    i = i0 + ng
                else:
                    i += 1
                continue

            s = fuse_sizes[label]
            current_groups.append(tuple(range(i, i + s)))
            i += s
        if current_groups:
            axs_fuse.append(tuple(current_groups))

    # handle expansion
    axs_expand.reverse()

    return tuple(axs_unfuse), tuple(axs_fuse), tuple(axs_expand)


@functools.lru_cache(2**14)
def calc_fuse_group_info(axes_groups, duals):
    """Calculate the fusing information just to do with axes groups
    (not any specific blocks).
    """
    ndim = len(duals)

    # which group does each axis appear in, if any
    ax2group = {}
    # whether each group is overall dual
    group_duals = []
    # whether the group has only a single axis
    group_singlets = []
    for g, gaxes in enumerate(axes_groups):
        # take the dual-ness of the group to match the first axis
        group_duals.append(duals[gaxes[0]])
        for ax in gaxes:
            ax2group[ax] = g
        if len(gaxes) == 1:
            group_singlets.append(g)
    # assign `None` to ungrouped axes
    for i in range(ndim):
        ax2group.setdefault(i, None)

    # the permutation will be the same for every block: precalculate
    # n.b. all new groups will be inserted at the *first fused axis*:
    position = min((min(gaxes) for gaxes in axes_groups))
    axes_before = tuple(ax for ax in range(position) if ax2group[ax] is None)
    axes_after = tuple(
        ax for ax in range(position, ndim) if ax2group[ax] is None
    )
    perm = (
        *axes_before,
        *(ax for g in axes_groups for ax in g),
        *axes_after,
    )

    # track where each axis will be in the new array
    num_groups = len(axes_groups)
    new_axes = {ax: ax for ax in axes_before}
    for g, gaxes in enumerate(axes_groups):
        for ax in gaxes:
            new_axes[ax] = position + g
    for i, ax in enumerate(axes_after):
        new_axes[ax] = position + num_groups + i
    new_ndim = len(axes_before) + num_groups + len(axes_after)

    return (
        num_groups,
        group_singlets,
        new_ndim,
        perm,
        position,
        axes_before,
        axes_after,
        ax2group,
        group_duals,
        new_axes,
    )


def calc_fuse_block_info(self, axes_groups):
    """Calculate the fusing information for a specific set of sectors/blocks."""
    # basic info that doesn't depend on sectors themselves
    (
        num_groups,
        group_singlets,
        new_ndim,
        perm,
        position,
        axes_before,
        axes_after,
        ax2group,
        group_duals,
        new_axes,
    ) = calc_fuse_group_info(axes_groups, self.duals)

    # then we process the blocks one by one into new fused sectors
    old_indices = self.indices
    blockmap = {}
    # for each group, map each subsector to a new charge and size
    subinfos = [{} for _ in range(num_groups)]
    combine = self.symmetry.combine
    sign = self.symmetry.sign

    # cache the results of each ax and charge lookup for speed
    lookup = {}

    # keep track of a shape in order to fuse the actual array
    new_shape = [None] * new_ndim
    # the key of the new fused block to add this block to
    new_sector = [None] * new_ndim
    # only the parts of the sector that will be fused
    subsectors = [[] for _ in range(num_groups)]
    # the same but signed for combining into new charges
    grouped_charges = [[] for _ in range(num_groups)]

    for sector in self.blocks:
        # reset accumulated values
        for g in range(num_groups):
            new_shape[position + g] = 1
            subsectors[g].clear()
            grouped_charges[g].clear()

        # n.b. we have to use `perm` here, not `enumerate(sector)`, so
        # that subsectors are built in matching order for tensordot e.g.
        for ax in perm:
            c = sector[ax]
            try:
                d, g, g_is_singlet, new_ax, signed_c = lookup[ax, c]
            except KeyError:
                # the size of charge `c` along axis `ax`
                ix = old_indices[ax]
                d = ix.size_of(c)

                # which group is this axis in, if any, and where is it going
                g = ax2group[ax]
                g_is_singlet = g in group_singlets
                new_ax = new_axes[ax]
                if g is None or g_is_singlet:
                    # not fusing
                    signed_c = None
                else:
                    # need to match current dualness to group dualness
                    signed_c = sign(c, group_duals[g] != ix.dual)

                lookup[ax, c] = d, g, g_is_singlet, new_ax, signed_c

            if signed_c is None:
                # not fusing, new value is just copied
                new_sector[new_ax] = c
                new_shape[new_ax] = d
                if g is not None:
                    subsectors[g].append(c)
            else:
                # fusing: need to accumulate
                new_shape[new_ax] *= d
                subsectors[g].append(c)
                grouped_charges[g].append(signed_c)

        # make hashable version
        _subsectors = tuple(map(tuple, subsectors))
        # process grouped charges
        for g in range(num_groups):
            if g not in group_singlets:
                # sum grouped charges
                new_charge = combine(*grouped_charges[g])
                new_sector[position + g] = new_charge
                # keep track of the new blocksize of each fused
                # index, for unfusing and also missing blocks
                new_size = new_shape[position + g]
                subsector = _subsectors[g]
                subinfos[g][subsector] = (new_charge, new_size)

        # to fuse (via transpose+reshape) the actual array, and concat later
        # first group the subblock into the correct new fused block
        blockmap[sector] = (tuple(new_shape), tuple(new_sector), _subsectors)

    # sort and accumulate subsectors into their new charges for each group
    chargemaps = []
    extents = []
    for g in range(num_groups):
        if g not in group_singlets:
            chargemap = {}
            extent = {}
            for subsector, (new_c, new_d) in sorted(subinfos[g].items()):
                if new_c not in chargemap:
                    chargemap[new_c] = new_d
                    extent[new_c] = {subsector: new_d}
                else:
                    chargemap[new_c] += new_d
                    extent[new_c][subsector] = new_d
            chargemaps.append(chargemap)
            extents.append(extent)
        else:
            # singlet group, no fusing
            chargemaps.append(None)
            extents.append(None)

    new_indices = (
        *(old_indices[ax] for ax in axes_before),
        # the new fused indices
        *(
            # don't need subinfo for size 1 groups
            old_indices[axes_groups[g][0]]
            if g in group_singlets
            else BlockIndex(
                chargemap=chargemaps[g],
                dual=group_duals[g],
                # for unfusing
                subinfo=SubIndexInfo(
                    indices=tuple(old_indices[i] for i in axes_groups[g]),
                    extents=extents[g],
                ),
            )
            for g in range(num_groups)
        ),
        *(old_indices[ax] for ax in axes_after),
    )

    return (
        num_groups,
        group_singlets,
        perm,
        position,
        axes_before,
        axes_after,
        new_axes,
        new_indices,
        blockmap,
    )


_fuseinfos = OrderedDict()

try:
    _fuseinfo_cache_maxsize = int(os.environ["SYMMRAY_FUSE_CACHE_MAXSIZE"])
    print(f"Using SYMMRAY_FUSE_CACHE_MAXSIZE={_fuseinfo_cache_maxsize}.")
except KeyError:
    _fuseinfo_cache_maxsize = 8192
except (TypeError, ValueError):
    print("SYMMRAY_FUSE_CACHE_MAXSIZE must be an integer, using default.")
    _fuseinfo_cache_maxsize = 8192

try:
    _fuseinfo_cache_maxsectors = int(
        os.environ["SYMMRAY_FUSE_CACHE_MAXSECTORS"]
    )
    print(f"Using SYMMRAY_FUSE_CACHE_MAXSECTORS={_fuseinfo_cache_maxsectors}.")
except KeyError:
    _fuseinfo_cache_maxsectors = 512
except (TypeError, ValueError):
    print("SYMMRAY_FUSE_CACHE_MAXSECTORS must be an integer, using default.")
    _fuseinfo_cache_maxsectors = 512

_fi_missed = 0
_fi_hit = 0
_fi_missed_too_long = 0


def print_fuseinfo_cache_stats():
    print(
        f"Cache size: {len(_fuseinfos)}\n"
        f"missed: {_fi_missed}, hit: {_fi_hit}\n"
        f"missed too long: {_fi_missed_too_long}\n"
        f"ratio: {_fi_missed / (_fi_missed + _fi_hit):.2f}\n"
    )


def cached_fuse_block_info(self, axes_groups):
    """Calculating fusing block information is expensive, so cache the results.
    This is a LRU cache that also skips caching if there are too many sectors.
    """

    if _fuseinfo_cache_maxsize == 0:
        # cache disabled
        return calc_fuse_block_info(self, axes_groups)

    if len(self.blocks) > _fuseinfo_cache_maxsectors:
        # too many sectors to cache
        global _fi_missed_too_long
        _fi_missed_too_long += 1
        return calc_fuse_block_info(self, axes_groups)

    key = hasher(
        (
            tuple(ix.hashkey() for ix in self.indices),
            tuple(self.blocks),
            self.symmetry,
            axes_groups,
        )
    )

    try:
        res = _fuseinfos[key]
        # mark as most recently used
        _fuseinfos.move_to_end(key)
        global _fi_hit
        _fi_hit += 1
    except KeyError:
        # compute new info
        res = _fuseinfos[key] = calc_fuse_block_info(self, axes_groups)
        # possibly trim cache
        if len(_fuseinfos) > _fuseinfo_cache_maxsize:
            # cache is full, remove the oldest entry
            _fuseinfos.popitem(last=False)
        global _fi_missed
        _fi_missed += 1

    return res


def _fuse_blocks_via_insert(
    blocks,
    num_groups,
    group_singlets,
    perm,
    position,
    new_indices,
    blockmap,
    _transpose,
    _reshape,
    _zeros,
    zeros_kwargs,
):
    """Perform the actual block fusing by inserting blocks into a new array."""
    new_blocks = {}

    # for each group, map each subsector to a range in the new charge
    slice_lookup = [
        {
            k: dict(zip(v, accum_for_split(v.values())))
            for k, v in new_indices[position + g].subinfo.extents.items()
        }
        if g not in group_singlets
        else None
        for g in range(num_groups)
    ]
    selector = [slice(None)] * len(new_indices)

    for sector, array in blocks.items():
        new_shape, new_sector, subsectors = blockmap[sector]
        # fuse (via transpose+reshape) actual array, to insert now
        new_array = _transpose(array, perm)
        new_array = _reshape(new_array, new_shape)

        # get subslice this block fits into within fused block
        for g, subsector in enumerate(subsectors):
            if g not in group_singlets:
                ax = position + g
                new_charge = new_sector[ax]
                selector[ax] = slice_lookup[g][new_charge][subsector]

        # get the target new fused block
        try:
            new_fused_array = new_blocks[new_sector]
        except KeyError:
            # create if it doesn't exist yet
            fused_shape = tuple(
                ix.size_of(c) for ix, c in zip(new_indices, new_sector)
            )
            new_fused_array = new_blocks[new_sector] = _zeros(
                fused_shape, **zeros_kwargs
            )

        # insert the block into the fused block
        new_fused_array[tuple(selector)] = new_array

    return new_blocks


def _fuse_blocks_via_concat(
    old_indices,
    blocks,
    num_groups,
    group_singlets,
    perm,
    position,
    axes_before,
    axes_after,
    new_axes,
    new_indices,
    blockmap,
    backend,
    _transpose,
    _reshape,
    _zeros,
    zeros_kwargs,
):
    """Perform the actual block fusing, by recusively concatenating blocks
    (more compatible with e.g. autodiff since requires no inplace updates).
    """
    _concatenate = ar.get_lib_fn(backend, "concatenate")

    new_blocks = {}

    # first we group subsectors into their new fused blocks
    for sector, array in blocks.items():
        new_shape, new_sector, subsectors = blockmap[sector]
        # fuse (via transpose+reshape) actual array, to concat later
        new_array = _transpose(array, perm)
        new_array = _reshape(new_array, new_shape)
        # group the subblock into the correct new fused block
        new_blocks.setdefault(new_sector, {})[subsectors] = new_array

    # then we actually have to combine the groups of subsectors

    def _recurse_concat(new_sector, g=0, subkey=()):
        if g in group_singlets:
            # singlet group, no need to concatenate
            new_subkey = subkey + ((new_sector[position + g],),)
            if g == num_groups - 1:
                return new_blocks[new_sector][new_subkey]
            else:
                return _recurse_concat(new_sector, g + 1, new_subkey)

        # else fused group of multiple axes
        new_charge = new_sector[position + g]
        extent = new_indices[position + g].subinfo.extents[new_charge]
        # given the current partial sector, get next possible charges
        next_subkeys = [(*subkey, subsector) for subsector in extent]

        if g == num_groups - 1:
            # final group (/level of recursion), get actual arrays
            arrays = []
            for new_subkey in next_subkeys:
                try:
                    array = new_blocks[new_sector][new_subkey]
                except KeyError:
                    # subsector is missing - need to create zeros
                    shape_before = (
                        old_indices[ax].size_of(new_sector[new_axes[ax]])
                        for ax in axes_before
                    )
                    shape_new = (
                        new_indices[position + gg].subinfo.extents[
                            new_sector[position + gg]
                        ][ss]
                        for gg, ss in enumerate(new_subkey)
                    )
                    shape_after = (
                        old_indices[ax].size_of(new_sector[new_axes[ax]])
                        for ax in axes_after
                    )
                    new_shape = (
                        *shape_before,
                        *shape_new,
                        *shape_after,
                    )
                    array = _zeros(new_shape, **zeros_kwargs)
                arrays.append(array)
        else:
            # recurse to next group
            arrays = (
                _recurse_concat(new_sector, g + 1, new_subkey)
                for new_subkey in next_subkeys
            )

        return _concatenate(tuple(arrays), axis=position + g)

    return {
        new_sector: _recurse_concat(new_sector) for new_sector in new_blocks
    }


class AbelianCommon:
    """Common base class for arrays with Abelian symmetry."""

    @property
    def symmetry(self) -> Symmetry:
        """The symmetry object of the array."""
        return self._symmetry

    @classmethod
    def get_class_symmetry(cls, symmetry=None) -> Symmetry:
        if symmetry is None:
            if cls.static_symmetry is None:
                # symmetry must be given if not static
                raise ValueError("Symmetry must be given.")
            symmetry = cls.static_symmetry
        elif cls.static_symmetry and symmetry != cls.static_symmetry:
            raise ValueError("Cannot override static symmetry of class.")

        return get_symmetry(symmetry)

    @property
    def signature(self):
        return "".join("-" if f else "+" for f in self.duals)

    @property
    def T(self):
        """The transpose of the block array."""
        return self.transpose()

    def dagger(self, inplace=False):
        """Return the adjoint of this block array."""
        return self.conj(inplace=inplace).transpose(inplace=True)

    @property
    def H(self):
        return self.dagger()

    def fuse(
        self,
        *axes_groups,
        expand_empty=True,
        inplace=False,
        **kwargs,
    ):
        """Fuse the given group or groups of axes. The new fused axes will be
        inserted at the minimum index of any fused axis (even if it is not in
        the first group). For example, ``x.fuse([5, 3], [7, 2, 6])`` will
        produce an array with axes like::

            groups inserted at axis 2, removed beyond that.
                   ......<--
            (0, 1, g0, g1, 4, 8, ...)
                   |   |
                   |   g1=(7, 2, 6)
                   g0=(5, 3)

        The fused axes will carry subindex information, which can be used to
        automatically unfuse them back into their original components.
        Depending on `expand_empty`, any empty groups can be expanded to new
        singlet dimensions, or simply ignored.

        Parameters
        ----------
        axes_groups : Sequence[Sequence[int]]
            The axes to fuse. Each group of axes will be fused into a single
            axis.
        expand_empty : bool, optional
            Whether to expand empty groups into new axes.
        mode : "auto", "insert", "concat", optional
            The method to use for fusing. `"insert"` creates the new fused
            blocks and insert the subblocks inplace. `"concat"` recursively
            concatenates the subblocks, which can be slightly slower but is
            more compatible with e.g. autodiff. `"auto"` will use `"insert"` if
            the backend is numpy, otherwise `"concat"`.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.
        kwargs : dict, optional
            Additional keyword arguments to pass to the core fusing method.

        Returns
        -------
        AbelianCommon
        """
        # handle empty groups and ensure hashable
        _axes_groups = []
        _axes_expand = []
        for ax, group in enumerate(axes_groups):
            if group:
                _axes_groups.append(tuple(group))
            else:
                _axes_expand.append(ax)

        if _axes_groups:
            xf = self._fuse_core(*_axes_groups, inplace=inplace, **kwargs)
        else:
            xf = self if inplace else self.copy()

        if expand_empty and _axes_expand:
            g0 = min(g for groups in _axes_groups for g in groups)
            for ax in _axes_expand:
                xf.expand_dims(g0 + ax, inplace=True)

        return xf

    def unfuse_all(self, inplace=False):
        """Unfuse all indices that carry subindex information, likely from a
        fusing operation.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianCommon
        """
        new = self if inplace else self.copy()
        for ax in reversed(range(self.ndim)):
            if new.is_fused(ax):
                new.unfuse(ax, inplace=True)
        return new

    def reshape(self, newshape, inplace=False):
        """Reshape this abelian array to ``newshape``, assuming it can be done
        by any mix of fusing, unfusing, and expanding new axes.

        Restrictions and complications vs normal array reshaping arise from the
        fact that

            A) only previously fused axes can be unfused, and their total size
               may not be the product of the individual sizes due to sparsity.
            B) the indices carry information beyond size and how they are
               grouped potentially matters, relevant for singleton dimensions.

        Accordingly the approach here is as follows:

            1. Unfuse any axes that match the new shape.

            2. If there are singleton dimensions that don't appear in the new
               shape, (i.e. are being 'squeezed') these are grouped with the
               axis to the their left to then be fused. If they are already
               left-most, they are grouped with the right.

            3. Fuse any groups of axes required to match the new shape.
               Adjacent groups are fused simultaneously for efficiency.

            4. Expand new axes required to match singlet dimensions in the new
               shape. By default these will have zero charge and dual-ness
               iherited from whichever axis is to their left, or right if they
               are the left-most axis already.

        To avoid the effective grouping of 'squeezed' axes you can explicitly
        squeeze them before reshaping. Similarly use ``expand_dims`` to add
        new axes with specific charges and dual-ness.

        Parameters
        ----------
        newshape : tuple[int]
            The new shape to reshape to.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray

        See Also
        --------
        fuse, unfuse, squeeze, expand_dims
        """
        x = self if inplace else self.copy()

        if not isinstance(newshape, tuple):
            newshape = tuple(newshape)
        newshape = find_full_reshape(newshape, self.size)

        subshapes = tuple(ix.subshape for ix in x.indices)

        # cached parsing of reshape arguments
        axs_unfuse, axs_fuse_groupings, axs_expand = calc_reshape_args(
            x.shape, newshape, subshapes
        )

        for ax in axs_unfuse:
            x.unfuse(ax, inplace=True)
        for grouping in axs_fuse_groupings:
            x.fuse(*grouping, inplace=True)
        for ax in axs_expand:
            x.expand_dims(ax, inplace=True)

        return x

    def __str__(self):
        lines = [
            (
                f"{self.__class__.__name__}(ndim={self.ndim}, "
                f"charge={self.charge}, indices=["
            )
        ]
        for i in range(self.ndim):
            lines.extend(
                f"    {line}" for line in str(self.indices[i]).split("\n")
            )
        lines.append(
            (
                f"], num_blocks={self.num_blocks}, backend={self.backend}, "
                f"dtype={self.dtype})"
            )
        )
        return "\n".join(lines)

    def __repr__(self):
        if self.static_symmetry is not None:
            c = f"{self.__class__.__name__}("
        else:
            c = f"{self.__class__.__name__}{self.symmetry}("

        return "".join(
            [
                c,
                (
                    f"shape~{self.shape}:[{self.signature}]"
                    if self.indices
                    else f"{self.get_any_array()}"
                ),
                f", charge={self.charge}",
                f", num_blocks={self.num_blocks})",
            ]
        )


class AbelianArray(AbelianCommon, BlockCommon):
    """A block sparse array with symmetry constraints.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices of the array.
    charge : hashable, optionals
        The total charge of the array, if not given it will be inferred from
        either the first sector or set to the identity charge, if no sectors
        are given.
    blocks : dict[tuple[hashable], array_like]
        A mapping of each 'sector' (tuple of charges) to the data array.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = ("_indices", "_blocks", "_charge", "_symmetry")
    fermionic = False
    static_symmetry = None

    def __init__(
        self,
        indices,
        charge=None,
        blocks=(),
        symmetry=None,
    ):
        self._indices = tuple(indices)
        self._blocks = dict(blocks)

        self._symmetry = self.get_class_symmetry(symmetry)

        if charge is None:
            if self._blocks:
                # infer the charge total from any sector
                sector = next(iter(self._blocks))
                signed_sector = (
                    self.symmetry.sign(c, ix.dual)
                    for c, ix in zip(sector, self._indices)
                )
                self._charge = self.symmetry.combine(*signed_sector)
            else:
                # default to the identity charge
                self._charge = self.symmetry.combine()
        else:
            self._charge = charge

        if DEBUG:
            self.check()

    def copy(self):
        """Copy this block array."""
        new = self.__new__(self.__class__)
        new._indices = self._indices
        new._charge = self._charge
        new._blocks = self._blocks.copy()
        new._symmetry = self._symmetry
        return new

    def copy_with(self, indices=None, charge=None, blocks=None):
        """A copy of this block array with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self.__new__(self.__class__)
        new._indices = self._indices if indices is None else indices
        new._charge = self._charge if charge is None else charge
        new._blocks = self._blocks.copy() if blocks is None else blocks
        new._symmetry = self._symmetry

        if DEBUG:
            new.check()

        return new

    def modify(self, indices=None, charge=None, blocks=None):
        """Modify this block array in place with some attributes replaced. Note
        that checks are not performed on the new properties, this is intended
        for internal use.
        """
        if indices is not None:
            self._indices = indices
        if charge is not None:
            self._charge = charge
        if blocks is not None:
            self._blocks = blocks

        if DEBUG:
            self.check()

        return self

    def _modify_or_copy(
        self, indices=None, charge=None, blocks=None, inplace=False
    ):
        if inplace:
            return self.modify(indices=indices, charge=charge, blocks=blocks)
        else:
            return self.copy_with(
                indices=indices, charge=charge, blocks=blocks
            )

    @property
    def indices(self):
        """The indices of the array."""
        return self._indices

    @property
    def sizes(self):
        """The sizes of each index."""
        return tuple(ix.sizes for ix in self._indices)

    @property
    def charges(self):
        """The possible charges of each index."""
        return tuple(ix.charges for ix in self._indices)

    @property
    def duals(self):
        """The dual-ness of each index."""
        return tuple(ix.dual for ix in self._indices)

    @property
    def charge(self):
        """The total charge of the array."""
        return self._charge

    @property
    def shape(self):
        """The shape of the array, i.e. product of total size of each index."""
        return tuple(ix.size_total for ix in self._indices)

    @property
    def size(self):
        """The number of possible elements in this array, if it was dense."""
        return math.prod(self.shape)

    @property
    def ndim(self):
        """The number of dimensions/indices."""
        return len(self._indices)

    def is_fused(self, ax):
        """Does axis `ax` carry subindex information, i.e., is it a fused
        index?
        """
        return self.indices[ax].subinfo is not None

    def sync_charges(self, inplace=False):
        """Given the blocks currently present, adjust the index chargemaps to
        match only those charges present in at least one sector.
        """
        charges_drop = [set(ix.charges) for ix in self.indices]
        for sector in self.blocks:
            for i, c in enumerate(sector):
                charges_drop[i].discard(c)

        new_indices = tuple(
            ix.drop_charges(cs_drop) if cs_drop else ix
            for ix, cs_drop in zip(self.indices, charges_drop)
        )

        return self._modify_or_copy(indices=new_indices, inplace=inplace)

    def is_valid_sector(self, sector):
        """Check if a sector is valid for the block array, i.e., whether the
        total symmetry charge is satisfied.
        """
        signed_sector = (
            self.symmetry.sign(c, ix.dual)
            for c, ix in zip(sector, self._indices)
        )
        block_charge = self.symmetry.combine(*signed_sector)
        return block_charge == self.charge

    def gen_valid_sectors(self):
        """Generate all valid sectors for the block array."""
        if self.ndim == 0:
            if self.charge == self.symmetry.combine():
                yield ()
            return

        *first_charges, last_charges = self.charges
        *first_duals, last_dual = self.duals

        for partial_sector in itertools.product(*first_charges):
            # signed first charges + signed last charge = overall charge
            # thus
            # last charge = signed(overall charge - signed first charges)
            signed_partial_sector = self.symmetry.combine(
                *(
                    self.symmetry.sign(c, not dual)
                    for c, dual in zip(partial_sector, first_duals)
                )
            )
            required_charge = self.symmetry.sign(
                self.symmetry.combine(
                    self.charge,
                    signed_partial_sector,
                ),
                last_dual,
            )
            if required_charge in last_charges:
                # but only if it is a valid charge for that index
                yield partial_sector + (required_charge,)

    def get_sparsity(self):
        """Return the sparsity of the array, i.e. the number of blocks
        divided by the number of possible blocks.
        """
        num_possible_blocks = len(tuple(self.gen_valid_sectors()))
        return self.num_blocks / num_possible_blocks

    def get_block_shape(self, sector):
        """Get the shape of the block corresponding to a given sector."""
        return tuple(ix.size_of(c) for ix, c in zip(self._indices, sector))

    def fill_missing_blocks(self):
        """Insert any missing blocks for valid sectors with zeros, resulting
        in a sparsity of 1.
        """
        _ex_array = self.get_any_array()
        for sector in self.gen_valid_sectors():
            if sector not in self.blocks:
                shape = self.get_block_shape(sector)
                array = ar.do("zeros", shape, like=_ex_array)
                self.blocks[sector] = array

    def drop_missing_blocks(self):
        """Drop any present blocks that are all zero, resulting in a sparsity
        possibly less that 1.
        """
        _all = ar.get_lib_fn(self.backend, "all")
        for sector in list(self.blocks.keys()):
            if _all(self.blocks[sector] == 0.0):
                del self.blocks[sector]

    def check(self):
        """Check that all the block sizes and charges are consistent."""
        for idx in self.indices:
            idx.check()

        for sector, array in self.blocks.items():
            if not self.is_valid_sector(sector):
                raise ValueError(
                    f"Invalid sector {sector} for array with {self.duals}"
                    f" and charge total {self.charge}."
                )

            if not all(
                di == dj
                for di, dj in zip(
                    ar.shape(array), self.get_block_shape(sector)
                )
            ):
                raise ValueError(
                    f"Block shape {ar.shape(array)} does not match "
                    f"expected shape {self.get_block_shape(sector)} "
                    f"for sector {sector}."
                )

            if not ar.do(
                "all",
                ar.do("isfinite", array, like=self.backend),
                like=self.backend,
            ):
                raise ValueError(f"Block {sector} contains non-finite values.")

    def check_chargemaps_aligned(self):
        """Check that the chargemaps of the indices are consistent with the
        block sectors.
        """
        actual_charges = [set() for _ in range(self.ndim)]

        for sector in self.blocks:
            for i, c in enumerate(sector):
                actual_charges[i].add(c)

        if self.blocks:
            # only check if we have filled anything
            for actual, index in zip(actual_charges, self.indices):
                expected = set(index.chargemap)
                if actual != expected:
                    raise ValueError(
                        f"Charges for index {index} are inconsistent: "
                        f"{actual} != {expected}."
                    )

    def check_with(self, other, *args):
        """Check that this block array is compatible with another, that is,
        that the indices match and the blocks are compatible.

        Parameters
        ----------
        other : AbelianArray or BlockVector
            The other array or vector to compare to.
        *args
            The axes to compare, if ``other`` is a vector, the axis to compare
            with. If ``other`` is an array, then `axes_a` and `axes_b` should
            be given as if for a tensordot.
        """
        if isinstance(other, BlockVector):
            (ax,) = args
            for sector, array in self.blocks.items():
                charge = sector[ax]
                v_block = other.blocks[charge]
                assert ar.shape(array)[ax] == ar.size(v_block)

        else:
            assert self.symmetry == other.symmetry

            axes_a, axes_b = args
            for axa, axb in zip(axes_a, axes_b):
                assert self.indices[axa].matches(other.indices[axb])

    def allclose(self, other, **allclose_opts):
        """Test whether this ``AbelianArray`` is close to another, that is,
        has all the same sectors, and the corresponding arrays are close.

        Parameters
        ----------
        other : AbelianArray
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Returns
        -------
        bool
        """
        # charge and signature must match
        if self.charge != other.charge:
            return False

        if self.duals != other.duals:
            return False

        # defined on BlockBase:
        return BlockCommon.allclose(self, other)

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        indices,
        charge=None,
        symmetry=None,
        **kwargs,
    ):
        """Generate a block array from a filling function. Every valid sector
        will be filled with the result of the filling function.

        Parameters
        ----------
        fill_fn : callable
            The filling function, with signature ``fill_fn(shape)``.
        indices : tuple[BlockIndex]
            The indices of the array.
        charge : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.
        symmetry : str or Symmetry, optional
            The symmetry of the array, if not using a specific symmetry class.

        Returns
        -------
        AbelianArray
        """
        symmetry = cls.get_class_symmetry(symmetry)

        if charge is None:
            charge = symmetry.combine()
        else:
            charge = charge

        new = cls(indices=indices, charge=charge, symmetry=symmetry, **kwargs)

        for sector in new.gen_valid_sectors():
            new.blocks[sector] = fill_fn(new.get_block_shape(sector))

        return new

    @classmethod
    def random(
        cls,
        indices,
        charge=None,
        seed=None,
        dist="normal",
        dtype="float64",
        scale=1.0,
        loc=0.0,
        symmetry=None,
        **kwargs,
    ):
        """Create a block array with random values.

        Parameters
        ----------
        indices : tuple[BlockIndex]
            The indices of the array.
        charge : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.
        seed : None, int or numpy.random.Generator
            The random seed or generator to use.
        dist : str
            The distribution to use. Can be one of ``"normal"``, ``"uniform"``,
            etc., see :func:`numpy.random.default_rng` for details.
        symmetry : str or Symmetry, optional
            The symmetry of the array, if not using a specific symmetry class.

        Returns
        -------
        AbelianArray
        """
        from .utils import get_random_fill_fn

        fill_fn = get_random_fill_fn(
            dist=dist,
            dtype=dtype,
            loc=loc,
            scale=scale,
            seed=seed,
        )

        return cls.from_fill_fn(
            fill_fn, indices, charge, symmetry=symmetry, **kwargs
        )

    @classmethod
    def from_blocks(cls, blocks, duals, charge=None, symmetry=None, **kwargs):
        """Create a block array from a dictionary of blocks and sequence of
        duals.

        Parameters
        ----------
        blocks : dict[tuple[hashable], array_like]
            A mapping of each 'sector' (tuple of charges) to the data array.
        duals : tuple[bool]
            The dual-ness of each index.
        charge : hashable
            The total charge of the array. If not given, it will be
            taken computed from the first sector, or set to the
            identity / zero element if no sectors are given.
        symmetry : str or Symmetry, optional
            The symmetry of the array, if not using a specific symmetry class.
        kwargs
            Additional keyword arguments to pass to the constructor.

        Returns
        -------
        AbelianArray
        """
        symmetry = cls.get_class_symmetry(symmetry)

        ndim = len(next(iter(blocks.keys())))
        charge_size_maps = [{} for _ in range(ndim)]

        for sector, array in blocks.items():
            for i, (c, d) in enumerate(zip(sector, ar.shape(array))):
                d = int(d)
                d_existing = charge_size_maps[i].get(c, None)
                if d_existing is None:
                    charge_size_maps[i][c] = d
                elif d != d_existing:
                    raise ValueError(
                        f"Inconsistent block sizes for index {i}"
                        f" with charge {c}: {d_existing} != {d}."
                    )

            if charge is None:
                # infer the charge total from the first sector
                signed_sector = tuple(
                    symmetry.sign(c, dual) for c, dual in zip(sector, duals)
                )
                charge = symmetry.combine(*signed_sector)

        if charge is None:
            # no blocks given, default to identity charge
            charge = symmetry.combine()

        duals = tuple(duals)
        if len(duals) != ndim:
            raise ValueError(f"Expected {ndim} duals, got {len(duals)}.")

        indices = tuple(
            BlockIndex(x, dual) for x, dual in zip(charge_size_maps, duals)
        )

        return cls(
            blocks=blocks,
            indices=indices,
            charge=charge,
            symmetry=symmetry,
            **kwargs,
        )

    @classmethod
    def from_dense(
        cls,
        array,
        index_maps,
        duals,
        charge=None,
        symmetry=None,
        invalid_sectors="warn",
        **kwargs,
    ):
        """Create a block array from a dense array by supplying a mapping for
        each axis that labels each linear index with a particular charge.

        Parameters
        ----------
        array : array_like
            The dense array.
        index_maps : tuple[dict[int, hashable]]
            A mapping for each axis that labels each linear index with a
            particular charge. There should be ``ndim`` such mappings, each of
            size ``shape[i]``.
        duals : tuple[bool]
            The dualness of each index.
        charge : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.
        symmetry : str or Symmetry, optional
            The symmetry of the array, if not using a specific symmetry class.
        invalid_sectors : {"warn", "raise", "ignore"}, optional
            How to handle invalid sectors that have non-zero entries.

        Returns
        -------
        AbelianArray
        """
        # XXX: warn if invalid blocks are non-zero?
        symmetry = cls.get_class_symmetry()

        if charge is None:
            charge = symmetry.combine()

        # first we work out which indices of which axes belong to which charges
        charge_groups = []
        for d, index_map in zip(ar.shape(array), index_maps):
            which_charge = {}
            for i in range(d):
                which_charge.setdefault(index_map[i], []).append(i)
            charge_groups.append(which_charge)

        # then we recusively visit all the potential blocks, by slicing using
        # the above generated charge groupings
        blocks = {}
        ndim = ar.ndim(array)
        all_sliced = [slice(None)] * ndim

        def _recurse(ary, j=0, sector=()):
            if j < ndim:
                for c, indices in charge_groups[j].items():
                    # for each charge, select all the indices along axis j
                    # that belong to it, then recurse further
                    selector = all_sliced.copy()
                    selector[j] = indices
                    subarray = ary[tuple(selector)]
                    _recurse(subarray, j + 1, sector + (c,))
            else:
                # we have reached a fully specified block
                signed_sector = tuple(
                    symmetry.sign(c, dual) for c, dual in zip(sector, duals)
                )
                if symmetry.combine(*signed_sector) == charge:
                    # ... but only add valid ones:
                    blocks[sector] = ary

                elif invalid_sectors != "ignore":
                    # check for non zero entries
                    has_nnz = ar.do("any", ar.do("abs", ary) > 1e-12)

                    if has_nnz:
                        base_msg = (
                            f"Block with sector {sector} has non-zero elements"
                            f" but does not match the total charge {charge}."
                        )

                        if invalid_sectors == "warn":
                            warnings.warn(
                                f"{base_msg} Ignoring them. Set "
                                "`invalid_sectors` to 'ignore' to suppress "
                                "this warning, or 'raise' to actively error.",
                            )

                        elif invalid_sectors == "raise":
                            raise ValueError(base_msg)

        # generate the blocks
        _recurse(array)

        # generate the indices -> the charge_map is simply the group size
        indices = [
            BlockIndex({c: len(g) for c, g in charge_group.items()}, dual=dual)
            for charge_group, dual in zip(charge_groups, duals)
        ]

        # create the block array!
        return cls(
            blocks=blocks,
            indices=indices,
            charge=charge,
            symmetry=symmetry,
            **kwargs,
        )

    def to_dense(self):
        """Convert this block array to a dense array."""
        backend = self.backend
        _ex_array = self.get_any_array()
        _concat = ar.get_lib_fn(backend, "concatenate")

        def filler(shape):
            return ar.do("zeros", shape, like=_ex_array)

        def _recurse_all_charges(partial_sector=()):
            i = len(partial_sector)
            if i == self.ndim:
                # full sector, return the block, making zeros if necessary
                array = self._blocks.get(partial_sector, None)
                if array is None:
                    array = filler(self.get_block_shape(partial_sector))
                return array
            else:
                # partial sector -> recurse further
                arrays = tuple(
                    _recurse_all_charges(partial_sector + (c,))
                    for c in sorted(self.indices[i].charges)
                )
                # then concatenate along the current axis
                return _concat(arrays, axis=i)

        return _recurse_all_charges()

    def conj(self, inplace=False):
        """Return the complex conjugate of this block array, including the
        indices."""
        new = self if inplace else self.copy()
        _conj = ar.get_lib_fn(new.backend, "conj")
        new.apply_to_arrays(_conj)

        new.modify(
            indices=tuple(ix.conj() for ix in self._indices),
            charge=self.symmetry.sign(self._charge),
        )

        return new

    def transpose(self, axes=None, inplace=False):
        """Transpose the block array.

        Parameters
        ----------
        axes : tuple[int, ...] | None, optional
            A permutation of the axes to transpose the array by. If None,
            the axes will be reversed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        new = self if inplace else self.copy()

        _transpose = ar.get_lib_fn(new.backend, "transpose")

        if axes is None:
            # reverse the axes
            axes = tuple(range(new.ndim - 1, -1, -1))

        return new.modify(
            indices=permuted(new._indices, axes),
            blocks={
                permuted(sector, axes): _transpose(array, axes)
                for sector, array in new.blocks.items()
            },
        )

    def select_charge(self, axis, charge, inplace=False):
        """Drop all but the specified charge along the specified axis. Note the
        axis is not removed, it is simply restricted to a single charge.

        Parameters
        ----------
        axis : int
            The axis along which to select the charge.
        charge : int
            The charge to select along the specified axis.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        if axis < 0:
            axis += self.ndim

        # update indices
        new_indices = (
            *self.indices[:axis],
            self.indices[axis].select_charge(charge),
            *self.indices[axis + 1 :],
        )

        # and filter blocks
        new_blocks = {
            k: v for k, v in self.blocks.items() if k[axis] == charge
        }

        return self._modify_or_copy(
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def squeeze(self, axis=None, inplace=False):
        """Squeeze the block array, removing axes of size 1.

        Parameters
        ----------
        axis : int or sequence of int, optional
            The axes to squeeze. If not given, all axes of size 1 will be
            removed.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        AbelianArray
        """
        x = self if inplace else self.copy()

        if isinstance(axis, int):
            axis = (axis,)

        keep = []
        selector = []
        new_indices = []
        new_charge = x.charge

        for ax, ix in enumerate(x.indices):
            if axis is None:
                remove = ix.size_total == 1
            else:
                remove = ax in axis
                if remove and ix.size_total > 1:
                    raise ValueError("Cannot squeeze d > 1 index")

            if remove:
                selector.append(0)
                (charge,) = ix.chargemap
                new_charge = x.symmetry.combine(
                    new_charge,
                    x.symmetry.sign(charge, not ix.dual),
                )
            else:
                new_indices.append(ix)
                keep.append(ax)
                selector.append(slice(None))

        selector = tuple(selector)

        x._map_blocks(
            fn_sector=lambda sector: tuple(sector[ax] for ax in keep),
            fn_block=lambda block: block[selector],
        )
        return x.modify(
            indices=tuple(new_indices),
            charge=new_charge,
        )

    def isel(self, axis, idx, inplace=False):
        """Select a single index along the specified axis."""
        if axis < 0:
            axis += self.ndim
        new = self.select_charge(axis, idx, inplace=inplace)
        return new.squeeze(axis, inplace=True)

    def __getitem__(self, item):
        axis = None
        idx = None

        if not isinstance(item, tuple):
            raise TypeError(
                f"Expected a tuple for indexing, got {type(item)}: {item}"
            )

        for i, s in enumerate(item):
            if isinstance(s, slice):
                if not s.start is s.stop is s.step is None:
                    raise NotImplementedError("Can only slice whole axes.")
            else:
                if axis is not None:
                    raise ValueError(
                        "Can only index one axis at a time, "
                        f"got {item} with multiple indices."
                    )
                axis = i
                idx = s

        return self.isel(axis, idx)

    def expand_dims(self, axis, c=None, dual=None, inplace=False):
        """Expand the shape of an abelian array.

        Parameters
        ----------
        axis : int
            The position along which to expand.
        c : hashable, optional
            The charge to insert at the new axis. If not given, a zero charge
            will be inserted.
        dual : bool, optional
            The dual-ness of the new index. If not given, it will be inherited
            from the axis before or after, if any. If there is no axis before
            or after, it will default to `False`.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        AbelianArray
        """
        x = self if inplace else self.copy()

        if axis < 0:
            axis += x.ndim + 1

        if dual is None:
            # we inherit the dual-ness from the axis before or after
            # to make fusing and unfusing singleton axes commutative
            if axis > 0:
                # inherit from left
                dual = x.indices[axis - 1].dual
            elif axis < x.ndim:
                # inherit from right
                dual = x.indices[axis].dual
            else:
                # no axes to inherit from
                dual = False

        charge = x.charge
        if c is None:
            # zero charge
            c = x.symmetry.combine()
            new_charge = charge
        else:
            # need to update overall charge
            new_charge = x.symmetry.combine(charge, x.symmetry.sign(c, dual))

        selector = (
            (slice(None),) * axis + (None,) + (slice(None),) * (x.ndim - axis)
        )

        x._map_blocks(
            fn_sector=lambda sector: (*sector[:axis], c, *sector[axis:]),
            fn_block=lambda block: block[selector],
        )

        new_indices = (
            *x.indices[:axis],
            BlockIndex({c: 1}, dual=dual),
            *x.indices[axis:],
        )

        return x.modify(indices=new_indices, charge=new_charge)

    def _fuse_core(
        self,
        *axes_groups,
        mode="auto",
        inplace=False,
    ):
        (
            num_groups,
            group_singlets,
            perm,
            position,
            axes_before,
            axes_after,
            new_axes,
            new_indices,
            blockmap,
        ) = cached_fuse_block_info(self, axes_groups)
        # NOTE: to turn off caching, we would use the following line instead:
        # ) = calc_fuse_block_info(self, axes_groups)

        _ex_array = self.get_any_array()
        backend = ar.infer_backend(_ex_array)
        _transpose = ar.get_lib_fn(backend, "transpose")
        _reshape = ar.get_lib_fn(backend, "reshape")

        # explicity handle zeros function and dtype and device kwargs
        _zeros = ar.get_lib_fn(backend, "zeros")
        zeros_kwargs = {}
        if hasattr(_ex_array, "dtype"):
            zeros_kwargs["dtype"] = _ex_array.dtype
        if hasattr(_ex_array, "device"):
            zeros_kwargs["device"] = _ex_array.device

        if mode == "auto":
            if backend == "numpy":
                mode = "insert"
            else:
                mode = "concat"

        if mode == "insert":
            new_blocks = _fuse_blocks_via_insert(
                self.blocks,
                num_groups,
                group_singlets,
                perm,
                position,
                new_indices,
                blockmap,
                _transpose,
                _reshape,
                _zeros,
                zeros_kwargs,
            )
        elif mode == "concat":
            new_blocks = _fuse_blocks_via_concat(
                self._indices,
                self._blocks,
                num_groups,
                group_singlets,
                perm,
                position,
                axes_before,
                axes_after,
                new_axes,
                new_indices,
                blockmap,
                backend,
                _transpose,
                _reshape,
                _zeros,
                zeros_kwargs,
            )
        else:
            raise ValueError(f"Unknown mode {mode}.")

        return self._modify_or_copy(
            indices=new_indices, blocks=new_blocks, inplace=inplace
        )

    def unfuse(self, axis, inplace=False):
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.

        Parameters
        ----------
        axis : int
            The axis to unfuse. It must have subindex information (`.subinfo`).
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        backend = self.backend
        # _split = ar.get_lib_fn(backend, "split")
        _reshape = ar.get_lib_fn(backend, "reshape")

        # get required information from the fused index
        subinfo = self.indices[axis].subinfo

        # info for how to split/slice the linear index into sub charges
        subindex_slices = {
            c: accum_for_split(d for d in charge_extent.values())
            for c, charge_extent in subinfo.extents.items()
        }
        selector = tuple(slice(None) for _ in range(axis))

        new_blocks = {}
        for sector, array in self.blocks.items():
            old_charge = sector[axis]
            old_shape = ar.shape(array)

            charge_extent = subinfo.extents[old_charge]

            new_arrays = []
            for slc in subindex_slices[old_charge]:
                new_arrays.append(array[(*selector, slc)])
            # new_arrays = _split(array, splits, axis=axis)

            for subsector, new_array in zip(charge_extent, new_arrays):
                # expand the old charge into the new subcharges
                new_key = replace_with_seq(sector, axis, subsector)

                # reshape the array to the correct shape
                subshape = tuple(
                    ix.size_of(c) for ix, c in zip(subinfo.indices, subsector)
                )
                new_shape = replace_with_seq(old_shape, axis, subshape)

                # reshape and store!
                new_blocks[new_key] = _reshape(new_array, new_shape)

        new_indices = replace_with_seq(self.indices, axis, subinfo.indices)
        new_blocks = new_blocks

        return self._modify_or_copy(
            indices=new_indices, blocks=new_blocks, inplace=inplace
        )

    def __matmul__(self, other, preserve_array=False):
        if self.ndim > 2 or other.ndim > 2:
            raise ValueError("Only 1D and 2D arrays supported.")

        left_axes, axes_a, axes_b, right_axes = {
            (1, 1): ((), (0,), (0,), ()),
            (1, 2): ((), (0,), (0,), (1,)),
            (2, 1): ((0,), (1,), (0,), ()),
            (2, 2): ((0,), (1,), (0,), (1,)),
        }[self.ndim, other.ndim]

        # block diagonal -> shortcut to tensordot
        c = _tensordot_blockwise(
            self,
            other,
            left_axes=left_axes,
            axes_a=axes_a,
            axes_b=axes_b,
            right_axes=right_axes,
        )

        if (c.ndim == 0) and (not preserve_array):
            try:
                return c.blocks[()]
            except KeyError:
                # no aligned blocks, return zero
                return 0.0

        return c

    def trace(self):
        """Compute the trace of the block array, assuming it is a square
        matrix.
        """
        if self.ndim != 2:
            raise ValueError("Trace requires a 2D array.")

        _trace = ar.get_lib_fn(self.backend, "trace")

        return sum(
            _trace(array)
            for sector, array in self.blocks.items()
            # only take diagonal blocks
            if sector[0] == sector[1]
        )

    def multiply_diagonal(self, v: BlockVector, axis, inplace=False):
        """Multiply this block array by a vector as if contracting a diagonal
        matrix along the given axis.

        Parameters
        ----------
        v : BlockVector
            The vector to contract with.
        axis : int
            The axis along which to contract.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        AbelianArray
        """
        x = self if inplace else self.copy()

        _reshape = ar.get_lib_fn(v.backend, "reshape")
        new_shape = tuple(-1 if i == axis else 1 for i in range(x.ndim))

        # sort by axis charge to group vector blocks
        sectors = sorted(x.sectors, key=lambda s: s[axis])
        v_charge = None

        for sector in sectors:
            charge = sector[axis]

            # only compute reshaped vector block when charge changes
            if charge != v_charge:
                v_block = v.blocks.get(charge, None)
                if v_block is not None:
                    v_block = _reshape(v_block, new_shape)
                v_charge = charge

            if v_block is not None:
                # use broadcasting to perform "ab...X...c,X-> ab...X...c"
                x.blocks[sector] = x.blocks[sector] * v_block
            else:
                # block isn't present -> like multiplying by zero
                del x.blocks[sector]

        if DEBUG:
            x.check()

        return x

    def align_axes(self, other, axes):
        """Align the axes of this block array with another, by dropping any
        sectors that are not aligned along the given axes, these can then be
        fused into a single axis that matches on both arrays.

        Parameters
        ----------
        other : AbelianArray
            The other array to align with.
        axes : tuple[tuple[int]]
            The pairs of axes to align, given as tuples of the corresponding
            axes in this and the other array, a la tensordot.
        """
        return drop_misaligned_sectors(self, other, *axes)

    def einsum(self, eq, preserve_array=False):
        """Einsum for abelian arrays, currently only single term.

        Parameters
        ----------
        eq : str
            The einsum equation, e.g. "abcb->ca". The output indices must be
            specified and only trace and permutations are allowed.
        preserve_array : bool, optional
            If tracing to a scalar, whether to return an AbelainArray object
            with no indices, or simply scalar itself (the default).

        Returns
        -------
        AbelianArray or scalar
        """
        _einsum = ar.get_lib_fn(self.backend, "einsum")

        # parse equation
        lhs, rhs = eq.split("->")
        ind_map = {}
        traced = {}
        for j, (q, ind) in enumerate(zip(lhs, self.indices)):
            if q in rhs:
                ind_map[q] = ind
            else:
                traced.setdefault(q, []).append(j)
        # how to permute kept indices to match output
        perm = tuple(map(lhs.index, rhs))

        if DEBUG:
            for q, js in traced.items():
                if len(js) != 2:
                    raise ValueError(
                        f"Can only trace two indices, got {len(js)}."
                    )
                j1, j2 = js
                assert self.indices[j1].matches(self.indices[j2])

        new_blocks = {}
        for sector, array in self.blocks.items():
            if all(sector[ja] == sector[jb] for ja, jb in traced.values()):
                # only trace diagonal blocks
                new_sector = tuple(sector[i] for i in perm)
                new_array = _einsum(eq, array)

                try:
                    new_blocks[new_sector] = new_blocks[new_sector] + new_array
                except KeyError:
                    new_blocks[new_sector] = new_array

        new_indices = tuple(ind_map[q] for q in rhs)

        if rhs or preserve_array:
            # wrap in new array
            return self.copy_with(
                indices=new_indices,
                blocks=new_blocks,
            )

        try:
            return new_blocks[()]
        except KeyError:
            # no aligned blocks, return zero
            return 0.0

    def to_flat(self):
        """ """
        from .flat_core import get_zn_array_flat_cls
        from .symmetries import ZN

        if not isinstance(self.symmetry, ZN):
            raise ValueError("Only ZN symmetry supported.")

        cls = get_zn_array_flat_cls(self.symmetry.N)
        return cls.from_blocksparse(self)


# --------------------------------------------------------------------------- #


def _tensordot_blockwise(a, b, left_axes, axes_a, axes_b, right_axes):
    """Perform a tensordot between two block arrays, performing the contraction
    of each pair of aligned blocks separately.
    """
    aligned_blocks = defaultdict(list)

    # iterate over all valid sectors of the new AbelianArray
    _tensordot = ar.get_lib_fn(a.backend, "tensordot")
    # _stack = ar.get_lib_fn(a.backend, "stack")

    # group blocks of `b` by which contracted charges they are aligned to
    for sector, array_b in b.blocks.items():
        sector_contracted = tuple(sector[i] for i in axes_b)
        sector_right = tuple(sector[i] for i in right_axes)
        aligned_blocks[sector_contracted].append((sector_right, array_b))

    # accumulate aligned blocks of `a` into a pair of lists
    new_blocks = {}
    for sector, array_a in a.blocks.items():
        sector_contracted = tuple(sector[i] for i in axes_a)
        sector_left = tuple(sector[i] for i in left_axes)
        for sector_right, array_b in aligned_blocks[sector_contracted]:
            new_sector = sector_left + sector_right
            try:
                arrays_suba, arrays_subb = new_blocks[new_sector]
            except KeyError:
                arrays_suba, arrays_subb = [], []
                new_blocks[new_sector] = arrays_suba, arrays_subb
            arrays_suba.append(array_a)
            arrays_subb.append(array_b)

    # XXX: this has better performance? but only works w/ shape-matching blocks
    # stacked_axes = (
    #     (0,) + tuple(ax + 1 for ax in axes_a),
    #     (0,) + tuple(ax + 1 for ax in axes_b),
    # )
    # for sector, (arrays_suba, arrays_subb) in new_blocks.items():
    #     if len(arrays_suba) == 1:
    #         # only one aligned block pair, simply tensordot
    #         new_blocks[sector] = _tensordot(
    #             arrays_suba[0], arrays_subb[0], axes=(axes_a, axes_b)
    #         )
    #     else:
    #         # multiple aligned blocks: stack and tensordot including new
    #         # stacked axis, which effectively sums over it
    #         arrays_suba = _stack(tuple(arrays_suba))
    #         arrays_subb = _stack(tuple(arrays_subb))
    #         new_blocks[sector] = _tensordot(
    #             arrays_suba, arrays_subb, axes=stacked_axes
    #         )

    new_indices = list(without(a.indices, axes_a) + without(b.indices, axes_b))
    # track charges that are no longer present due to block alignment
    charges_drop = [set(ix.charges) for ix in new_indices]

    for sector, (arrays_suba, arrays_subb) in new_blocks.items():
        new_blocks[sector] = functools.reduce(
            operator.add,
            (
                _tensordot(a, b, axes=(axes_a, axes_b))
                for a, b in zip(arrays_suba, arrays_subb)
            ),
        )
        # mark charges that are still present
        for i, c in enumerate(sector):
            charges_drop[i].discard(c)

    for i, cs in enumerate(charges_drop):
        if cs:
            new_indices[i] = new_indices[i].drop_charges(cs)

    return a.copy_with(
        indices=tuple(new_indices),
        charge=a.symmetry.combine(a.charge, b.charge),
        blocks=new_blocks,
    )


def drop_misaligned_sectors(
    a: AbelianArray,
    b: AbelianArray,
    axes_a: tuple[int, ...],
    axes_b: tuple[int, ...],
    inplace=False,
) -> tuple[AbelianArray, AbelianArray]:
    """Eagerly drop misaligned sectors of ``a`` and ``b`` so that they can be
    contracted via fusing.

    Parameters
    ----------
    a, b : AbelianArray
        The arrays to be contracted.
    axes_a, axes_b : tuple[int]
        The axes that will be contracted, defined like in `tensordot`.

    Returns
    -------
    a, b : AbelianArray
        The new arrays with misaligned sectors dropped.
    """
    # compute the intersection of fused charges for a and b
    sub_sectors_a = {
        sector: tuple(sector[ax] for ax in axes_a) for sector in a.sectors
    }
    sub_sectors_b = {
        sector: tuple(sector[ax] for ax in axes_b) for sector in b.sectors
    }
    allowed_subsectors = set(sub_sectors_a.values()).intersection(
        sub_sectors_b.values()
    )

    # filter out sectors of a that are not aligned with b
    new_blocks_a = {}
    charges_drop = [set(ix.charges) for ix in a.indices]
    for sector, array in a.blocks.items():
        if sub_sectors_a[sector] in allowed_subsectors:
            # keep the block
            new_blocks_a[sector] = array
            for i, c in enumerate(sector):
                # mark each charge as still present
                charges_drop[i].discard(c)

    # sync a index chargemaps with present sectors
    new_indices_a = tuple(
        ix.drop_charges(cs) if cs else ix
        for ix, cs in zip(a.indices, charges_drop)
    )

    # filter out sectors of b that are not aligned with a
    new_blocks_b = {}
    charges_drop = [set(ix.charges) for ix in b.indices]
    for sector, array in b.blocks.items():
        if sub_sectors_b[sector] in allowed_subsectors:
            # keep the block
            new_blocks_b[sector] = array
            for i, c in enumerate(sector):
                # mark each charge as still present
                charges_drop[i].discard(c)

    # sync b index chargemaps with present sectors
    new_indices_b = tuple(
        ix.drop_charges(cs) if cs else ix
        for ix, cs in zip(b.indices, charges_drop)
    )

    a = a._modify_or_copy(
        blocks=new_blocks_a, indices=new_indices_a, inplace=inplace
    )
    b = b._modify_or_copy(
        blocks=new_blocks_b, indices=new_indices_b, inplace=inplace
    )

    return a, b


def _tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes):
    """Perform a tensordot between two block arrays, by first fusing both into
    matrices and unfusing afterwards.

    Parameters
    ----------
    a, b : AbelianArray
        The arrays to be contracted.
    left_axes : tuple[int]
        The axes of ``a`` that will not be contracted.
    axes_a, axes_b : tuple[int]
        The axes that will be contracted, defined like in `tensordot`.
    right_axes : tuple[int]
        The axes of ``b`` that will not be contracted.
    """
    a, b = drop_misaligned_sectors(a, b, axes_a, axes_b)

    if not a.blocks or not b.blocks:
        # no aligned sectors, return empty array
        return a.copy_with(
            indices=without(a.indices, axes_a) + without(b.indices, axes_b),
            charge=a.symmetry.combine(a.charge, b.charge),
            blocks={},
        )

    # fuse into matrices or maybe vectors
    af = AbelianArray.fuse(a, left_axes, axes_a, expand_empty=False)
    bf = AbelianArray.fuse(b, axes_b, right_axes, expand_empty=False)

    # handle potential vector and scalar cases
    left_axes, axes_a = {
        (False, False): ((), ()),  # left scalar
        (False, True): ((), (0,)),  # left vector inner
        (True, False): ((0,), ()),  # left vector outer
        (True, True): ((0,), (1,)),  # left matrix
    }[bool(left_axes), bool(axes_a)]

    axes_b, right_axes = {
        (False, False): ((), ()),  # right scalar
        (False, True): ((), (0,)),  # right vector outer
        (True, False): ((0,), ()),  # right vector inner
        (True, True): ((0,), (1,)),  # right matrix
    }[bool(axes_b), bool(right_axes)]

    # tensordot the fused blocks
    cf = _tensordot_blockwise(af, bf, left_axes, axes_a, axes_b, right_axes)

    # unfuse result into (*left_axes, *right_axes)
    for ax in reversed(range(cf.ndim)):
        if cf.is_fused(ax):
            AbelianArray.unfuse(cf, ax, inplace=True)

    return cf


_DEFAULT_TENSORDOT_MODE = "auto"


def get_default_tensordot_mode():
    """Get the current default tensordot mode."""
    return _DEFAULT_TENSORDOT_MODE


def set_default_tensordot_mode(mode):
    """Set the default tensordot mode.

    Parameters
    ----------
    mode : {"auto", "fused", "blockwise", None}
        The mode to use for the contraction. None is no-op.
    """
    if mode is not None:
        global _DEFAULT_TENSORDOT_MODE
        _DEFAULT_TENSORDOT_MODE = mode


@contextlib.contextmanager
def default_tensordot_mode(mode):
    """Context manager to temporarily change the default tensordot mode.

    Parameters
    ----------
    mode : {"auto", "fused", "blockwise"}
        The mode to use for the contraction.
    """
    global _DEFAULT_TENSORDOT_MODE
    old_mode = _DEFAULT_TENSORDOT_MODE
    _DEFAULT_TENSORDOT_MODE = mode
    try:
        yield
    finally:
        _DEFAULT_TENSORDOT_MODE = old_mode


def parse_tensordot_axes(axes, ndim_a, ndim_b):
    """Parse the axes argument for single integer and also negative indices.
    Returning the 4 axes groups that can be used for fusing.
    """
    if isinstance(axes, int):
        axes_a = tuple(range(ndim_a - axes, ndim_a))
        axes_b = tuple(range(0, axes))
    else:
        axes_a, axes_b = axes
        axes_a = tuple(x % ndim_a for x in axes_a)
        axes_b = tuple(x % ndim_b for x in axes_b)
        if not len(axes_a) == len(axes_b):
            raise ValueError("Axes must have same length.")

    # axes left on the left and right tensors respectively
    left_axes = without(range(ndim_a), axes_a)
    right_axes = without(range(ndim_b), axes_b)

    return left_axes, axes_a, axes_b, right_axes


@tensordot.register(AbelianArray)
def tensordot_abelian(a, b, axes=2, mode="auto", preserve_array=False):
    """Tensordot between two block sparse abelian symmetric arrays.

    Parameters
    ----------
    a, b : AbelianArray
        The arrays to be contracted.
    axes : int or tuple[int]
        The axes to contract. If an integer, the last ``axes`` axes of ``a``
        will be contracted with the first ``axes`` axes of ``b``. If a tuple,
        the axes to contract in ``a`` and ``b`` respectively.
    mode : {"auto", "fused", "blockwise"}
        The mode to use for the contraction. If "auto", it will choose between
        "fused" and "blockwise" based on the number of axes to contract.
    preserve_array : bool, optional
        Whether to return a scalar if the result is a scalar.
    """
    if not isinstance(b, AbelianArray):
        if getattr(b, "ndim", 0) == 0:
            # assume scalar
            return a * b
        else:
            raise TypeError(f"Expected AbelianArray, got {type(b)}.")

    if DEBUG:
        a.check()
        b.check()

    left_axes, axes_a, axes_b, right_axes = parse_tensordot_axes(
        axes, a.ndim, b.ndim
    )

    if DEBUG:
        a.check_with(b, axes_a, axes_b)

    if mode is None:
        mode = _DEFAULT_TENSORDOT_MODE

    if mode == "auto":
        if len(axes_a) == 0:
            # outer product
            mode = "blockwise"
        else:
            mode = "fused"

    if mode == "fused":
        _tdot = _tensordot_via_fused
    elif mode == "blockwise":
        _tdot = _tensordot_blockwise
    else:
        raise ValueError(f"Unknown tensordot mode: {mode}.")

    c = _tdot(a, b, left_axes, axes_a, axes_b, right_axes)

    if DEBUG:
        c.check()
        c.check_chargemaps_aligned()
    # cf = _tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes)
    # cb = _tensordot_blockwise(a, b, left_axes, axes_a, axes_b, right_axes)
    # if not cf.allclose(cb):
    #     breakpoint()
    #     _tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes)
    #     raise ValueError("Blocks do not match.")

    if (c.ndim == 0) and (not preserve_array):
        try:
            return c.blocks[()]
        except KeyError:
            # no aligned blocks, return zero
            return 0.0

    return c


# --------------------------------------------------------------------------- #


class Z2Array(AbelianArray):
    """A block array with Z2 symmetry."""

    static_symmetry = get_symmetry("Z2")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.core import SparseTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import Z2

        blocks = [
            SubTensor(array, q_labels=tuple(map(Z2, sector)))
            for sector, array in self.blocks.items()
        ]

        data = SparseTensor(blocks)

        if flat:
            from pyblock3.algebra.flat import FlatSparseTensor

            data = FlatSparseTensor.from_sparse(data)

        data.shape = self.shape

        return data


@functools.cache
def get_zn_array_cls(n):
    """Get a block array class with ZN symmetry."""
    if n == 2:
        return Z2Array

    return type(
        f"Z{n}Array",
        (AbelianArray,),
        {"static_symmetry": get_symmetry(f"Z{n}")},
    )


class U1Array(AbelianArray):
    """A block array with U1 symmetry."""

    static_symmetry = get_symmetry("U1")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.core import SparseTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import U1

        blocks = [
            SubTensor(array, q_labels=tuple(map(U1, sector)))
            for sector, array in self.blocks.items()
        ]

        data = SparseTensor(blocks)

        if flat:
            from pyblock3.algebra.flat import FlatSparseTensor

            data = FlatSparseTensor.from_sparse(data)

        data.shape = self.shape

        return data

    def to_yastn(self, **config_opts):
        import yastn

        t = yastn.Tensor(
            config=yastn.make_config(sym="U1", **config_opts),
            s=tuple(-1 if ix.dual else 1 for ix in self.indices),
            n=self.charge,
        )
        for sector, array in self.blocks.items():
            t.set_block(ts=sector, Ds=array.shape, val=array)

        return t


class Z2Z2Array(AbelianArray):
    """A block array with Z2 x Z2 symmetry."""

    static_symmetry = get_symmetry("Z2Z2")


class U1U1Array(AbelianArray):
    """A block array with U1 x U1 symmetry."""

    static_symmetry = get_symmetry("U1U1")
