"""Methods that apply to abelian arrays with block sparse structure, both
fermionic and bosonic.
"""

import contextlib
import functools
import itertools
import math
import numbers
import operator
import os
import warnings
from collections import OrderedDict, defaultdict

import autoray as ar

from ..abelian_common import parse_tensordot_axes, without
from ..utils import DEBUG, get_array_cls, hasher, lazyabstractmethod
from .sparse_index import BlockIndex, SubIndexInfo
from .sparse_vector import BlockVector

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

    for sector in self.sectors:
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

    if self.num_blocks > _fuseinfo_cache_maxsectors:
        # too many sectors to cache
        global _fi_missed_too_long
        _fi_missed_too_long += 1
        return calc_fuse_block_info(self, axes_groups)

    key = hasher(
        (
            tuple(ix.hashkey() for ix in self.indices),
            self.sectors,
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
    sector_block_pairs,
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

    for sector, array in sector_block_pairs:
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
    sector_block_pairs,
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
    for sector, array in sector_block_pairs:
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


class SparseArrayCommon:
    def _init_abelian(
        self,
        indices,
        charge=None,
        blocks=(),
        symmetry=None,
        label=None,
    ):
        self._init_blockcommon(blocks)

        self._indices = tuple(indices)
        self._symmetry = self.get_class_symmetry(symmetry)
        self._label = label

        if charge is None:
            if self.num_blocks > 0:
                # infer the charge total from any sector
                sector = self.get_any_sector()
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

    def _new_with_abelian(self, indices, charge, blocks):
        new = self._new_with_blockcommon(blocks)
        new._indices = indices
        new._charge = charge
        new._symmetry = self._symmetry
        new._label = None
        return new

    def _copy_abelian(self):
        """Copy this abelian block sparse array."""
        new = self._copy_blockcommon()
        new._indices = self._indices
        new._charge = self._charge
        new._symmetry = self._symmetry
        new._label = self._label
        return new

    @lazyabstractmethod
    def copy(self):
        pass

    def _copy_with_abelian(self, indices=None, charge=None, blocks=None):
        """A copy of this block array with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self._copy_with_blockcommon(blocks=blocks)
        new._indices = self._indices if indices is None else indices
        new._charge = self._charge if charge is None else charge
        new._symmetry = self._symmetry
        new._label = self._label
        return new

    @lazyabstractmethod
    def copy_with(self, indices=None, charge=None, blocks=None):
        pass

    def _modify_abelian(self, indices=None, charge=None, blocks=None):
        """Modify this block array in place with some attributes replaced. Note
        that checks are not performed on the new properties, this is intended
        for internal use.
        """
        self._modify_blockcommon(blocks=blocks)

        if indices is not None:
            self._indices = indices
        if charge is not None:
            self._charge = charge

        return self

    @lazyabstractmethod
    def modify(self, indices=None, charge=None, blocks=None):
        pass

    @property
    def label(self):
        """The label of the array, possibly used for ordering odd parity
        fermionic modes."""
        return self._label

    @property
    def sizes(self):
        """The sizes of each index."""
        return tuple(ix.sizes for ix in self._indices)

    @property
    def charges(self):
        """The possible charges of each index."""
        return tuple(ix.charges for ix in self._indices)

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

    def sync_charges(self, inplace=False):
        """Given the blocks currently present, adjust the index chargemaps to
        match only those charges present in at least one sector.
        """
        charges_drop = [set(ix.charges) for ix in self.indices]
        for sector in self.sectors:
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
            if not self.has_sector(sector):
                shape = self.get_block_shape(sector)
                array = ar.do("zeros", shape, like=_ex_array)
                self.set_block(sector, array)

    def drop_missing_blocks(self):
        """Drop any present blocks that are all zero, resulting in a sparsity
        possibly less that 1.
        """
        _all = ar.get_lib_fn(self.backend, "all")
        for sector in self.sectors:
            if _all(self.get_block(sector) == 0.0):
                self.del_block(sector)

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
        SparseArrayCommon
        """
        symmetry = cls.get_class_symmetry(symmetry)

        if charge is None:
            charge = symmetry.combine()
        else:
            charge = charge

        new = cls(indices=indices, charge=charge, symmetry=symmetry, **kwargs)

        for sector in new.gen_valid_sectors():
            new.set_block(sector, fill_fn(new.get_block_shape(sector)))

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
        SparseArrayCommon
        """
        from ..utils import get_random_fill_fn

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

    def check(self):
        """Check that all the block sizes and charges are consistent."""
        for idx in self.indices:
            idx.check()

        for sector, array in self.get_sector_block_pairs():
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

        for sector in self.sectors:
            for i, c in enumerate(sector):
                actual_charges[i].add(c)

        if self.num_blocks > 0:
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
        other : SparseArrayCommon or BlockVector
            The other array or vector to compare to.
        *args
            The axes to compare, if ``other`` is a vector, the axis to compare
            with. If ``other`` is an array, then `axes_a` and `axes_b` should
            be given as if for a tensordot.
        """
        if isinstance(other, BlockVector):
            (ax,) = args
            for sector, array in self.get_sector_block_pairs():
                charge = sector[ax]
                v_block = other.get_block(charge)
                assert ar.shape(array)[ax] == ar.size(v_block)

        else:
            assert self.symmetry == other.symmetry

            axes_a, axes_b = args
            for axa, axb in zip(axes_a, axes_b):
                assert self.indices[axa].matches(other.indices[axb])

    def _allclose_abelian(self, other, **allclose_opts):
        """Test whether this ``SparseArrayCommon`` is close to another, that is,
        has all the same sectors, and the corresponding arrays are close.

        Parameters
        ----------
        other : SparseArrayCommon
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Returns
        -------
        bool
        """
        if self.is_zero() and other.is_zero():
            # charges can mismatch if both are zero
            return True

        # charge and signature must match, though if final object is scalar
        # then charge can not be inferred for flat arrays
        if self.charge != other.charge:
            return False

        if self.duals != other.duals:
            return False

        # now check blocks
        return self._allclose_blockcommon(other, **allclose_opts)

    @lazyabstractmethod
    def allclose(self, other, **allclose_opts):
        pass

    def _test_allclose_abelian(self, other, **allclose_opts):
        """Assert that this ``SparseArrayCommon`` is close to another,
        that is, has all the same sectors, and the corresponding arrays are
        close. Unlike `allclose`, this raises an AssertionError with details
        if not.

        Parameters
        ----------
        other : SparseArrayCommon
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Raises
        ------
        AssertionError
            If the arrays are not close.
        """
        if self.is_zero() and other.is_zero():
            # charges can mismatch if both are zero
            return True

        # charge and signature must match, though if final object is scalar
        # then charge can not be inferred for flat arrays
        if self.charge != other.charge:
            raise AssertionError(
                f"Charge mismatch: {self.charge} != {other.charge}"
            )

        if self.duals != other.duals:
            raise AssertionError(
                f"Signature mismatch: {self.duals} != {other.duals}"
            )

        # now check blocks
        return self._test_allclose_blockcommon(other, **allclose_opts)

    @lazyabstractmethod
    def test_allclose(self, other, **allclose_opts):
        pass

    @classmethod
    def from_blocks(cls, blocks, duals, charge=None, symmetry=None, **kwargs):
        """Create a block array from a dictionary of blocks and sequence of
        duals.

        Parameters
        ----------
        blocks : dict[tuple[hashable], array_like]
            A mapping of each 'sector' (tuple of charges) to the data array.
        duals : tuple[bool] or tuple[BlockIndex]
            The dual-ness of each index, or an explicitly given sequence of
            already constructed block sparse indices. Note in the latter case
            no checks for consistency are performed.
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
        SparseArrayCommon
        """
        symmetry = cls.get_class_symmetry(symmetry)

        ndim = len(next(iter(blocks.keys())))

        duals = tuple(duals)
        if len(duals) != ndim:
            raise ValueError(f"Expected {ndim} duals, got {len(duals)}.")

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

        indices = tuple(
            dual if isinstance(dual, BlockIndex) else BlockIndex(x, dual)
            for x, dual in zip(charge_size_maps, duals)
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
        index_maps : Sequence[Sequence[hashable]]
            For each dimension, the sequence mapping linear index to charge
            sector. There should be ``ndim`` such mappings, each of length
            ``shape[i]``.
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
        SparseArrayCommon
        """
        # XXX: warn if invalid blocks are non-zero?
        symmetry = cls.get_class_symmetry(symmetry)

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

    def _transpose_abelian(self, axes=None, inplace=False):
        """Transpose this block sparse abelian array.

        Parameters
        ----------
        axes : tuple[int, ...] | None, optional
            A permutation of the axes to transpose the array by. If None,
            the axes will be reversed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        SparseArrayCommon
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
                for sector, array in new.get_sector_block_pairs()
            },
        )

    @lazyabstractmethod
    def transpose(self, axes=None, inplace=False):
        pass

    def _conj_abelian(self, inplace=False) -> "SparseArrayCommon":
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

    def select_charge(self, axis, charge, subselect=None, inplace=False):
        """Drop all but the specified charge along the specified axis. Note the
        axis is not removed, it is simply restricted to a single charge.

        Parameters
        ----------
        axis : int
            The axis along which to select the charge.
        charge : int
            The charge to select along the specified axis.
        subselect : slice or array_like, optional
            If provided, a range of indices within the selected charge block
            to keep. If not provided, the entire block is kept.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        SparseArrayCommon
        """
        new = self if inplace else self.copy()

        if axis < 0:
            axis += new.ndim

        # update indices
        new_indices = (
            *new.indices[:axis],
            new.indices[axis].select_charge(charge, subselect),
            *new.indices[axis + 1 :],
        )

        # filter sectors

        def fn_filter(sector):
            return sector[axis] == charge

        if subselect is None:
            fn_block = None
        else:
            # ... and possibly slice blocks
            if isinstance(subselect, numbers.Integral):
                raise ValueError("subselect must be a slice or sequence.")

            selector = (
                *itertools.repeat(slice(None), axis),
                subselect,
                *itertools.repeat(slice(None), new.ndim - axis - 1),
            )

            def fn_block(block):
                return block[selector]

        new._map_blocks(fn_block=fn_block, fn_filter=fn_filter)
        new.modify(indices=new_indices)
        return new

    def _squeeze_abelian(self, axis=None, inplace=False):
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
        SparseArrayCommon
        """
        x = self if inplace else self.copy()

        if isinstance(axis, numbers.Integral):
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
        """Select a single (linear) index along the specified axis. The linear
        index is first converted to the corresponding charge and offset within
        that charge sector.

        Parameters
        ----------
        axis : int
            The axis to select along.
        idx : int
            The linear index to select.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.
        """
        if axis < 0:
            axis += self.ndim
        charge, offset = self.indices[axis].linear_to_charge_and_offset(idx)
        new = self.select_charge(
            axis,
            charge,
            subselect=(offset,),
            inplace=inplace,
        )
        return new.squeeze(axis, inplace=True)

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
        SparseArrayCommon
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

    def _fuse_core_abelian(
        self,
        *axes_groups,
        mode="auto",
        inplace=False,
    ):
        # ignore empty groups, expanding them is handled by `fuse`
        axes_groups = tuple(gaxes for gaxes in axes_groups if gaxes)

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
                self.get_sector_block_pairs(),
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
                self.get_sector_block_pairs(),
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

    @lazyabstractmethod
    def _fuse_core(self, *axes_groups, mode="auto", inplace=False):
        pass

    def _unfuse_abelian(self, axis, inplace=False):
        if axis < 0:
            axis += self.ndim

        backend = self.backend
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
        for sector, array in self.get_sector_block_pairs():
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

    def _matmul_abelian(self, other, preserve_array=False):
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
                return c.get_block(())
            except KeyError:
                # no aligned blocks, return zero
                return 0.0

        return c

    @lazyabstractmethod
    def __matmul__(self, other, preserve_array=False):
        pass

    def _trace_abelian(self):
        """Compute the trace of the block array, assuming it is a square
        matrix.
        """
        if self.ndim != 2:
            raise ValueError("Trace requires a 2D array.")

        _trace = ar.get_lib_fn(self.backend, "trace")

        return sum(
            _trace(array)
            for sector, array in self.get_sector_block_pairs()
            # only take diagonal blocks
            if sector[0] == sector[1]
        )

    @lazyabstractmethod
    def trace(self):
        pass

    def multiply_diagonal(self, v: BlockVector, axis, power=1, inplace=False):
        """Multiply this block array by a vector as if contracting a diagonal
        matrix along the given axis.

        Parameters
        ----------
        v : BlockVector
            The vector to contract with.
        axis : int
            The axis along which to contract.
        power : int or float, optional
            The power to raise the diagonal elements to.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        SparseArrayCommon
        """
        x = self if inplace else self.copy()

        if axis < 0:
            axis += x.ndim

        _reshape = ar.get_lib_fn(v.backend, "reshape")
        new_shape = tuple(-1 if i == axis else 1 for i in range(x.ndim))

        # sort by axis charge to group vector blocks
        sectors = sorted(x.sectors, key=lambda s: s[axis])
        v_charge = None

        for sector in sectors:
            charge = sector[axis]

            # only compute reshaped vector block when charge changes
            if charge != v_charge:
                try:
                    v_block = v.get_block(charge)
                    v_block = _reshape(v_block, new_shape)
                except KeyError:
                    v_block = None
                v_charge = charge

            if v_block is not None:
                # use broadcasting to perform "ab...X...c,X-> ab...X...c"

                if power == 1:
                    new_block = x.get_block(sector) * v_block
                elif power == -1:
                    new_block = x.get_block(sector) / v_block

                x.set_block(sector, new_block)
            else:
                # block isn't present -> like multiplying by zero
                if power == -1:
                    raise ZeroDivisionError(
                        "Cannot divide by implicitly zero (missing) "
                        f"block for charge {charge}."
                    )
                x.del_block(sector)

        if DEBUG:
            x.check()

        return x

    def ldmul(self, v: BlockVector, inplace=False):
        return self.multiply_diagonal(v, axis=-2, inplace=inplace)

    def rdmul(self, v: BlockVector, inplace=False):
        return self.multiply_diagonal(v, axis=-1, inplace=inplace)

    def lddiv(self, v: BlockVector, inplace=False):
        return self.multiply_diagonal(v, axis=-2, power=-1, inplace=inplace)

    def rddiv(self, v: BlockVector, inplace=False):
        return self.multiply_diagonal(v, axis=-1, power=-1, inplace=inplace)

    def align_axes(self, other, axes):
        """Align the axes of this block array with another, by dropping any
        sectors that are not aligned along the given axes, these can then be
        fused into a single axis that matches on both arrays.

        Parameters
        ----------
        other : SparseArrayCommon
            The other array to align with.
        axes : tuple[tuple[int]]
            The pairs of axes to align, given as tuples of the corresponding
            axes in this and the other array, a la tensordot.
        """
        return drop_misaligned_sectors(self, other, *axes)

    def _einsum_abelian(self, eq, preserve_array=False):
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
        SparseArrayCommon or scalar
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
        for sector, array in self.get_sector_block_pairs():
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

    @lazyabstractmethod
    def einsum(self, eq, preserve_array=False):
        pass

    def _tensordot_abelian(
        self, other, axes=2, mode="auto", preserve_array=False
    ):
        """Tensordot between two block sparse abelian symmetric arrays.

        Parameters
        ----------
        a, b : SparseArrayCommon
            The arrays to be contracted.
        axes : int or tuple[int]
            The axes to contract. If an integer, the last ``axes`` axes of
            ``a`` will be contracted with the first ``axes`` axes of ``b``. If
            a tuple, the axes to contract in ``a`` and ``b`` respectively.
        mode : {"auto", "fused", "blockwise"}
            The mode to use for the contraction. If "auto", it will choose
            between "fused" and "blockwise" based on the number of axes to
            contract.
        preserve_array : bool, optional
            Whether to return a scalar if the result is a scalar.
        """
        return tensordot_abelian(
            self,
            other,
            axes=axes,
            mode=mode,
            preserve_array=preserve_array,
        )

    def _to_dense_abelian(self):
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
                try:
                    array = self.get_block(partial_sector)
                except KeyError:
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

    @lazyabstractmethod
    def to_dense(self):
        pass

    def to_flat(self):
        """Convert this block sparse backend abelian or fermionic array to a
        flat backend abelian or fermionic array.
        """
        cls = get_array_cls(self.symmetry, self.fermionic, flat=True)
        return cls.from_blocksparse(self, symmetry=self.symmetry)

    # --------------------------- linalg methods ---------------------------- #

    def _qr_abelian(
        self, stabilized=False
    ) -> tuple["SparseArrayCommon", "SparseArrayCommon"]:
        if self.ndim != 2:
            raise NotImplementedError(
                "qr only implemented for 2D AbelianArrays,"
                f" got {self.ndim}D. Consider fusing first."
            )

        # get the 'lower' qr function that acts on the blocks
        _qr = _get_qr_fn(self.backend, stabilized=stabilized)

        q_blocks = {}
        r_blocks = {}
        new_chargemap = {}

        for sector, array in self.get_sector_block_pairs():
            q, r = _qr(array)
            q_blocks[sector] = q
            new_chargemap[sector[1]] = ar.shape(q)[1]
            # on r charge is 0, and dualnesses always opposite
            r_sector = (sector[1], sector[1])
            r_blocks[r_sector] = r

        bond_index = BlockIndex(new_chargemap, dual=self.indices[1].dual)

        q = self.copy_with(
            indices=(self.indices[0], bond_index),
            blocks=q_blocks,
        )
        r = self.new_with(
            indices=(bond_index.conj(), self.indices[1]),
            charge=self.symmetry.combine(),
            blocks=r_blocks,
        )

        if DEBUG:
            q.check()
            r.check()
            q.check_with(r, (1,), (0,))

        return q, r

    def _svd_abelian(self):
        if self.ndim != 2:
            raise NotImplementedError(
                "svd only implemented for 2D AbelianArrays,"
                f" got {self.ndim}D. Consider fusing first."
            )

        if self.backend == "numpy":
            _svd = get_numpy_svd_with_fallback()
        else:
            _svd = ar.get_lib_fn(self.backend, "linalg.svd")

        u_blocks = {}
        s_store = {}
        v_blocks = {}
        new_chargemap = {}

        for sector, array in self.get_sector_block_pairs():
            u, s, v = _svd(array)
            u_blocks[sector] = u
            # v charge is 0, and dualnesses always opposite
            s_charge = sector[1]
            v_sector = (s_charge, s_charge)
            s_store[s_charge] = s
            v_blocks[v_sector] = v
            new_chargemap[sector[1]] = ar.shape(u)[1]

        bond_index = BlockIndex(new_chargemap, dual=self.indices[1].dual)

        u = self.copy_with(
            indices=(self.indices[0], bond_index),
            blocks=u_blocks,
        )
        s = BlockVector(s_store)
        v = self.new_with(
            indices=(bond_index.conj(), self.indices[1]),
            charge=self.symmetry.combine(),
            blocks=v_blocks,
        )

        if DEBUG:
            u.check()
            s.check()
            v.check()
            u.check_with(s, 1)
            u.check_with(v, (1,), (0,))
            v.check_with(s, 0)

        return u, s, v

    def svd_truncated(
        self,
        cutoff=-1.0,
        cutoff_mode=4,
        max_bond=-1,
        absorb=0,
        renorm=0,
        **kwargs,
    ) -> tuple["SparseArrayCommon", "BlockVector", "SparseArrayCommon"]:
        """Truncated singular value decomposition of this sparse abelian
        symmetric array.

        Parameters
        ----------
        cutoff : float, optional
            Singular value cutoff threshold.
        cutoff_mode : int or str, optional
            How to perform the truncation:

            - 1 or 'abs': trim values below ``cutoff``
            - 2 or 'rel': trim values below ``s[0] * cutoff``
            - 3 or 'sum2': trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4 or 'rsum2': trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5 or 'sum1': trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6 or 'rsum1': trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

        max_bond : int
            An explicit maximum bond dimension, use -1 for none.
        absorb : {-1, 0, 1, None}
            How to absorb the singular values.

            - -1 or 'left': absorb into the left factor (U).
            - 0 or 'both': absorb the square root into both factors.
            - 1 or 'right': absorb into the right factor (VH).
            - None: do not absorb, return singular values as a BlockVector.

        renorm : {0, 1}
            Whether to renormalize the singular values (depends on
            `cutoff_mode`).
        """
        if kwargs:
            import warnings

            warnings.warn(
                f"Got unexpected kwargs {kwargs} in svd_truncated "
                f"for {self.__class__}. Ignoring them.",
                UserWarning,
            )

        # raw abelian or fermionic svd
        U, s, VH = self.svd()

        # then truncate according to the options
        return truncate_svd_result_blocksparse(
            U,
            s,
            VH,
            cutoff,
            _CUTOFF_MODE_MAP[cutoff_mode],
            max_bond,
            _ABSORB_MAP[absorb],
            renorm,
            backend=self.backend,
        )

    def _eigh_abelian(self) -> tuple["BlockVector", "SparseArrayCommon"]:
        if self.ndim != 2:
            raise NotImplementedError(
                "eigh only implemented for 2D AbelianArrays,"
                f" got {self.ndim}D. Consider fusing first."
            )
        if self.charge != self.symmetry.combine():
            raise ValueError(
                "Total charge much be the identity (zero) element."
            )

        _eigh = ar.get_lib_fn(self.backend, "linalg.eigh")

        eval_blocks = {}
        evec_blocks = {}

        for sector, array in self.get_sector_block_pairs():
            evals, evecs = _eigh(array)
            charge = sector[1]
            eval_blocks[charge] = evals
            evec_blocks[sector] = evecs

        eigenvalues = BlockVector(eval_blocks)
        eigenvectors = self.copy_with(blocks=evec_blocks)

        if DEBUG:
            eigenvectors.check_with(eigenvalues, 1)
            eigenvalues.check()

        return eigenvalues, eigenvectors

    def _eigh_truncated_abelian(
        self,
        cutoff=-1.0,
        cutoff_mode=4,
        max_bond=-1,
        absorb=0,
        renorm=0,
        positive=0,
        **kwargs,
    ) -> tuple["SparseArrayCommon", "BlockVector", "SparseArrayCommon"]:
        if kwargs:
            import warnings

            warnings.warn(
                f"Got unexpected kwargs {kwargs} in svd_truncated "
                f"for {self.__class__}. Ignoring them.",
                UserWarning,
            )

        s, U = self._eigh_abelian()

        # inplace sort by descending magnitude
        for sector, charge in zip(U.sectors, s.sectors):
            evals = s.get_block(charge)
            evecs = U.get_block(sector)

            if not positive:
                idx = ar.do(
                    "argsort",
                    -ar.do("abs", evals, like=self.backend),
                    like=self.backend,
                )
                s.set_block(charge, evals[idx])
                U.set_block(sector, evecs[:, idx])
            else:
                # assume positive, just need to flip
                s.set_block(charge, evals[::-1])
                U.set_block(sector, evecs[:, ::-1])

        if DEBUG:
            U.check()
            s.check()
            U.check_with(s, 1)

        return truncate_svd_result_blocksparse(
            U,
            s,
            U.H,
            cutoff,
            cutoff_mode,
            max_bond,
            absorb,
            renorm,
            backend=self.backend,
            use_abs=True,
        )

    def _solve_abelian(self, b: "SparseArrayCommon") -> "SparseArrayCommon":
        _solve = ar.get_lib_fn(self.backend, "linalg.solve")

        x_blocks = {}
        if (self.ndim, b.ndim) == (2, 1):
            # solve for single vector
            for sector, array in self.get_sector_block_pairs():
                b_sector = (sector[0],)
                if b.has_sector(b_sector):
                    x_sector = (sector[1],)
                    x_blocks[x_sector] = _solve(array, b.get_block(b_sector))
            x_indices = (self.indices[1].conj(),)

        elif (self.ndim, b.ndim) == (2, 2):
            # solve for stack of vectors
            map_b_sector = {}
            for sector in b.gen_valid_sectors():
                # charge of b array is fixed,
                # so each sector has unique sector[0]
                map_b_sector[sector[0]] = sector
            x_blocks = {}
            for sector, array in self.get_sector_block_pairs():
                if sector[0] in map_b_sector:
                    b_sector = map_b_sector[sector[0]]
                    b_array = b.get_block(b_sector)
                    x_sector = (sector[1], b_sector[1])
                    x_blocks[x_sector] = _solve(array, b_array)
            x_indices = (self.indices[1].conj(), b.indices[1])

        else:
            raise NotImplementedError(
                "solve only implemented for 2D and 1D or 2D and 2D "
                f"AbelianArrays, got {self.ndim}D and {b.ndim}D. "
                "Consider fusing first."
            )

        # c_x = c_b - c_A
        sym = self.symmetry
        x_charge = sym.combine(b.charge, sym.sign(self.charge))

        x = b.copy_with(
            blocks=x_blocks,
            indices=x_indices,
            charge=x_charge,
        )

        if DEBUG:
            x.check()
            self.check_with(x, (1,), (0,))

        return x


@functools.cache
def _get_qr_fn(backend, stabilized=False):
    """The lower level qr_stabilized is not necessarily already defined."""
    _qr = ar.get_lib_fn(backend, "linalg.qr")

    if not stabilized:
        return _qr

    try:
        _qr_stab = ar.get_lib_fn(backend, "qr_stabilized")

        def _qr(x):
            q, _, r = _qr_stab(x)
            return q, r

    except ImportError:
        _qr_ubstab = _qr
        _diag = ar.get_lib_fn(backend, "diag")
        _reshape = ar.get_lib_fn(backend, "reshape")
        _abs = ar.get_lib_fn(backend, "abs")

        def _sgn(x):
            x0 = x == 0.0
            return (x + x0) / (_abs(x) + x0)

        def _qr(x):
            q, r = _qr_ubstab(x)
            s = _sgn(_diag(r))
            q = q * _reshape(s, (1, -1))
            r = r * _reshape(s, (-1, 1))
            return q, r

    return _qr


@functools.cache
def get_numpy_svd_with_fallback():
    import numpy as np

    def svd_with_fallback(x):
        try:
            return np.linalg.svd(x, full_matrices=False)
        except np.linalg.LinAlgError:
            import scipy.linalg as sla

            return sla.svd(x, full_matrices=False, lapack_driver="gesvd")

    return svd_with_fallback


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


@functools.lru_cache(maxsize=2**14)
def calc_sub_max_bonds(sizes, max_bond):
    if max_bond < 0:
        # no limit
        return sizes

    # overall fraction of the total bond dimension to use
    frac = max_bond / sum(sizes)
    if frac >= 1.0:
        # keep all singular values
        return sizes

    # number of singular values to keep in each sector
    sub_max_bonds = [int(frac * sz) for sz in sizes]

    # distribute any remaining singular values to the smallest sectors
    rem = max_bond - sum(sub_max_bonds)

    for i in argsort(sub_max_bonds)[:rem]:
        sub_max_bonds[i] += 1

    return tuple(sub_max_bonds)


_CUTOFF_MODE_MAP = {
    1: 1,
    "abs": 1,
    2: 2,
    "rel": 2,
    3: 3,
    "sum2": 3,
    4: 4,
    "rsum2": 4,
    5: 5,
    "sum1": 5,
    6: 6,
    "rsum1": 6,
}

_ABSORB_MAP = {
    -1: -1,
    "left": -1,
    0: 0,
    "both": 0,
    1: 1,
    "right": 1,
    None: None,
}


def truncate_svd_result_blocksparse(
    U: SparseArrayCommon,
    s: BlockVector,
    VH: SparseArrayCommon,
    cutoff: float,
    cutoff_mode: int,
    max_bond: int,
    absorb: int | str | None,
    renorm: int,
    backend: str = None,
    use_abs: bool = False,
) -> tuple[SparseArrayCommon, BlockVector, SparseArrayCommon]:
    if renorm:
        raise NotImplementedError("renorm not implemented yet.")

    if cutoff > 0.0:
        # first combine all singular values into a single, sorted array
        sall = s.to_dense()
        if use_abs:
            sall = ar.do("abs", sall, like=backend)
        sall = ar.do("sort", sall, like=backend)

        cutoff_mode = _CUTOFF_MODE_MAP[cutoff_mode]

        if cutoff_mode == 1:
            # absolute cutoff
            abs_cutoff = cutoff
        elif cutoff_mode == 2:
            # relative cutoff
            abs_cutoff = sall[-1] * cutoff
        else:
            # possibly square singular values
            power = {3: 2, 4: 2, 5: 1, 6: 1}[cutoff_mode]
            if power == 1:
                # sum1 or rsum1
                cum_spow = ar.do("cumsum", sall, 0, like=backend)
            else:
                # sum2 or rsum2
                cum_spow = ar.do("cumsum", sall**power, 0, like=backend)

            if cutoff_mode in (4, 6):
                # rsum1 or rsum2: relative cumulative cutoff
                cond = cum_spow >= cutoff * cum_spow[-1]
            else:
                # sum1 or sum2: absolute cumulative cutoff
                cond = cum_spow >= cutoff

            # translate to total number of singular values to keep
            n_chi_all = ar.do("count_nonzero", cond, like=backend)
            # and then to an absolute cutoff value
            abs_cutoff = sall[-n_chi_all]

        if 0 < max_bond < ar.size(sall):
            # also take into account a total maximum bond
            max_bond_cutoff = sall[-max_bond]
            if max_bond_cutoff > abs_cutoff:
                abs_cutoff = max_bond_cutoff

        # now find number of values to keep per sector
        sub_max_bonds = [
            int(ar.do("count_nonzero", ss >= abs_cutoff, like=backend))
            for ss in s.get_all_blocks()
        ]
    else:
        # size of each sector
        sector_sizes = tuple(map(ar.size, s.get_all_blocks()))
        # distribute max_bond proportionally to sector sizes
        sub_max_bonds = calc_sub_max_bonds(sector_sizes, max_bond)

    new_inner_chargemap = {}
    for (c0, c1), n_chi in zip(U.sectors, sub_max_bonds):
        # check how many singular values from this sector are valid
        if n_chi == 0:
            # remove this sector entirely
            U.del_block((c0, c1))
            s.del_block(c1)
            VH.del_block((c1, c1))
            continue

        # slice the values and left and right vectors
        U.set_block((c0, c1), U.get_block((c0, c1))[:, :n_chi])
        s.set_block(c1, s.get_block(c1)[:n_chi])
        VH.set_block((c1, c1), VH.get_block((c1, c1))[:n_chi, :])

        # make sure the index chargemaps are updated too
        new_inner_chargemap[c1] = n_chi

    new_inner_chargemap = dict(sorted(new_inner_chargemap.items()))

    # make sure to drop the inner fusing info which is not longer valid
    U.modify(
        indices=(
            U.indices[0],
            U.indices[1].copy_with(
                chargemap=new_inner_chargemap,
                subinfo=None,
            ),
        )
    )
    VH.modify(
        indices=(
            VH.indices[0].copy_with(
                chargemap=new_inner_chargemap,
                subinfo=None,
            ),
            VH.indices[1],
        )
    )

    if absorb is None:
        if DEBUG:
            U.check_with(s, 1)
            s.check()
            VH.check_with(s, 0)
            U.check_with(VH, (1,), (0,))

        return U, s, VH

    # absorb the singular values block by block
    for c0, c1 in U.sectors:
        if absorb in (-1, "left"):
            U.set_block(
                (c0, c1),
                U.get_block((c0, c1)) * s.get_block(c1).reshape((1, -1)),
            )
        elif absorb in (1, "right"):
            VH.set_block(
                (c1, c1),
                VH.get_block((c1, c1)) * s.get_block(c1).reshape((-1, 1)),
            )
        elif absorb in (0, "both"):
            s_sqrt = ar.do("sqrt", s.get_block(c1), like=backend)
            U.set_block(
                (c0, c1), U.get_block((c0, c1)) * s_sqrt.reshape((1, -1))
            )
            VH.set_block(
                (c1, c1), VH.get_block((c1, c1)) * s_sqrt.reshape((-1, 1))
            )
        else:
            raise ValueError(f"Unknown absorb value: {absorb}")

    if DEBUG:
        U.check()
        U.check_with(s, 1)
        s.check()
        VH.check()
        VH.check_with(s, 0)
        U.check_with(VH, (1,), (0,))

    return U, None, VH


# --------------------------------------------------------------------------- #


def _tensordot_blockwise(a, b, left_axes, axes_a, axes_b, right_axes):
    """Perform a tensordot between two block arrays, performing the contraction
    of each pair of aligned blocks separately.
    """
    aligned_blocks = defaultdict(list)

    # iterate over all valid sectors of the new SparseArrayCommon
    _tensordot = ar.get_lib_fn(a.backend, "tensordot")
    # _stack = ar.get_lib_fn(a.backend, "stack")

    # group blocks of `b` by which contracted charges they are aligned to
    for sector, array_b in b.get_sector_block_pairs():
        sector_contracted = tuple(sector[i] for i in axes_b)
        sector_right = tuple(sector[i] for i in right_axes)
        aligned_blocks[sector_contracted].append((sector_right, array_b))

    # accumulate aligned blocks of `a` into a pair of lists
    new_blocks = {}
    for sector, array_a in a.get_sector_block_pairs():
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

    return a.new_with(
        indices=tuple(new_indices),
        charge=a.symmetry.combine(a.charge, b.charge),
        blocks=new_blocks,
    )


def drop_misaligned_sectors(
    a: SparseArrayCommon,
    b: SparseArrayCommon,
    axes_a: tuple[int, ...],
    axes_b: tuple[int, ...],
    inplace=False,
) -> tuple[SparseArrayCommon, SparseArrayCommon]:
    """Eagerly drop misaligned sectors of ``a`` and ``b`` so that they can be
    contracted via fusing.

    Parameters
    ----------
    a, b : SparseArrayCommon
        The arrays to be contracted.
    axes_a, axes_b : tuple[int]
        The axes that will be contracted, defined like in `tensordot`.

    Returns
    -------
    a, b : SparseArrayCommon
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
    for sector, array in a.get_sector_block_pairs():
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
    for sector, array in b.get_sector_block_pairs():
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
    a, b : SparseArrayCommon
        The arrays to be contracted.
    left_axes : tuple[int]
        The axes of ``a`` that will not be contracted.
    axes_a, axes_b : tuple[int]
        The axes that will be contracted, defined like in `tensordot`.
    right_axes : tuple[int]
        The axes of ``b`` that will not be contracted.
    """
    a, b = drop_misaligned_sectors(a, b, axes_a, axes_b, inplace=False)

    if a.num_blocks == 0 or b.num_blocks == 0:
        # no aligned sectors, return empty array
        return a.new_with(
            indices=without(a.indices, axes_a) + without(b.indices, axes_b),
            charge=a.symmetry.combine(a.charge, b.charge),
            blocks={},
        )

    # fuse into matrices or maybe vectors
    if len(left_axes) > 1 or len(axes_a) > 1:
        af = a._fuse_core_abelian(left_axes, axes_a, inplace=True)
    elif left_axes + axes_a == (1, 0):
        # this is only other case where not already aligned
        af = a._transpose_abelian(left_axes + axes_a, inplace=True)
    else:
        af = a
    unfuse_left = len(left_axes) > 1

    if len(axes_b) > 1 or len(right_axes) > 1:
        bf = b._fuse_core_abelian(axes_b, right_axes, inplace=True)
    elif axes_b + right_axes == (1, 0):
        # this is only other case where not already aligned
        bf = b._transpose_abelian(axes_b + right_axes, inplace=True)
    else:
        bf = b
    unfuse_right = len(right_axes) > 1

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
    if unfuse_right:
        cf._unfuse_abelian(-1, inplace=True)
    if unfuse_left:
        cf._unfuse_abelian(0, inplace=True)

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


def tensordot_abelian(a, b, axes=2, mode="auto", preserve_array=False):
    """Tensordot between two block sparse abelian symmetric arrays.

    Parameters
    ----------
    a, b : SparseArrayCommon
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
    if not isinstance(b, SparseArrayCommon):
        if getattr(b, "ndim", 0) == 0:
            # assume scalar
            return a * b
        else:
            raise TypeError(f"Expected SparseArrayCommon, got {type(b)}.")

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
            return c.get_block(())
        except KeyError:
            # no aligned blocks, return zero
            return 0.0

    return c
