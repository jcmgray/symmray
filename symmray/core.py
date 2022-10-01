import math
import functools
import itertools
from collections import defaultdict

import autoray as ar
from autoray.lazy.core import find_full_reshape


# defining symmetries just requires composition function (which returns the
# identity element if empty) and probably a negation function for non-int reprs


@functools.lru_cache(2**15)
def z2_symmetry(*charges):
    return sum(charges) % 2


symmetry = z2_symmetry


# --------------------------------------------------------------------------- #


class BlockIndex:
    """An index of a blocked tensor.

    Parameters
    ----------
    chargemap : dict[int, int]
        A mapping from charge to size.
    flow : bool, optional
        Whether the index flows 'inwards' / (+ve) = ``False`` or 'outwards' /
        (-ve) = ``True``. I.e. the sign is given by ``(-1) ** flow``.
    subinfo : SubIndexInfo, optional
        Information about the subindices of this index and their extents if
        this index was formed from fusing.
    """

    __slots__ = (
        "_chargemap",
        "_flow",
        "_subinfo",
    )

    def __init__(self, chargemap, flow=False, subinfo=None):
        self._chargemap = dict(chargemap)
        self._flow = flow
        self._subinfo = subinfo

    @property
    def chargemap(self):
        """A mapping from charge to size."""
        return self._chargemap

    @property
    def flow(self):
        """Whether the index flows 'inwards' / (+ve) = ``False`` or 'outwards'
        / (-ve) = ``True``. I.e. the sign is given by ``(-1) ** flow``.
        """
        return self._flow

    @property
    def subinfo(self):
        """Information about the subindices of this index and their extents if
        this index was formed from fusing.
        """
        return self._subinfo

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

    def copy(self):
        """A copy of this index."""
        new = self.__new__(self.__class__)
        new._chargemap = self._chargemap.copy()
        new._flow = self._flow
        new._subinfo = self._subinfo
        return new

    def conj(self):
        """A copy of this index with the flow reversed."""
        new = self.__new__(self.__class__)
        new._chargemap = self._chargemap.copy()
        new._flow = not self._flow
        new._subinfo = None if self._subinfo is None else self._subinfo.conj()
        return new

    def size_of(self, c):
        """The size of the block with charge ``c``."""
        return self._chargemap[c]

    def matches(self, other):
        """Whether this index matches ``other`` index, namely, whether the
        ``chargemap`` of each matches, their flows are opposite, and also
        whether their subindices match, if they have any. For debugging.
        """
        return (
            (self.chargemap == other.chargemap)
            and (self.flow ^ other.flow)
            and (
                (self.subinfo is other.subinfo is None)
                or (self.subinfo.matches(other.subinfo))
            )
        )

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"chargemap={self.chargemap}, flow={self.flow}",
                (
                    f", subinfo={self.subinfo}"
                    if self.subinfo is not None
                    else ""
                ),
                ")",
            ]
        )


class SubIndexInfo:
    """Holder class for storing the relevant information for unfusing.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices (ordered) that were fused to make this index.
    extents : dict[int, dict[int, int]]
        A mapping of each charge of the fused index to a mapping of each
        subsector (combination of sub charges) to the size of that subsector.
    """

    __slots__ = ("extents", "indices")

    def __init__(self, indices, extents):
        self.indices = indices
        self.extents = extents

    def conj(self):
        """A copy of this subindex information with the relevant flows
        reversed.
        """
        new = self.__new__(self.__class__)
        new.indices = tuple(ix.conj() for ix in self.indices)
        new.extents = self.extents
        return new

    def matches(self, other):
        """Whether this subindex information matches ``other`` subindex
        information, namely, whether the ``indices`` and ``extents`` match.
        For debugging.
        """
        return (
            all(i.matches(j) for i, j in zip(self.indices, other.indices))
            and self.extents == other.extents
        )

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"indices={self.indices}",
                f", extents={self.extents}",
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
    for size in sizes:
        x.append(x[-1] + size)
    return tuple(x[1:-1])


def nested_dict():
    """Return an arbitrarily nest-able dict."""
    return defaultdict(nested_dict)


def nested_setitem(d, keys, value):
    """Set a value in an arbitrarily nested dict."""
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = value


@functools.lru_cache(2**15)
def reshape_to_fuse_axes(shape, newshape):
    """Assuming only fuses need to happen, convert from ``reshhape`` form to
    ``fuse`` form -  a sequence of axes groups.
    """
    i = 0
    d = 1
    groups = []
    group = []
    for dnew in newshape:
        while dnew > d:
            # accumulate the next axis
            d *= shape[i]
            group.append(i)
            i += 1
        if d != dnew:
            raise ValueError(f"Cannot reshape {shape} to {newshape}")
        if len(group) > 1:
            # only record fused axes
            groups.append(tuple(group))
        # reset
        d = 1
        group = []
    return tuple(groups)


def reshape_to_unfuse_axes(indices, newshape):
    """Assuming only unfuses need to happen, convert from ``reshape`` form to
    a sequence of ``axes`` to be supplied to ``unfuse``.
    """
    i = 0
    d = 1
    unfused = []
    for j, ix in enumerate(indices):
        dold = ix.size_total
        if ix.subinfo is None:
            # unfused axis, just make sure dimension matches
            if dold != newshape[i]:
                shape = tuple(ix.size_total for ix in indices)
                raise ValueError(
                    f"Cannot reshape {shape} to {newshape}, "
                    f"axis {j} has size {dold} but new axis has size {d}"
                )
            i += 1
        else:
            # fused axis, make all subindex dimensions match too
            unfused.append(j)
            for subindex in ix.subinfo.indices:
                dsub = newshape[i]
                if dsub != subindex.size_total:
                    shape = tuple(ix.size_total for ix in indices)
                    raise ValueError(
                        f"Cannot reshape {shape} to {newshape}, "
                        f"subindex sizes for axes {j} do not match"
                    )
                d *= dsub
                i += 1
            d = 1

    return tuple(sorted(unfused, reverse=True))


class BlockArray:
    """A block sparse array with symmetry constraints.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices of the array.
    charge_total : hashable
        The total charge of the array.
    blocks : dict[tuple[hashable], array_like]
        A mapping of each 'sector' (tuple of charges) to the data array.
    """

    __slots__ = (
        "_indices",
        "_blocks",
        "_charge_total",
    )

    def __init__(
        self,
        indices,
        charge_total,
        blocks=(),
    ):
        self._indices = tuple(indices)
        self._charge_total = charge_total
        self._blocks = dict(blocks)

    @property
    def indices(self):
        """The indices of the array."""
        return self._indices

    @property
    def blocks(self):
        """The blocks of the array."""
        return self._blocks

    @property
    def sizes(self):
        """The sizes of each index."""
        return tuple(ix.sizes for ix in self._indices)

    @property
    def charges(self):
        """The possible charges of each index."""
        return tuple(ix.charges for ix in self._indices)

    @property
    def flows(self):
        """The flows of each index."""
        return tuple(ix.flow for ix in self._indices)

    @property
    def charge_total(self):
        """The total charge of the array."""
        return self._charge_total

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

    def _get_any_array(self):
        return next(iter(self._blocks.values()))

    @property
    def dtype(self):
        return ar.get_dtype_name(self._get_any_array())

    @property
    def backend(self):
        return ar.infer_backend(self._get_any_array())

    @property
    def num_blocks(self):
        """The number of blocks in the array."""
        return len(self._blocks)

    def copy(self):
        """Copy this block array."""
        new = self.__new__(self.__class__)
        new._indices = self._indices
        new._charge_total = self._charge_total
        new._blocks = self._blocks.copy()
        return new

    def is_valid_sector(self, sector):
        """Check if a sector is valid for the block array, i.e., whether the
        total symmetry charge is satisfied.
        """
        flowed_sector = (
            -c if i.flow else c for c, i in zip(sector, self._indices)
        )
        block_charge = symmetry(*flowed_sector)
        return block_charge == self.charge_total

    def gen_valid_sectors(self):
        """Generate all valid sectors for the block array."""
        return filter(self.is_valid_sector, itertools.product(*self.charges))

    def get_block_shape(self, sector):
        """Get the shape of the block corresponding to a given sector."""
        return tuple(ix.size_of(c) for ix, c in zip(self._indices, sector))

    def check(self):
        """Check that all the block sizes and charges are consistent."""
        for sector, array in self.blocks.items():
            assert self.is_valid_sector(sector)
            assert all(
                di == dj
                for di, dj in zip(array.shape, self.get_block_shape(sector))
            )

    @classmethod
    def from_fill_fn(
        cls,
        fill_fn,
        indices,
        charge_total=None,
    ):
        """Generate a block array from a filling function. Every valid sector
        will be filled with the result of the filling function.

        Parameters
        ----------
        fill_fn : callable
            The filling function, with signature ``fill_fn(shape)``.
        indices : tuple[BlockIndex]
            The indices of the array.
        charge_total : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.
        """
        new = cls.__new__(cls)
        new._indices = tuple(indices)
        if charge_total is None:
            charge_total = symmetry()
        else:
            new._charge_total = charge_total
        new._blocks = {
            sector: fill_fn(new.get_block_shape(sector))
            for sector in new.gen_valid_sectors()
        }
        return new

    @classmethod
    def random(cls, indices, charge_total=None, seed=None, dist="normal"):
        """Create a block array with random values.

        Parameters
        ----------
        indices : tuple[BlockIndex]
            The indices of the array.
        charge_total : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.
        seed : int
            The random seed.
        dist : str
            The distribution to use. Can be one of ``"normal"``, ``"uniform"``,
            etc., see :func:`numpy.random.default_rng` for details.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        rand_fn = getattr(rng, dist)

        def fill_fn(shape):
            return rand_fn(size=shape)

        return cls.from_fill_fn(fill_fn, indices, charge_total)

    @classmethod
    def from_blocks(cls, blocks, flows, charge_total=None):
        """Create a block array from a dictionary of blocks and sequence of
        flows.

        Parameters
        ----------
        blocks : dict[tuple[hashable], array_like]
            A mapping of each 'sector' (tuple of charges) to the data array.
        flows : tuple[bool]
            The flow of each index.
        charge_total : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.

        Returns
        -------
        BlockArray
        """
        self = cls.__new__(cls)
        if charge_total is None:
            self._charge_total = symmetry()
        else:
            self._charge_total = charge_total
        self._blocks = dict(blocks)

        ndim = len(next(iter(blocks.keys())))
        charge_size_maps = [{} for _ in range(ndim)]

        for sector, array in self.blocks.items():
            for i, (c, d) in enumerate(zip(sector, array.shape)):
                d = int(d)
                d_existing = charge_size_maps[i].get(c, None)
                if d_existing is None:
                    charge_size_maps[i][c] = d
                elif d != d_existing:
                    raise ValueError(
                        f"Inconsistent block sizes for index {i}"
                        f" with charge {c}: {d_existing} != {d}."
                    )

        flows = tuple(flows)
        if len(flows) != ndim:
            raise ValueError(f"Expected {ndim} flows, got {len(flows)}.")

        self._indices = tuple(
            BlockIndex(x, f) for x, f in zip(charge_size_maps, flows)
        )

        return self

    @classmethod
    def from_dense(cls, array, index_maps, flows, charge_total=None):
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
        flows : tuple[bool]
            The flow of each index.
        charge_total : hashable
            The total charge of the array. If not given, it will be
            taken as the identity / zero element.
        """
        if charge_total is None:
            charge_total = symmetry()

        # first we work out which indices of which axes belong to which charges
        charge_groups = []
        for d, index_map in zip(array.shape, index_maps):
            which_charge = {}
            for i in range(d):
                which_charge.setdefault(index_map[i], []).append(i)
            charge_groups.append(which_charge)

        # then we recusively visit all the potential blocks, by slicing using
        # the above generated charge groupings
        blocks = {}
        ndim = array.ndim
        all_sliced = [slice(None)] * ndim

        def _recurse(ary, j=0, sector=()):
            if j < ndim:
                for charge, indices in charge_groups[j].items():
                    # for each charge, select all the indices along axis j
                    # that belong to it, then recurse further
                    selector = all_sliced.copy()
                    selector[j] = indices
                    subarray = ary[tuple(selector)]
                    _recurse(subarray, j + 1, sector + (charge,))
            else:
                # we have reached a fully specified block
                if symmetry(*sector) == charge_total:
                    # ... but only add valid ones:
                    blocks[sector] = ary

        # generate the blocks
        _recurse(array)

        # generate the indices -> the charge_map is simply the group size
        indices = [
            BlockIndex({c: len(g) for c, g in charge_group.items()}, flow=flow)
            for charge_group, flow in zip(charge_groups, flows)
        ]

        # create the block array!
        return cls(blocks=blocks, indices=indices, charge_total=charge_total)

    def apply_to_arrays(self, fn):
        """Apply the ``fn`` inplace to the array of every block."""
        for sector, array in self._blocks.items():
            self._blocks[sector] = fn(array)

    def to_dense(self):
        """Convert this block array to a dense array."""
        backend = self.backend
        _zeros = ar.get_lib_fn(backend, "zeros")
        _concat = ar.get_lib_fn(backend, "concatenate")

        def filler(shape):
            return _zeros(shape, dtype=self.dtype)

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
                    for c in self.indices[i].charges
                )
                # then concatenate along the current axis
                return _concat(arrays, axis=i)

        return _recurse_all_charges()

    def conj(self):
        """Return the complex conjugate of this block array, including the
        indices."""
        new = self.copy()
        _conj = ar.get_lib_fn(new.backend, "conj")
        new.apply_to_arrays(_conj)
        new._indices = tuple(ix.conj() for ix in self._indices)
        return new

    def transpose(self, axes=None):
        """Transpose the block array."""
        _transpose = ar.get_lib_fn(self.backend, "transpose")

        if axes is None:
            # reverse the axes
            axes = tuple(range(self.ndim - 1, -1, -1))

        new = self.__new__(self.__class__)
        new._indices = permuted(self._indices, axes)
        new._charge_total = self.charge_total
        new._blocks = {
            permuted(sector, axes): _transpose(array, axes)
            for sector, array in self.blocks.items()
        }
        return new

    def fuse(self, *axes_groups):
        """Fuse the give group or groups of axes. The new fused axes will be
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

        Parameters
        ----------
        axes_groups : sequence of sequences of int
            The axes to fuse. Each group of axes will be fused into a single
            axis.
        """
        # handle empty groups
        axes_groups = tuple(filter(None, axes_groups))
        if not axes_groups:
            # ... and no groups -> nothing to do
            return self.copy()

        backend = self.backend
        _transpose = ar.get_lib_fn(backend, "transpose")
        _reshape = ar.get_lib_fn(backend, "reshape")
        _concatenate = ar.get_lib_fn(backend, "concatenate")

        # which group does each axis appear in, if any
        num_groups = len(axes_groups)
        ax2group = {ax: g for g, axes in enumerate(axes_groups) for ax in axes}
        for i in range(self.ndim):
            ax2group.setdefault(i, None)

        # the permutation will be the same for every block: precalculate
        # n.b. all new groups will be inserted at the *first fused axis*
        position = min((min(g) for g in axes_groups))
        axes_before = tuple(
            ax for ax in range(position) if ax2group[ax] is None
        )
        axes_after = tuple(
            ax for ax in range(position, self.ndim) if ax2group[ax] is None
        )
        perm = (
            *axes_before,
            *(ax for g in axes_groups for ax in g),
            *axes_after,
        )

        # track where each axis will be in the new array
        new_axes = {ax: ax for ax in axes_before}
        for i, g in enumerate(axes_groups):
            for ax in g:
                new_axes[ax] = position + i
        for i, ax in enumerate(axes_after):
            new_axes[ax] = position + num_groups + i
        new_ndim = len(axes_before) + num_groups + len(axes_after)

        old_indices = self._indices
        new_blocks = nested_dict()
        subindex_extents = [defaultdict(dict) for _ in range(num_groups)]

        for sector, array in self.blocks.items():
            # keep track of a perm+shape in order to fuse the actual array
            new_shape = [1] * new_ndim
            # the key of the new fused block to add this block to
            new_sector = [symmetry()] * new_ndim
            # only the parts of the sector that will be fused
            subsectors = [[] for g in range(num_groups)]

            for i, c in enumerate(sector):
                # the size of charge `c` on the `i`th axis
                ix = old_indices[i]
                d = ix.size_of(c)

                g = ax2group[i]
                new_ax = new_axes[i]
                if g is None:
                    # not fusing, new value is just copied
                    new_sector[new_ax] = c
                    new_shape[new_ax] = d
                else:
                    # fusing: need to accumulate
                    new_shape[new_ax] *= d
                    new_sector[new_ax] = symmetry(
                        new_sector[new_ax], -c if ix.flow else c
                    )
                    subsectors[g].append(c)

            # make hashable
            new_sector = tuple(new_sector)
            new_shape = tuple(new_shape)

            # fuse (via transpose+reshape) the actual array, to concat, later
            new_array = _transpose(array, perm)
            new_array = _reshape(new_array, new_shape)
            # the following is like performing:
            # new_blocks[new_sector][subsect0][subsect1]... = new_array
            nested_setitem(
                new_blocks, (new_sector, *map(tuple, subsectors)), new_array
            )

            # keep track of the new blocksize of each fused index, for unfusing
            for g, subsector in enumerate(subsectors):
                new_charge = new_sector[position + g]
                new_size = new_shape[position + g]
                subindex_extents[g][new_charge][tuple(subsector)] = new_size

        new = self.__new__(self.__class__)
        new._charge_total = self._charge_total

        def _recurse_sorted_concat(d, group_level=0):
            if group_level == num_groups:
                return d  # is lowest level -> i.e. the actual arrays
            # else, recurse and concat the next level down
            return _concatenate(
                tuple(
                    _recurse_sorted_concat(x[1], group_level + 1)
                    for x in sorted(d.items(), key=lambda x: x[0])
                ),
                axis=position + group_level,
            )

        new._blocks = {
            k: _recurse_sorted_concat(v) for k, v in new_blocks.items()
        }

        # the unique sequence of new charges/sizes in ascending order
        grouped_indices = (
            BlockIndex(
                chargemap={
                    c: sum(charge_extent.values())
                    for c, charge_extent in subindex_extents[g].items()
                },
                flow=None,
                # flow=old_indices[gaxes[0]].flow,
                # for unfusing
                subinfo=SubIndexInfo(
                    indices=tuple(old_indices[i] for i in gaxes),
                    extents={
                        c: tuple(sorted(charge_extent.items()))
                        for c, charge_extent in subindex_extents[g].items()
                    },
                ),
            )
            for g, gaxes in enumerate(axes_groups)
        )
        new._indices = (
            *(old_indices[ax] for ax in axes_before),
            *grouped_indices,
            *(old_indices[ax] for ax in axes_after),
        )

        return new

    def unfuse(self, axis):
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.
        """
        backend = self.backend
        _split = ar.get_lib_fn(backend, "split")
        _reshape = ar.get_lib_fn(backend, "reshape")

        # get required information from the fused index
        subinfo = self.indices[axis].subinfo

        # info for how to split/slice the linear index into sub charges
        subindex_splits = {
            c: accum_for_split(x[1] for x in charge_extent)
            for c, charge_extent in subinfo.extents.items()
        }

        new_blocks = {}
        for sector, array in self.blocks.items():
            old_charge = sector[axis]
            old_shape = array.shape

            charge_extent = subinfo.extents[old_charge]
            splits = subindex_splits[old_charge]
            new_arrays = _split(array, splits, axis=axis)

            for (subsector, _), new_array in zip(charge_extent, new_arrays):
                # expand the old charge into the new subcharges
                new_key = replace_with_seq(sector, axis, subsector)

                # reshape the array to the correct shape
                subshape = tuple(
                    ix.size_of(c) for ix, c in zip(subinfo.indices, subsector)
                )
                new_shape = replace_with_seq(old_shape, axis, subshape)

                # reshape and store!
                new_blocks[new_key] = _reshape(new_array, new_shape)

        new = self.__new__(self.__class__)
        new._indices = replace_with_seq(self.indices, axis, subinfo.indices)
        new._charge_total = self._charge_total
        new._blocks = new_blocks
        return new

    def unfuse_all(self):
        """Unfuse all indices that carry subindex information, likely from a
        fusing operation.
        """
        new = self.copy()
        for ax in range(self.ndim - 1, -1, -1):
            if self.indices[ax].subinfo is not None:
                new = new.unfuse(ax)
        return new

    def __str__(self):
        s = (
            f"{self.__class__.__name__}(ndim={self.ndim}, "
            f"charge_total={self.charge_total}, dims=[\n"
        )
        for i in range(self.ndim):
            s += (
                f"    ({self.shape[i]} = "
                f"{f'+'.join(map(str, self.sizes[i]))} : "
                f"{'+' if self.flows[i] else '-'}"
                f"[{f','.join(map(str, self.charges[i]))}]),\n"
            )
        s += (
            f"], num_blocks={self.num_blocks}, backend={self.backend}, "
            f"dtype={self.dtype})"
        )
        return s

    def _reshape_via_fuse(self, newshape):
        axes_groups = reshape_to_fuse_axes(self.shape, newshape)
        return self.fuse(*axes_groups)

    def _reshape_via_unfuse(self, newshape):
        fused_axes = reshape_to_unfuse_axes(self.indices, newshape)
        # n.b. these are returned in descending order already
        new = self
        for i in fused_axes:
            new = new.unfuse(i)
        return new

    def reshape(self, newshape):
        """Reshape the block array to ``newshape``, assuming it can be done by
        purly fusing, or unfusing the relevant indices.
        """
        if not isinstance(newshape, tuple):
            newshape = tuple(newshape)
        newshape = find_full_reshape(newshape, self.size)
        if len(newshape) < self.ndim:
            return self._reshape_via_fuse(newshape)
        elif len(newshape) > self.ndim:
            return self._reshape_via_unfuse(newshape)
        elif newshape == self.shape:
            return self.copy()
        else:
            raise ValueError("reshape must be pure fuse or unfuse.")

    def __repr__(self):
        return (
            f"BlockArray(indices={self.indices}, "
            f"charge_total={self._charge_total}, "
            f"num_blocks={self.num_blocks})"
        )


# --------------------------------------------------------------------------- #


def conj(x):
    return x.conj()


def transpose(a, axes=None):
    return a.transpose(axes)


def reshape(a, newshape):
    return a.reshape(newshape)


def tensordot_via_blocks(a, b, left_axes, axes_a, axes_b, right_axes):
    """Perform a tensordot between two block arrays, performing the contraction
    of each pair of aligned blocks separately.
    """
    aligned_blocks = defaultdict(list)

    # iterate over all valid sectors of the new BlockArray
    _tensordot = ar.get_lib_fn(a.backend, "tensordot")
    _stack = ar.get_lib_fn(a.backend, "stack")

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

    stacked_axes = (
        (0,) + tuple(ax + 1 for ax in axes_a),
        (0,) + tuple(ax + 1 for ax in axes_b),
    )

    for sector, (arrays_suba, arrays_subb) in new_blocks.items():
        if len(arrays_suba) == 1:
            # only one aligned block pair, simply tensordot
            new_blocks[sector] = _tensordot(
                arrays_suba[0], arrays_subb[0], axes=(axes_a, axes_b)
            )
        else:
            # multiple aligned blocks: stack and tensordot including new
            # stacked axis, which effectively sums over it
            arrays_suba = _stack(tuple(arrays_suba))
            arrays_subb = _stack(tuple(arrays_subb))
            new_blocks[sector] = _tensordot(
                arrays_suba, arrays_subb, axes=stacked_axes
            )

    new = BlockArray.__new__(BlockArray)
    new._indices = without(a.indices, axes_a) + without(b.indices, axes_b)
    new._charge_total = symmetry(a.charge_total, b.charge_total)
    new._blocks = new_blocks
    return new


def tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes):

    # fuse into matrices or maybe vectors
    af = a.fuse(left_axes, axes_a)
    bf = b.fuse(axes_b, right_axes)

    # handle possible 'vector' cases
    left_axes, axes_a = ((0,), (1,)) if left_axes else ((), (0,))
    axes_b, right_axes = ((0,), (1,)) if right_axes else ((0,), ())

    # tensordot the fused blocks
    cf = tensordot_via_blocks(af, bf, left_axes, axes_a, axes_b, right_axes)

    # unfuse result into (*left_axes, *right_axes)
    return cf.unfuse_all()


def tensordot(a, b, axes=2, mode="auto"):

    # a.check()
    # b.check()

    # parse the axes argument for single integer and also negative indices
    if isinstance(axes, int):
        axes_a = tuple(range(a.ndim - axes, a.ndim))
        axes_b = tuple(range(0, axes))
    else:
        axes_a, axes_b = axes
        axes_a = tuple(x % a.ndim for x in axes_a)
        axes_b = tuple(x % b.ndim for x in axes_b)
        if not len(axes_a) == len(axes_b):
            raise ValueError("Axes must have same length.")

    left_axes = without(range(a.ndim), axes_a)
    right_axes = without(range(b.ndim), axes_b)

    if mode != "blocks":
        _tdot = tensordot_via_fused
    else:
        _tdot = tensordot_via_blocks

    c = _tdot(a, b, left_axes, axes_a, axes_b, right_axes)
    # c.check()
    return c
