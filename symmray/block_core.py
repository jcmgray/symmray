import functools
import itertools
import math
import operator
from collections import defaultdict

import autoray as ar
from autoray.lazy.core import find_full_reshape

from .interface import tensordot
from .symmetries import get_symmetry


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

    __slots__ = ("_chargemap", "_flow", "_subinfo")

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

    def check(self):
        """Check that the index is well-formed, i.e. all sizes are positive."""
        for c, d in self._chargemap.items():
            if d <= 0:
                raise ValueError(
                    f"Size of charge {c} is {d}, must be positive."
                )
            if not isinstance(d, int):
                raise ValueError(f"Size of charge {c} is {d}, must be an int.")

    def matches(self, other):
        """Whether this index matches ``other`` index, namely, whether the
        ``chargemap`` of each matches, their flows are opposite, and also
        whether their subindices match, if they have any. For debugging.

        Parameters
        ----------
        other : BlockIndex
            The other index to compare to.
        """
        return (
            all(
                self.chargemap[k] == other.chargemap[k]
                for k in set(self.chargemap).intersection(other.chargemap)
            )
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
                f"indices={self.indices}, ",
                f"extents={self.extents}",
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


@functools.lru_cache(2**15)
def reshape_to_fuse_axes(shape, newshape):
    """Assuming only fuses need to happen, convert from ``reshape`` form to
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


_blockarray_slots = (
    "_indices",
    "_blocks",
    "_charge_total",
)


class BlockArray:
    """A block sparse array with symmetry constraints.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices of the array.
    charge_total : hashable, optionals
        The total charge of the array, if not given it will be inferred from
        either the first sector or set to the identity charge, if no sectors
        are given.
    blocks : dict[tuple[hashable], array_like]
        A mapping of each 'sector' (tuple of charges) to the data array.
    """

    __slots__ = _blockarray_slots

    def __init__(
        self,
        indices,
        charge_total=None,
        blocks=(),
    ):
        self._indices = tuple(indices)
        self._blocks = dict(blocks)

        if charge_total is None:
            if self._blocks:
                # infer the charge total from any sector
                sector = next(iter(self._blocks))
                self._charge_total = self.symmetry.combine(*sector)
            else:
                # default to the identity charge
                self._charge_total = self.symmetry.combine()
        else:
            self._charge_total = charge_total

    def copy(self):
        """Copy this block array."""
        new = self.__new__(self.__class__)
        new._indices = self._indices
        new._charge_total = self._charge_total
        new._blocks = self._blocks.copy()
        return new

    def copy_with(self, indices=None, blocks=None):
        new = self.__new__(self.__class__)
        new._indices = self._indices if indices is None else indices
        new._charge_total = self._charge_total
        new._blocks = self._blocks.copy() if blocks is None else blocks
        return new

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

    @property
    def sectors(self):
        return tuple(self._blocks.keys())

    def gen_signed_sectors(self):
        flows = self.flows
        for sector in self.blocks:
            yield tuple(
                self.symmetry.negate(c, f) for c, f in zip(sector, flows)
            )

    def get_sparsity(self):
        """Return the sparsity of the array, i.e. the number of blocks
        divided by the number of possible blocks.
        """
        num_possible_blocks = sum(
            self.symmetry.combine(*sector) == self.charge_total
            for sector in self.gen_signed_sectors()
        )
        return self.num_blocks / num_possible_blocks

    def is_valid_sector(self, sector):
        """Check if a sector is valid for the block array, i.e., whether the
        total symmetry charge is satisfied.
        """
        signed_sector = (
            self.symmetry.negate(c, i.flow)
            for c, i in zip(sector, self._indices)
        )
        block_charge = self.symmetry.combine(*signed_sector)
        return block_charge == self.charge_total

    def gen_valid_sectors(self):
        """Generate all valid sectors for the block array."""
        return filter(self.is_valid_sector, itertools.product(*self.charges))

    def get_block_shape(self, sector):
        """Get the shape of the block corresponding to a given sector."""
        return tuple(ix.size_of(c) for ix, c in zip(self._indices, sector))

    def check(self):
        """Check that all the block sizes and charges are consistent."""
        for idx in self.indices:
            idx.check()
        for sector, array in self.blocks.items():
            assert self.is_valid_sector(sector)
            assert all(
                di == dj
                for di, dj in zip(
                    ar.shape(array), self.get_block_shape(sector)
                )
            )

    def allclose(self, other, **allclose_opts):
        """Test whether this ``BlockArray`` is close to another, that is,
        has all the same sectors, and the corresponding arrays are close.

        Parameters
        ----------
        other : BlockArray
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Returns
        -------
        bool
        """
        _allclose = ar.get_lib_fn(self.backend, "allclose")
        other_blocks = other.blocks.copy()
        for sector, array in self.blocks.items():
            if sector not in other_blocks:
                return False
            other_array = other_blocks.pop(sector)
            if not _allclose(array, other_array, **allclose_opts):
                return False
        if other_blocks:
            return False
        return True

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
        self = cls.__new__(cls)

        self._indices = tuple(indices)

        if charge_total is None:
            self._charge_total = cls.symmetry.combine()
        else:
            self._charge_total = charge_total

        self._blocks = {
            sector: fill_fn(self.get_block_shape(sector))
            for sector in self.gen_valid_sectors()
        }
        return self

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
            self._charge_total = cls.symmetry.combine()
        else:
            self._charge_total = charge_total
        self._blocks = dict(blocks)

        ndim = len(next(iter(blocks.keys())))
        charge_size_maps = [{} for _ in range(ndim)]

        for sector, array in self.blocks.items():
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
        # XXX: warn if invalid blocks are non-zero?

        if charge_total is None:
            charge_total = cls.symmetry.combine()

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
                for charge, indices in charge_groups[j].items():
                    # for each charge, select all the indices along axis j
                    # that belong to it, then recurse further
                    selector = all_sliced.copy()
                    selector[j] = indices
                    subarray = ary[tuple(selector)]
                    _recurse(subarray, j + 1, sector + (charge,))
            else:
                # we have reached a fully specified block
                signed_sector = tuple(
                    cls.symmetry.negate(charge, flow)
                    for charge, flow in zip(sector, flows)
                )
                if cls.symmetry.combine(*signed_sector) == charge_total:
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

    def get_params(self):
        """Get the parameters of this block array as a pytree (dict).

        Returns
        -------
        dict[tuple, array_like]
        """
        return self.blocks.copy()

    def set_params(self, params):
        """Set the parameters of this block array from a pytree (dict).

        Parameters
        ----------
        params : dict[tuple, array_like]
        """
        self.blocks.update(params)

    def apply_to_arrays(self, fn):
        """Apply the ``fn`` inplace to the array of every block."""
        for sector, array in self._blocks.items():
            self._blocks[sector] = fn(array)

    def item(self):
        """Convert the block array to a scalar if it is a scalar block array."""
        (array,) = self.blocks.values()
        return array.item()

    def to_dense(self):
        """Convert this block array to a dense array."""
        backend = self.backend
        _ex_array = self._get_any_array()
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

    def __float__(self):
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def __mul__(self, other):
        if isinstance(other, BlockArray):
            raise NotImplementedError("Multiplication of block arrays.")

        new = self.copy()
        new.apply_to_arrays(lambda x: x * other)
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, BlockArray):
            raise NotImplementedError("Division of block arrays.")

        new = self.copy()
        new.apply_to_arrays(lambda x: x / other)
        return new

    def __rtruediv__(self, other):
        if isinstance(other, BlockArray):
            raise NotImplementedError("Division of block arrays.")

        new = self.copy()
        new.apply_to_arrays(lambda x: other / x)
        return new

    def conj(self, inplace=False):
        """Return the complex conjugate of this block array, including the
        indices."""
        new = self if inplace else self.copy()
        _conj = ar.get_lib_fn(new.backend, "conj")
        new.apply_to_arrays(_conj)
        new._indices = tuple(ix.conj() for ix in self._indices)
        return new

    def transpose(self, axes=None, inplace=False):
        """Transpose the block array."""
        new = self if inplace else self.copy()

        _transpose = ar.get_lib_fn(new.backend, "transpose")

        if axes is None:
            # reverse the axes
            axes = tuple(range(new.ndim - 1, -1, -1))

        new._indices = permuted(new._indices, axes)
        new._blocks = {
            permuted(sector, axes): _transpose(array, axes)
            for sector, array in new.blocks.items()
        }
        return new

    def fuse(self, *axes_groups, inplace=False):
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
        # XXX: error or warn about empty groups?

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
        ax2group = {}
        # what flow each group has
        group_flows = []
        old_indices = self._indices
        for g, gaxes in enumerate(axes_groups):
            # take the flow of the group to match the first axis
            group_flows.append(old_indices[gaxes[0]].flow)
            for ax in gaxes:
                ax2group[ax] = g
        # assign `None` to ungrouped axes
        for i in range(self.ndim):
            ax2group.setdefault(i, None)

        # the permutation will be the same for every block: precalculate
        # n.b. all new groups will be inserted at the *first fused axis*:
        position = min((min(gaxes) for gaxes in axes_groups))
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
        num_groups = len(axes_groups)
        new_axes = {ax: ax for ax in axes_before}
        for g, gaxes in enumerate(axes_groups):
            for ax in gaxes:
                new_axes[ax] = position + g
        for g, ax in enumerate(axes_after):
            new_axes[ax] = position + num_groups + g
        new_ndim = len(axes_before) + num_groups + len(axes_after)

        # then we process the blocks one by one into new fused sectors
        new_blocks = {}
        subinfos = [{} for _ in range(num_groups)]

        for sector, array in self.blocks.items():
            # keep track of a perm+shape in order to fuse the actual array
            new_shape = [1] * new_ndim
            # the key of the new fused block to add this block to
            new_sector = [self.symmetry.combine()] * new_ndim
            # only the parts of the sector that will be fused
            subsectors = [[] for g in range(num_groups)]

            # n.b. we have to use `perm` here, not `enumerate(sector)`, so
            # that subsectors are built in matching order for tensordot e.g.
            for ax in perm:
                # the size of charge `c` along axis `ax`
                c = sector[ax]
                ix = old_indices[ax]
                d = ix.size_of(c)

                if not isinstance(d, int):
                    raise ValueError(
                        f"Expected integer size, got {d} of type {type(d)}."
                    )

                # which group is this axis in, if any, and where is it going
                g = ax2group[ax]
                new_ax = new_axes[ax]
                if g is None:
                    # not fusing, new value is just copied
                    new_sector[new_ax] = c
                    new_shape[new_ax] = d
                else:
                    # fusing: need to accumulate
                    new_shape[new_ax] *= d
                    subsectors[g].append(c)
                    # need to match current flow to group flow
                    flowed_c = self.symmetry.negate(
                        c, group_flows[g] ^ ix.flow
                    )
                    new_sector[new_ax] = self.symmetry.combine(
                        new_sector[new_ax], flowed_c
                    )

            # make hashable
            new_sector = tuple(new_sector)
            new_shape = tuple(new_shape)

            # fuse (via transpose+reshape) the actual array, to concat later
            new_array = _transpose(array, perm)
            new_array = _reshape(new_array, new_shape)

            # group the subblock into the correct new fused block
            subsectors = tuple(map(tuple, subsectors))
            new_blocks.setdefault(new_sector, {})[subsectors] = new_array

            # keep track of the new blocksize of each fused index, for unfusing
            # and also missing blocks
            for g, subsector in enumerate(subsectors):
                new_charge = new_sector[position + g]
                new_size = new_shape[position + g]
                subinfos[g][subsector] = (new_charge, new_size)

        # sort and accumulate subsectors into their new charges for each group
        chargemaps = []
        extents = []
        for g in range(num_groups):
            chargemap = {}
            extent = {}
            for subsector, (new_charge, new_size) in sorted(
                subinfos[g].items()
            ):
                if new_charge not in chargemap:
                    chargemap[new_charge] = new_size
                    extent[new_charge] = [(subsector, new_size)]
                else:
                    chargemap[new_charge] += new_size
                    extent[new_charge].append((subsector, new_size))
            chargemaps.append(chargemap)
            extents.append(extent)

        new_indices = (
            *(old_indices[ax] for ax in axes_before),
            # the new fused indices
            *(
                BlockIndex(
                    chargemap=chargemaps[g],
                    flow=group_flows[g],
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

        def _recurse_sorted_concat(new_sector, g=0, subkey=()):
            new_charge = new_sector[position + g]
            new_subkeys = [
                (*subkey, subsector) for subsector, _ in extents[g][new_charge]
            ]

            if g == num_groups - 1:
                # final group (/level of recursion), get actual arrays
                arrays = []
                for new_subkey in new_subkeys:
                    try:
                        array = new_blocks[new_sector][new_subkey]
                    except KeyError:
                        # subsector is missing - need to create zeros
                        shape_before = (
                            old_indices[ax].size_of(new_sector[new_axes[ax]])
                            for ax in axes_before
                        )
                        shape_new = (
                            subinfos[g][ss][1]
                            for g, ss in enumerate(new_subkey)
                        )
                        shape_after = (
                            old_indices[ax].size_of(new_sector[new_axes[ax]])
                            for ax in axes_after
                        )
                        new_shape = (*shape_before, *shape_new, *shape_after)
                        array = ar.do("zeros", shape=new_shape, like=backend)
                    arrays.append(array)
            else:
                # recurse to next group
                arrays = (
                    _recurse_sorted_concat(new_sector, g + 1, new_subkey)
                    for new_subkey in new_subkeys
                )

            return _concatenate(tuple(arrays), axis=position + g)

        new = self if inplace else self.copy()
        new._indices = new_indices
        new._blocks = {
            new_sector: _recurse_sorted_concat(new_sector)
            for new_sector in new_blocks
        }

        return new

    def unfuse(self, axis, inplace=False):
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.
        """
        backend = self.backend
        _split = ar.get_lib_fn(backend, "split")
        _reshape = ar.get_lib_fn(backend, "reshape")

        # get required information from the fused index
        subinfo = self.indices[axis].subinfo

        # info for how to split/slice the linear index into sub charges
        subindex_slices = {
            c: accum_for_split(x[1] for x in charge_extent)
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

        new = self if inplace else self.copy()
        new._indices = replace_with_seq(self.indices, axis, subinfo.indices)
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
                f"{'+'.join(map(str, self.sizes[i]))} : "
                f"{'+' if self.flows[i] else '-'}"
                f"[{','.join(map(str, self.charges[i]))}]),\n"
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

    def norm(self):
        """Get the frobenius norm of the block array."""
        backend = self.backend
        _sum = ar.get_lib_fn(backend, "sum")
        _abs = ar.get_lib_fn(backend, "abs")
        return (
            functools.reduce(
                operator.add,
                (_sum(_abs(x) ** 2) for x in self.blocks.values()),
            )
            ** 0.5
        )

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                (
                    f"indices={self.indices}, "
                    if self.indices
                    else f"{self._get_any_array()}, "
                ),
                f"charge_total={self._charge_total}, ",
                f"num_blocks={self.num_blocks})",
            ]
        )


# --------------------------------------------------------------------------- #


def tensordot_blockwise(a, b, left_axes, axes_a, axes_b, right_axes):
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

    # XXX: this has better performance, but only works w/ shape-matching blocks
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

    for sector, (arrays_suba, arrays_subb) in new_blocks.items():
        new_blocks[sector] = functools.reduce(
            operator.add,
            (
                _tensordot(a, b, axes=(axes_a, axes_b))
                for a, b in zip(arrays_suba, arrays_subb)
            ),
        )

    new = a.__new__(a.__class__)
    new._indices = without(a.indices, axes_a) + without(b.indices, axes_b)
    new._charge_total = new.symmetry.combine(a.charge_total, b.charge_total)
    new._blocks = new_blocks
    return new


def drop_misaligned_sectors(a, b, axes_a, axes_b):
    """Eagerly drop misaligned sectors of ``a`` and ``b`` so that they can be
    contracted via fusing.

    Parameters
    ----------
    a, b : BlockArray
        The arrays to be contracted.
    axes_a, axes_b : tuple[int]
        The axes that will be contracted, defined like in `tensordot`.

    Returns
    -------
    a, b : BlockArray
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

    # filter out sectors of a and b that are not aligned
    new_blocks_a = {
        sector: array
        for sector, array in a.blocks.items()
        if sub_sectors_a[sector] in allowed_subsectors
    }
    new_blocks_b = {
        sector: array
        for sector, array in b.blocks.items()
        if sub_sectors_b[sector] in allowed_subsectors
    }

    return a.copy_with(blocks=new_blocks_a), b.copy_with(blocks=new_blocks_b)


def tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes):
    """Perform a tensordot between two block arrays, by first fusing both into
    matrices and unfusing afterwards.

    Parameters
    ----------
    a, b : BlockArray
        The arrays to be contracted.
    left_axes : tuple[int]
        The axes of ``a`` that will not be contracted.
    axes_a, axes_b : tuple[int]
        The axes that will be contracted, defined like in `tensordot`.
    right_axes : tuple[int]
        The axes of ``b`` that will not be contracted.
    """
    a, b = drop_misaligned_sectors(a, b, axes_a, axes_b)

    # fuse into matrices or maybe vectors
    af = a.fuse(left_axes, axes_a)
    bf = b.fuse(axes_b, right_axes)

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
    cf = tensordot_blockwise(af, bf, left_axes, axes_a, axes_b, right_axes)

    # unfuse result into (*left_axes, *right_axes)
    return cf.unfuse_all()


@tensordot.register(BlockArray)
def tensordot_blocked(a, b, axes=2, mode="auto"):
    """ """
    ndim_a = a.ndim
    ndim_b = b.ndim

    # parse the axes argument for single integer and also negative indices
    if isinstance(axes, int):
        axes_a = tuple(range(ndim_a - axes, ndim_a))
        axes_b = tuple(range(0, axes))
    else:
        axes_a, axes_b = axes
        axes_a = tuple(x % ndim_a for x in axes_a)
        axes_b = tuple(x % ndim_b for x in axes_b)
        if not len(axes_a) == len(axes_b):
            raise ValueError("Axes must have same length.")

    for ax_a, ax_b in zip(axes_a, axes_b):
        if not a.indices[ax_a].matches(b.indices[ax_b]):
            raise ValueError("Axes are not matching.")

    left_axes = without(range(ndim_a), axes_a)
    right_axes = without(range(ndim_b), axes_b)

    if mode == "auto":
        if len(axes_a) == 0:
            mode = "blockwise"
        else:
            mode = "fused"

    if mode == "fused":
        _tdot = tensordot_via_fused
    elif mode == "blockwise":
        _tdot = tensordot_blockwise
    else:
        raise ValueError(f"Unknown tensordot mode: {mode}.")

    c = _tdot(a, b, left_axes, axes_a, axes_b, right_axes)

    c.check()
    cf = tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes)
    cb = tensordot_blockwise(a, b, left_axes, axes_a, axes_b, right_axes)
    if not cf.allclose(cb):
        breakpoint()
        tensordot_via_fused(a, b, left_axes, axes_a, axes_b, right_axes)
        raise ValueError("Blocks do not match.")

    return c


# --------------------------------------------------------------------------- #


class Z2Array(BlockArray):
    """A block array with Z2 symmetry."""

    __slots__ = _blockarray_slots

    symmetry = get_symmetry("Z2")


class U1Array(BlockArray):
    """A block array with U1 symmetry."""

    __slots__ = _blockarray_slots

    symmetry = get_symmetry("U1")