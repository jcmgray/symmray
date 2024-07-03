"""Blocks arrays with an abelian symmetry constraint."""

import functools
import itertools
import math
import operator
from collections import defaultdict

import autoray as ar
from autoray.lazy.core import find_full_reshape

from .block_core import BlockBase
from .interface import tensordot
from .symmetries import get_symmetry
from .utils import DEBUG


class BlockIndex:
    """An index of a block sparse, abelian symmetric tensor.

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

    __slots__ = ("_chargemap", "_dual", "_subinfo")

    def __init__(self, chargemap, dual=False, subinfo=None):
        # ensure always sorted
        if not isinstance(chargemap, dict):
            self._chargemap = dict(sorted(chargemap))
        else:
            self._chargemap = dict(sorted(chargemap.items()))
        self._dual = bool(dual)
        self._subinfo = subinfo

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
        new._dual = self._dual
        new._subinfo = self._subinfo
        return new

    def copy_with(self, chargemap=None, dual=None, subinfo=None):
        """A copy of this index with some attributes replaced."""
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
        return new

    def conj(self):
        """A copy of this index with the dualness reversed."""
        new = self.__new__(self.__class__)
        new._chargemap = self._chargemap.copy()
        new._dual = not self._dual
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

        assert sorted(self._chargemap) == list(self._chargemap)

        if self.subinfo:
            assert self.size_total == sum(
                e[1]
                for extent in self.subinfo.extents.values()
                for e in extent
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
            dicts_dont_conflict(self.chargemap, other.chargemap)
            and (self.dual ^ other.dual)
            and (
                (self.subinfo is other.subinfo is None)
                or (self.subinfo.matches(other.subinfo))
            )
        )

    def __hash__(self):
        return hash(
            (
                tuple(self._chargemap.items()),
                self._dual,
                self._subinfo,
            )
        )

    def __str__(self):
        lines = [
            f"({self.size_total} = "
            f"{'+'.join(map(str, self.chargemap.values()))} "
            f": {'-' if self.dual else '+'}"
            f"[{','.join(map(str, self.chargemap.keys()))}])"
        ]

        if self.subinfo:
            for charge, extent in sorted(self.subinfo.extents.items()):
                subcharges, subsizes = zip(*extent)
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
                f"chargemap={self.chargemap}, dual={self.dual}",
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
    extents : dict[int, dict[int, int]]
        A mapping of each charge of the fused index to a mapping of each
        subsector (combination of sub charges) to the size of that subsector.
    """

    __slots__ = ("extents", "indices")

    def __init__(self, indices, extents):
        self.indices = indices
        self.extents = extents

    def conj(self):
        """A copy of this subindex information with the relevant dualnesses
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
        return all(
            i.matches(j) for i, j in zip(self.indices, other.indices)
        ) and dicts_dont_conflict(self.extents, other.extents)

    def __hash__(self):
        raise NotImplementedError

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


@functools.lru_cache(2**14)
def calc_fuse_info(axes_groups, duals):
    ndim = len(duals)

    # which group does each axis appear in, if any
    ax2group = {}
    # whether each group is overall dual
    group_duals = []
    for g, gaxes in enumerate(axes_groups):
        # take the dual-ness of the group to match the first axis
        group_duals.append(duals[gaxes[0]])
        for ax in gaxes:
            ax2group[ax] = g
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
    for g, ax in enumerate(axes_after):
        new_axes[ax] = position + num_groups + g
    new_ndim = len(axes_before) + num_groups + len(axes_after)

    return (
        num_groups,
        new_ndim,
        perm,
        position,
        axes_before,
        axes_after,
        ax2group,
        group_duals,
        new_axes,
    )


_AbelianArray_slots = (
    "_indices",
    "_blocks",
    "_charge",
)


class AbelianArray(BlockBase):
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
    """

    __slots__ = _AbelianArray_slots
    fermionic = False

    def __init__(
        self,
        indices,
        charge=None,
        blocks=(),
    ):
        self._indices = tuple(indices)
        self._blocks = dict(blocks)

        if charge is None:
            if self._blocks:
                # infer the charge total from any sector
                sector = next(iter(self._blocks))
                self._charge = self.symmetry.combine(*sector)
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
        return new

    def copy_with(self, indices=None, charge=None, blocks=None):
        new = self.__new__(self.__class__)
        new._indices = self._indices if indices is None else indices
        new._charge = self._charge if charge is None else charge
        new._blocks = self._blocks.copy() if blocks is None else blocks

        if DEBUG:
            new.check()

        return new

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

    def sync_charges(self):
        current_charges = [set(index.chargemap) for index in self.indices]
        for sector in self.blocks:
            for i, c in enumerate(sector):
                current_charges[i].discard(c)

        for i, charges in enumerate(current_charges):
            for c in charges:
                self._indices[i].chargemap.pop(c)

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
        return filter(self.is_valid_sector, itertools.product(*self.charges))

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

        # actual_charges = [set() for _ in range(self.ndim)]

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

        # XXX: check no empty charges?
        #     for i, c in enumerate(sector):
        #         actual_charges[i].add(c)

        # for actual, index in zip(actual_charges, self.indices):
        #     expected = set(index.chargemap)
        #     if actual != expected:
        #         raise ValueError(
        #             f"Charges for index {index} are inconsistent: "
        #             f"{actual} != {expected}."
        #         )

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
        from .block_core import BlockVector

        if isinstance(other, BlockVector):
            (ax,) = args
            for sector, array in self.blocks.items():
                charge = sector[ax]
                v_block = other.blocks[charge]
                assert ar.shape(array)[ax] == ar.size(v_block)

        else:
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
        charge=None,
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
        """
        self = cls.__new__(cls)

        self._indices = tuple(indices)

        if charge is None:
            self._charge = cls.symmetry.combine()
        else:
            self._charge = charge

        self._blocks = {
            sector: fill_fn(self.get_block_shape(sector))
            for sector in self.gen_valid_sectors()
        }

        # self.sync_charges()

        return self

    @classmethod
    def random(cls, indices, charge=None, seed=None, dist="normal"):
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
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        rand_fn = getattr(rng, dist)

        def fill_fn(shape):
            return rand_fn(size=shape)

        return cls.from_fill_fn(fill_fn, indices, charge)

    @classmethod
    def from_blocks(cls, blocks, duals, charge=None):
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
            taken as the identity / zero element.

        Returns
        -------
        AbelianArray
        """
        self = cls.__new__(cls)
        if charge is None:
            self._charge = cls.symmetry.combine()
        else:
            self._charge = charge
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

        duals = tuple(duals)
        if len(duals) != ndim:
            raise ValueError(f"Expected {ndim} duals, got {len(duals)}.")

        self._indices = tuple(
            BlockIndex(x, f) for x, f in zip(charge_size_maps, duals)
        )

        return self

    @classmethod
    def from_dense(cls, array, index_maps, duals, charge=None):
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
        """
        # XXX: warn if invalid blocks are non-zero?

        if charge is None:
            charge = cls.symmetry.combine()

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
                    cls.symmetry.sign(c, dual)
                    for c, dual in zip(sector, duals)
                )
                if cls.symmetry.combine(*signed_sector) == charge:
                    # ... but only add valid ones:
                    blocks[sector] = ary

        # generate the blocks
        _recurse(array)

        # generate the indices -> the charge_map is simply the group size
        indices = [
            BlockIndex({c: len(g) for c, g in charge_group.items()}, dual=dual)
            for charge_group, dual in zip(charge_groups, duals)
        ]

        # create the block array!
        return cls(blocks=blocks, indices=indices, charge=charge)

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
        new._indices = tuple(ix.conj() for ix in self._indices)
        new._charge = self.symmetry.sign(self._charge)

        if DEBUG:
            new.check()

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
        # handle empty groups and ensure hashable
        # XXX: error or warn about empty groups?
        axes_groups = tuple(tuple(group) for group in axes_groups if group)
        if not axes_groups:
            # ... and no groups -> nothing to do
            return self.copy()

        backend = self.backend
        _transpose = ar.get_lib_fn(backend, "transpose")
        _reshape = ar.get_lib_fn(backend, "reshape")
        _concatenate = ar.get_lib_fn(backend, "concatenate")

        (
            num_groups,
            new_ndim,
            perm,
            position,
            axes_before,
            axes_after,
            ax2group,
            group_duals,
            new_axes,
        ) = calc_fuse_info(axes_groups, self.duals)

        # then we process the blocks one by one into new fused sectors
        new_blocks = {}
        subinfos = [{} for _ in range(num_groups)]
        old_indices = self._indices

        for sector, array in self.blocks.items():
            # keep track of a perm+shape in order to fuse the actual array
            new_shape = [1] * new_ndim
            # the key of the new fused block to add this block to
            new_sector = [self.symmetry.combine()] * new_ndim
            # only the parts of the sector that will be fused
            subsectors = [[] for _ in range(num_groups)]

            # n.b. we have to use `perm` here, not `enumerate(sector)`, so
            # that subsectors are built in matching order for tensordot e.g.
            for ax in perm:
                # the size of charge `c` along axis `ax`
                c = sector[ax]
                ix = old_indices[ax]
                d = ix.size_of(c)

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
                    # need to match current dualness to group dualness
                    signed_c = self.symmetry.sign(c, group_duals[g] ^ ix.dual)
                    new_sector[new_ax] = self.symmetry.combine(
                        new_sector[new_ax], signed_c
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

        new_blocks = {
            new_sector: _recurse_sorted_concat(new_sector)
            for new_sector in new_blocks
        }

        if inplace:
            self._indices = new_indices
            self._blocks = new_blocks
            return self
        else:
            return self.copy_with(
                indices=new_indices,
                blocks=new_blocks,
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

        new_indices = replace_with_seq(self.indices, axis, subinfo.indices)
        new_blocks = new_blocks

        if inplace:
            self._indices = new_indices
            self._blocks = new_blocks
            return self
        else:
            return self.copy_with(indices=new_indices, blocks=new_blocks)

    def unfuse_all(self, inplace=False):
        """Unfuse all indices that carry subindex information, likely from a
        fusing operation.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        new = self if inplace else self.copy()
        for ax in reversed(range(self.ndim)):
            if new.indices[ax].subinfo is not None:
                new.unfuse(ax, inplace=True)
        return new

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

    def __matmul__(self, other):
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D arrays.")

        # block diagonal -> shortcut to tensordot
        return _tensordot_blockwise(
            self,
            other,
            left_axes=(0,),
            axes_a=(1,),
            axes_b=(0,),
            right_axes=(1,),
        )

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

    def multiply_diagonal(self, v, axis, inplace=False):
        """Multiply this block array by a vector as if contracting a diagonal
        matrix along the given axis.
        """
        x = self if inplace else self.copy()

        _reshape = ar.get_lib_fn(v.backend, "reshape")
        new_shape = tuple(-1 if i == axis else 1 for i in range(x.ndim))

        for sector in tuple(x.blocks):
            charge = sector[axis]
            v_sector = v.blocks.get(charge, None)
            if v_sector is not None:
                # use broadcasting to perform "ab...X...c,X-> ab...X...c"
                x.blocks[sector] = x.blocks[sector] * _reshape(
                    v_sector, new_shape
                )
            else:
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
        pattern = "".join("-" if f else "+" for f in self.duals)
        return "".join(
            [
                f"{self.__class__.__name__}(",
                (
                    f"shape~{self.shape}:" f"[{pattern}]"
                    if self.indices
                    else f"{self.get_any_array()}"
                ),
                f", charge={self._charge}",
                f", num_blocks={self.num_blocks})",
            ]
        )


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
    new._charge = new.symmetry.combine(a.charge, b.charge)
    new._blocks = new_blocks
    return new


def drop_misaligned_sectors(a, b, axes_a, axes_b):
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
    af = AbelianArray.fuse(a, left_axes, axes_a)
    bf = AbelianArray.fuse(b, axes_b, right_axes)

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
        if cf.indices[ax].subinfo is not None:
            AbelianArray.unfuse(cf, ax, inplace=True)

    return cf


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
    if DEBUG:
        a.check()
        b.check()

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

    if DEBUG:
        a.check_with(b, axes_a, axes_b)

    left_axes = without(range(ndim_a), axes_a)
    right_axes = without(range(ndim_b), axes_b)

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

    __slots__ = _AbelianArray_slots

    symmetry = get_symmetry("Z2")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.fermion_symmetry import Z2
        from pyblock3.algebra.core import SparseTensor, SubTensor

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


class U1Array(AbelianArray):
    """A block array with U1 symmetry."""

    __slots__ = _AbelianArray_slots

    symmetry = get_symmetry("U1")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.fermion_symmetry import U1
        from pyblock3.algebra.core import SparseTensor, SubTensor

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


class Z2Z2Array(AbelianArray):
    """A block array with Z2 x Z2 symmetry."""

    __slots__ = _AbelianArray_slots

    symmetry = get_symmetry("Z2Z2")


class U1U1Array(AbelianArray):
    """A block array with U1 x U1 symmetry."""

    __slots__ = _AbelianArray_slots

    symmetry = get_symmetry("U1U1")
