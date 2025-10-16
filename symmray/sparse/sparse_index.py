"""Index objects for abelian arrays with block sparse backend."""

import numbers
import warnings

from ..index_common import Index, SubInfo
from ..utils import hasher


class BlockIndex(Index):
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
    linearmap : Sequence[tuple[hashable, int], ...], optional
        If provided, a sequence of (charge, offset) pairs for a linear ordering
        of this index. This can be supplied instead of ``chargemap``, in which
        case the ``chargemap`` is built from this. If not provided, it is built
        on demand from ``chargemap``, assuming blocks of sorted charges.
    """

    __slots__ = (
        "_chargemap",
        "_dual",
        "_subinfo",
        "_hashkey",
        "_linearmap",
    )

    def __init__(
        self,
        chargemap=None,
        dual=False,
        subinfo=None,
        linearmap=None,
    ):
        if chargemap is None:
            if linearmap is None:
                raise ValueError("Must provide either chargemap or linearmap.")
            # build chargemap from linearmap
            chargemap = {}
            for c, _ in linearmap:
                chargemap[c] = chargemap.get(c, 0) + 1

        # ensure always sorted
        if not isinstance(chargemap, dict):
            self._chargemap = dict(sorted(chargemap))
        else:
            self._chargemap = dict(sorted(chargemap.items()))
        self._dual = bool(dual)
        self._subinfo = subinfo
        self._hashkey = None
        self._linearmap = linearmap

    def copy_with(self, **kwargs):
        """A copy of this index with some attributes replaced. Note that checks
        are not performed on the new propoerties, this is intended for internal
        use.
        """
        new = self.__new__(self.__class__)
        keep_linearmap = True

        if "chargemap" in kwargs:
            chargemap = kwargs.pop("chargemap")
            new._chargemap = (
                dict(sorted(chargemap.items()))
                if isinstance(chargemap, dict)
                else dict(sorted(chargemap))
            )
            keep_linearmap = False
        else:
            new._chargemap = self._chargemap.copy()

        if "dual" in kwargs:
            new._dual = bool(kwargs.pop("dual"))
        else:
            new._dual = self._dual

        # need to pop from kwargs to handle 'not-set' vs 'set-to-None'
        if "subinfo" in kwargs:
            new._subinfo = kwargs.pop("subinfo")
        else:
            new._subinfo = self._subinfo

        # only keep linearmap if chargemap unchanged
        if keep_linearmap:
            new._linearmap = self._linearmap
        else:
            new._linearmap = None

        # always recompute this
        new._hashkey = None

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {kwargs}")

        return new

    def to_flat(self):
        """Convert this index to a flat index."""
        from ..flat.flat_index import FlatIndex

        charges = sorted(self.charges)
        num_charges = len(charges)
        charge_size = max(self.sizes)

        if (num_charges != 1) and (charges != list(range(num_charges))):
            raise ValueError(
                "FlatIndex requires a single charge or 0...N-1 as charges."
            )

        if self.subinfo is not None:
            warnings.warn(
                "Converting subinfo to flat not supported yet, dropping it."
            )

        return FlatIndex(
            num_charges=num_charges,
            charge_size=charge_size,
            dual=self.dual,
            linearmap=self._linearmap,
        )

    @property
    def chargemap(self):
        """A mapping from charge to size."""
        return self._chargemap

    @property
    def linearmap(self):
        """A sequence of (charge, offset) pairs for each element of this index,
        in linear order. This is built on demand if not provided at creation.
        """
        if self._linearmap is None:
            # build a default linearmap in sorted blocks
            linearmap = []
            for c, d in sorted(self._chargemap.items()):
                linearmap.extend((c, i) for i in range(d))
            self._linearmap = tuple(linearmap)
        return self._linearmap

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

    def select_charge(self, charge, subselect=None) -> "BlockIndex":
        """Drop all but the specified charge from this index.

        Parameters
        ----------
        charge : hashable
            The charge to keep.
        subselect : slice or array_like, optional
            If provided, a range of indices within the selected charge block
            to keep. If not provided, the entire block is kept.

        Returns
        -------
        BlockIndex
        """
        drop = set(self._chargemap)
        drop.remove(charge)
        new = self.drop_charges(drop)
        if subselect is None:
            return new

        # need to update charge size as well
        new_chargemap = new._chargemap.copy()
        current_size = new_chargemap[charge]

        if isinstance(subselect, slice):
            start, stop, step = subselect.indices(current_size)
            size = len(range(start, stop, step))
            new_chargemap[charge] = size
        elif hasattr(subselect, "size"):  # numpy array or similar
            new_chargemap[charge] = subselect.size
        else:
            new_chargemap[charge] = len(subselect)

        # XXX: could we update subinfo rather than remove?
        return new.copy_with(chargemap=new_chargemap, subinfo=None)

    def size_of(self, c):
        """The size of the block with charge ``c``."""
        return self._chargemap[c]

    def linear_to_charge_and_offset(self, i):
        """Given a linear index ``i`` into this index (as if it were a dense
        array), return the corresponding charge and offset within that charge
        block.

        Parameters
        ----------
        i : int
            The linear index into this index.

        Returns
        -------
        charge : hashable
            The charge corresponding to the linear index.
        offset : int
            The offset within the charge block corresponding to the linear
            index.
        """
        return self.linearmap[i]

    def check(self):
        """Check that the index is well-formed, i.e. all sizes are positive."""
        for c, d in self._chargemap.items():
            if d <= 0:
                raise ValueError(
                    f"Size of charge {c} is {d}, must be positive."
                )
            if not isinstance(d, numbers.Integral):
                raise ValueError(f"Size of charge {c} is {d}, must be an int.")

        assert sorted(self._chargemap) == list(self._chargemap)

        if self.subinfo:
            assert self.size_total == sum(
                d
                for extent in self.subinfo.extents.values()
                for d in extent.values()
            )

        if self._linearmap is not None:
            if len(self._linearmap) != self.size_total:
                raise ValueError(
                    f"Index map length {len(self._linearmap)} does not "
                    f"match index size {self.size_total}."
                )
            seen = set()
            for c, i in self._linearmap:
                if c not in self._chargemap:
                    raise ValueError(
                        f"Index map charge {c} not in chargemap "
                        f"{self._chargemap}."
                    )
                if i < 0 or i >= self._chargemap[c]:
                    raise ValueError(
                        f"Index map offset {i} out of bounds for charge {c} "
                        f"with size {self._chargemap[c]}."
                    )
                if (c, i) in seen:
                    raise ValueError(
                        f"Index map has duplicate entry {(c, i)}."
                    )
                seen.add((c, i))

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


class SubIndexInfo(SubInfo):
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
