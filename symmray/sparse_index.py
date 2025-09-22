"""Index objects for abelian arrays with block sparse backend."""

from .index_common import Index, SubInfo
from .utils import hasher


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

    def copy_with(self, **kwargs):
        """A copy of this index with some attributes replaced. Note that checks
        are not performed on the new propoerties, this is intended for internal
        use.
        """
        new = self.__new__(self.__class__)

        if "chargemap" in kwargs:
            chargemap = kwargs.pop("chargemap")
            new._chargemap = (
                dict(sorted(chargemap.items()))
                if isinstance(chargemap, dict)
                else dict(sorted(chargemap))
            )
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

        # always recompute this
        new._hashkey = None

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {kwargs}")

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
