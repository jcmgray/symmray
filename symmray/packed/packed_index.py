"""Index objects for abelian arrays with packed backend."""

from ..index_common import Index, SubInfo


def _as_int_array(x):
    import numpy as np

    return np.ascontiguousarray(x, dtype=np.int64)


class PackedIndex(Index):
    """Index of a packed abelian array. Charges are kept sorted. Arrays refer
    to each charge by its position, its 'charge id'. Treat as immutable.

    Parameters
    ----------
    charges : Sequence[hashable]
        The charges of this index. Must be sorted.
    sizes : Sequence[int]
        The size of each charge sector.
    dual : bool, optional
        Whether the index is 'dual' or not, see `BlockIndex`.
    subinfo : PackedSubIndexInfo, optional
        Information about the subindices of this index, if it was formed from
        fusing. Needed to unfuse.
    linearmap : Sequence[tuple[hashable, int]], optional
        An optional linear ordering of this index, see `BlockIndex`.
    """

    __slots__ = (
        "_cache",
        "_charges",
        "_dual",
        "_hashkey",
        "_linearmap",
        "_sizes",
        "_subinfo",
    )

    def __init__(
        self,
        charges,
        sizes,
        dual=False,
        subinfo=None,
        linearmap=None,
    ):
        self._charges = tuple(charges)
        self._sizes = _as_int_array(sizes)
        self._dual = bool(dual)
        self._subinfo = subinfo
        self._linearmap = linearmap
        self._hashkey = None
        self._cache = {}

    @classmethod
    def from_chargemap(cls, chargemap, dual=False, subinfo=None):
        """Create an index from a mapping of charge to size."""
        items = sorted(chargemap.items())
        return cls(
            charges=[c for c, _ in items],
            sizes=[d for _, d in items],
            dual=dual,
            subinfo=subinfo,
        )

    @classmethod
    def from_blockindex(cls, ix):
        """Create a packed index from a block sparse `BlockIndex`."""
        new = cls.from_chargemap(ix.chargemap, dual=ix.dual)
        if ix.subinfo is not None:
            new._subinfo = PackedSubIndexInfo.from_subindexinfo(
                ix.subinfo, new._charges
            )
        new._linearmap = ix._linearmap
        return new

    def to_blockindex(self):
        """Convert this index to a block sparse `BlockIndex`."""
        from ..sparse.sparse_index import BlockIndex

        subinfo = (
            None
            if self._subinfo is None
            else self._subinfo.to_subindexinfo(self._charges)
        )
        return BlockIndex(
            chargemap=self.chargemap,
            dual=self._dual,
            subinfo=subinfo,
            linearmap=self._linearmap,
        )

    def copy_with(self, **kwargs):
        """Copy this index with some attributes replaced. New values are not
        checked.
        """
        new = self.__new__(self.__class__)
        new._charges = kwargs.pop("charges", self._charges)
        new._sizes = kwargs.pop("sizes", self._sizes)
        new._dual = bool(kwargs.pop("dual", self._dual))
        new._subinfo = kwargs.pop("subinfo", self._subinfo)
        new._linearmap = kwargs.pop("linearmap", self._linearmap)
        new._hashkey = None
        new._cache = {}
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {kwargs}")
        return new

    @property
    def charges(self):
        """The charges of this index, sorted."""
        return self._charges

    @property
    def sizes(self):
        """The size of each charge sector, as an integer array."""
        return self._sizes

    @property
    def chargemap(self):
        """A mapping from charge to size."""
        return dict(zip(self._charges, self._sizes.tolist()))

    @property
    def num_charges(self):
        """The number of charges."""
        return len(self._charges)

    @property
    def size_total(self):
        """The total size of this index."""
        return int(self._sizes.sum())

    @property
    def subshape(self):
        if self._subinfo is None:
            return None
        return self._subinfo.subshape

    @property
    def linearmap(self):
        """The (charge, offset) pair of each element of this index, in linear
        order. Built on demand, assuming sorted charge blocks.
        """
        if self._linearmap is None:
            linearmap = []
            for c, d in zip(self._charges, self._sizes.tolist()):
                linearmap.extend((c, i) for i in range(d))
            self._linearmap = tuple(linearmap)
        return self._linearmap

    def charge_ids(self):
        """A mapping of each charge to its position."""
        try:
            return self._cache["ids"]
        except KeyError:
            ids = self._cache["ids"] = {
                c: i for i, c in enumerate(self._charges)
            }
            return ids

    def charge_array(self, symmetry):
        """The charges as an integer array, ``(num_charges, num_components)``,
        cached per symmetry.
        """
        key = ("array", symmetry)
        try:
            return self._cache[key]
        except KeyError:
            array = self._cache[key] = symmetry.charges_to_array(self._charges)
            return array

    def signed_charge_array(self, symmetry, flip):
        """The charge array, negated if ``flip``, cached."""
        key = ("signed", symmetry, bool(flip))
        try:
            return self._cache[key]
        except KeyError:
            import numpy as np

            array = self.charge_array(symmetry)
            if flip:
                array = symmetry.sign_array(
                    array, np.ones(array.shape[0], dtype=bool)
                )
            self._cache[key] = array
            return array

    def charge_parities(self, symmetry):
        """The parity, 0 or 1, of each charge as an int8 array, cached."""
        key = ("parity", symmetry)
        try:
            return self._cache[key]
        except KeyError:
            import numpy as np

            parities = symmetry.parity_array(self.charge_array(symmetry))
            parities = self._cache[key] = parities.astype(np.int8)
            return parities

    def size_of(self, c):
        """The size of the block with charge ``c``."""
        return int(self._sizes[self.charge_ids()[c]])

    def linear_to_charge_and_offset(self, i):
        """The charge and offset within it of linear index ``i``."""
        return self.linearmap[i]

    def conj(self):
        """A copy of this index with the dualness reversed."""
        subinfo = None if self._subinfo is None else self._subinfo.conj()
        return self.copy_with(dual=not self._dual, subinfo=subinfo)

    def select_charge_ids(self, ids):
        """Copy this index keeping only the charges with ids ``ids``, an
        ascending integer array.
        """
        ids = _as_int_array(ids)
        subinfo = (
            None
            if self._subinfo is None
            else self._subinfo.select_charge_ids(ids)
        )
        return self.copy_with(
            charges=tuple(self._charges[i] for i in ids.tolist()),
            sizes=self._sizes[ids],
            subinfo=subinfo,
            linearmap=None,
        )

    def drop_charges(self, charges):
        """A copy of this index with the charges in ``charges`` removed."""
        charges = set(charges)
        keep = [i for i, c in enumerate(self._charges) if c not in charges]
        return self.select_charge_ids(keep)

    def check(self):
        """Check that the index is well formed."""
        if len(self._charges) != len(self._sizes):
            raise ValueError("Number of charges and sizes differ.")
        if (self._sizes <= 0).any():
            raise ValueError(f"Sizes must be positive, got {self._sizes}.")
        assert list(self._charges) == sorted(self._charges)
        if self._subinfo is not None:
            self._subinfo.check(self)

    def matches(self, other):
        """Check if this index can be contracted with ``other``. Shared charges
        should have the same size, dualness should be opposite and any subindex
        info should match. For debugging.
        """
        sizes = other.chargemap
        for c, d in self.chargemap.items():
            if c in sizes and sizes[c] != d:
                return False
        if self._dual == other._dual:
            return False
        if (self._subinfo is None) != (other._subinfo is None):
            return False
        if self._subinfo is not None:
            return self.to_blockindex().subinfo.matches(
                other.to_blockindex().subinfo
            )
        return True

    def hashkey(self):
        """A hashable key describing the structure of this index."""
        if self._hashkey is None:
            self._hashkey = (
                self._charges,
                self._sizes.tobytes(),
                self._dual,
                None if self._subinfo is None else self._subinfo.hashkey(),
            )
        return self._hashkey

    def __hash__(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.to_blockindex())

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"chargemap={self.chargemap}, dual={self._dual}",
                (
                    f", subinfo={self._subinfo}"
                    if self._subinfo is not None
                    else ""
                ),
                ")",
            ]
        )


class PackedSubIndexInfo(SubInfo):
    """Info needed to unfuse a packed index. Each row describes one subsector,
    i.e. one combination of subindex charges. Rows are grouped by fused charge.
    This is the ragged version of the flat backend's
    ``(ncharge, nsectors, nsubcharges)`` subkeys.

    Parameters
    ----------
    indices : tuple[PackedIndex]
        The indices that were fused, in order.
    subkeys : array_like
        Integer array, ``(num_subsectors, len(indices))``, of the charge ids of
        each subsector in each subindex.
    charge_starts : array_like
        Integer array, ``(num_charges + 1,)``. Subsectors of fused charge id
        ``c`` are rows ``charge_starts[c]:charge_starts[c + 1]``.
    suboffsets : array_like
        Integer array, ``(num_subsectors,)``, of where each subsector starts
        within its fused charge.
    subsizes : array_like
        Integer array, ``(num_subsectors,)``, of the size of each subsector.
    """

    __slots__ = (
        "_charge_starts",
        "_hashkey",
        "_indices",
        "_subkeys",
        "_suboffsets",
        "_subsizes",
    )

    def __init__(self, indices, subkeys, charge_starts, suboffsets, subsizes):
        self._indices = tuple(indices)
        self._charge_starts = _as_int_array(charge_starts)
        self._subkeys = _as_int_array(subkeys).reshape(
            int(self._charge_starts[-1]), len(self._indices)
        )
        self._suboffsets = _as_int_array(suboffsets)
        self._subsizes = _as_int_array(subsizes)
        self._hashkey = None

    @classmethod
    def from_subindexinfo(cls, subinfo, charges):
        """Create from a block sparse `SubIndexInfo`, given the sorted
        ``charges`` of the fused index.
        """
        indices = tuple(map(PackedIndex.from_blockindex, subinfo.indices))
        id_maps = [ix.charge_ids() for ix in indices]
        subkeys, charge_starts, suboffsets, subsizes = [], [0], [], []
        for c in charges:
            start = 0
            for subsector, d in subinfo.extents.get(c, {}).items():
                subkeys.append([m[sc] for m, sc in zip(id_maps, subsector)])
                suboffsets.append(start)
                subsizes.append(d)
                start += d
            charge_starts.append(len(subkeys))
        return cls(
            indices=indices,
            subkeys=subkeys,
            charge_starts=charge_starts,
            suboffsets=suboffsets,
            subsizes=subsizes,
        )

    def to_subindexinfo(self, charges):
        """Convert to a block sparse `SubIndexInfo`, given the sorted
        ``charges`` of the fused index.
        """
        from ..sparse.sparse_index import SubIndexInfo

        sub_charges = [ix.charges for ix in self._indices]
        starts = self._charge_starts.tolist()
        keys = self._subkeys.tolist()
        sizes = self._subsizes.tolist()
        extents = {}
        for f, c in enumerate(charges):
            extent = extents[c] = {}
            for row in range(starts[f], starts[f + 1]):
                subsector = tuple(
                    cs[i] for cs, i in zip(sub_charges, keys[row])
                )
                extent[subsector] = sizes[row]
        return SubIndexInfo(
            indices=tuple(ix.to_blockindex() for ix in self._indices),
            extents=extents,
        )

    @property
    def subkeys(self):
        return self._subkeys

    @property
    def charge_starts(self):
        return self._charge_starts

    @property
    def suboffsets(self):
        return self._suboffsets

    @property
    def subsizes(self):
        return self._subsizes

    @property
    def subshape(self):
        return tuple(ix.size_total for ix in self._indices)

    def copy_with(self, **kwargs):
        new = self.__new__(self.__class__)
        new._indices = kwargs.pop("indices", self._indices)
        new._subkeys = kwargs.pop("subkeys", self._subkeys)
        new._charge_starts = kwargs.pop("charge_starts", self._charge_starts)
        new._suboffsets = kwargs.pop("suboffsets", self._suboffsets)
        new._subsizes = kwargs.pop("subsizes", self._subsizes)
        new._hashkey = None
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {kwargs}")
        return new

    def conj(self):
        """A copy with the dualness of each subindex reversed."""
        return self.copy_with(indices=tuple(ix.conj() for ix in self._indices))

    def select_charge_ids(self, ids):
        """Copy keeping only the subsectors of the fused charges with ids
        ``ids``, an ascending integer array. Charges are renumbered to match
        the selected index.
        """

        import numpy as np

        starts = self._charge_starts
        counts = starts[1:] - starts[:-1]
        rows = np.concatenate(
            [np.arange(starts[c], starts[c + 1]) for c in ids]
            or [np.zeros(0, dtype=np.int64)]
        )
        charge_starts = np.zeros(len(ids) + 1, dtype=np.int64)
        np.cumsum(counts.take(ids), out=charge_starts[1:])
        return self.copy_with(
            subkeys=self._subkeys[rows],
            charge_starts=charge_starts,
            suboffsets=self._suboffsets[rows],
            subsizes=self._subsizes[rows],
        )

    def check(self, parent):
        """Check consistency with the fused ``parent`` index."""
        import numpy as np

        starts = self._charge_starts
        assert len(starts) == parent.num_charges + 1
        assert starts[0] == 0 and (np.diff(starts) >= 0).all()
        cumulative = np.concatenate([[0], np.cumsum(self._subsizes)])
        totals = cumulative[starts[1:]] - cumulative[starts[:-1]]
        assert (totals == parent.sizes).all(), (totals, parent.sizes)
        for j, ix in enumerate(self._indices):
            assert (self._subkeys[:, j] < ix.num_charges).all()

    def hashkey(self):
        if self._hashkey is None:
            self._hashkey = (
                tuple(ix.hashkey() for ix in self._indices),
                self._subkeys.tobytes(),
                self._charge_starts.tobytes(),
                self._suboffsets.tobytes(),
                self._subsizes.tobytes(),
            )
        return self._hashkey

    def __hash__(self):
        raise NotImplementedError

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"indices={self._indices}, "
            f"num_subsectors={self._subkeys.shape[0]})"
        )
