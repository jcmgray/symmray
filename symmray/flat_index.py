import autoray as ar

from .index_common import Index, SubInfo
from .utils import DEBUG


class FlatIndex(Index):
    """Simple class to store dualness and any fuse information of an index.

    Parameters
    ----------
    dual : bool, optional
        Whether the index is dual (i.e., contributes a negative sign to the
        corresponding charge). Default is False.
    subinfo : FlatSubIndexInfo, optional
        Information about the subindex, if this index is a fused index.
        Default is None, which means the index is not fused.
    """

    __slots__ = ("_num_charges", "_charge_size", "_dual", "_subinfo")

    def __init__(self, num_charges, charge_size, dual=False, subinfo=None):
        self._num_charges = int(num_charges)
        self._charge_size = int(charge_size)
        self._dual = dual
        self._subinfo = subinfo

        if DEBUG:
            self.check()

    def copy_with(
        self, num_charges=None, charge_size=None, dual=None, **kwargs
    ):
        # handle subinfo to distinguish between passing None and not passing it
        if "subinfo" in kwargs:
            new_subinfo = kwargs.pop("subinfo")
        else:
            new_subinfo = self._subinfo

        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {kwargs}")

        return FlatIndex(
            num_charges if num_charges is not None else self._num_charges,
            charge_size if charge_size is not None else self._charge_size,
            dual if dual is not None else self._dual,
            new_subinfo,
        )

    @property
    def num_charges(self) -> int:
        """The number of charges associated with this index."""
        return self._num_charges

    @property
    def charge_size(self) -> int:
        """The size of the charges associated with this index."""
        return self._charge_size

    @property
    def subshape(self):
        if self._subinfo is None:
            return None
        return self._subinfo.subshape

    @property
    def size_total(self) -> int:
        return self._num_charges * self._charge_size

    def conj(self) -> "FlatIndex":
        """Return the conjugate of the index, i.e., flip the dualness and
        subinfo.
        """
        return FlatIndex(
            num_charges=self._num_charges,
            charge_size=self._charge_size,
            dual=not self._dual,
            subinfo=None if self._subinfo is None else self._subinfo.conj(),
        )

    def select_charge(self, charge):
        """Drop all but the specified charge from this index."""
        if self._subinfo is not None:
            new_subinfo = self._subinfo.select_charge(charge)
            return FlatIndex(
                num_charges=1,
                charge_size=self._charge_size,
                dual=self._dual,
                subinfo=new_subinfo,
            )
        return self

    def check(self):
        """Check that the index is valid."""
        assert self._num_charges > 0, "Number of charges must be positive."
        assert self._charge_size > 0, "Charge size must be positive."
        if self._subinfo is not None:
            assert isinstance(self._subinfo, FlatSubIndexInfo), (
                "Subindex info must be an instance of FlatSubIndexInfo."
            )
            self._subinfo.check()

    def __repr__(self):
        s = [f"{self.__class__.__name__}("]
        s.append(
            f"num_charges={self._num_charges}, "
            f"charge_size={self._charge_size}, "
            f"dual={self._dual}"
        )
        if self._subinfo is not None:
            s.append(f", subinfo={self._subinfo!r}")
        s.append(")")
        return "".join(s)


class FlatSubIndexInfo(SubInfo):
    """Information required to unfuse a fused index.

    Parameters
    ----------
    indices : tuple[FlatIndex]
        The indices that have been fused.
    subkeys : array_like
        The subkeys for the fused index, with shape
        (ncharge, nsectors, nsubcharges). I.e. the first axis selects the
        overall fused charge, the second axis selects the subsector within
        that charge, and the third axis selects the individual charge within
        that subsector.
    """

    __slots__ = (
        "_indices",
        "_subkeys",
        "_ncharge",
        "_nsectors",
        "_nsubcharges",
    )

    def __init__(self, indices, subkeys):
        self._indices = tuple(
            x if isinstance(x, FlatIndex) else FlatIndex(x) for x in indices
        )
        self._subkeys = subkeys
        subkey_shape = ar.do("shape", subkeys)
        # number of overall charges, e.g. {0, 1} -> 2
        # number of subsectors e.g. [000, 011, 101, 110] -> 4
        # number of subcharges, e.g. 3 for above
        self._ncharge, self._nsectors, self._nsubcharges = subkey_shape

        if DEBUG:
            self.check()

    @property
    def subshape(self) -> tuple[int]:
        """The subshape of the fused index."""
        return tuple(ix.size_total for ix in self._indices)

    @property
    def subkeys(self):
        """The subkeys for the fused index."""
        return self._subkeys

    @property
    def ncharge(self) -> int:
        """Number of overall charges in this subindex."""
        return self._ncharge

    @property
    def nsectors(self) -> int:
        """Number of subsectors in this subindex."""
        return self._nsectors

    @property
    def nsubcharges(self) -> int:
        """Number of subcharges in this subindex."""
        return self._nsubcharges

    def check(self):
        assert len(self._indices) == len(self._subkeys[0, 0])
        for ix in self._indices:
            ix.check()

    def conj(self):
        return FlatSubIndexInfo(
            indices=tuple(ix.conj() for ix in self._indices),
            subkeys=self._subkeys,
        )

    def select_charge(self, charge):
        new_subkeys = self.subkeys[[charge]]
        return FlatSubIndexInfo(
            indices=self.indices,
            subkeys=new_subkeys,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ncharge={self._ncharge}, "
            f"nsectors={self._nsectors}, "
            f"nsubcharges={self._nsubcharges})"
        )
