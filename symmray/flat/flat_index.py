"""Index objects for flat arrays."""

import autoray as ar

from ..index_common import Index, SubInfo
from ..utils import DEBUG


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

    def __init__(
        self,
        num_charges,
        charge_size,
        dual=False,
        subinfo=None,
        linearmap=None,
    ):
        self._num_charges = int(num_charges)
        self._charge_size = int(charge_size)
        self._dual = dual
        self._subinfo = subinfo
        self._linearmap = linearmap

        if DEBUG:
            self.check()

    def copy_with(
        self,
        num_charges=None,
        charge_size=None,
        dual=None,
        linearmap=None,
        **kwargs,
    ):
        """A copy of this index with some attributes replaced. Note that checks
        are not performed on the new propoerties, this is intended for internal
        use.
        """
        new = self.__new__(self.__class__)
        keep_linearmap = True

        if num_charges is None:
            new._num_charges = self._num_charges
        else:
            new._num_charges = num_charges
            keep_linearmap = False

        if charge_size is None:
            new._charge_size = self._charge_size
        else:
            new._charge_size = charge_size
            keep_linearmap = False

        if linearmap is not None:
            new._linearmap = linearmap
        elif keep_linearmap:
            new._linearmap = self._linearmap
        else:
            new._linearmap = None

        new._dual = dual if dual is not None else self._dual

        # handle subinfo to distinguish between passing None and not passing it
        if "subinfo" in kwargs:
            new._subinfo = kwargs.pop("subinfo")
        else:
            new._subinfo = self._subinfo
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {kwargs}")

        return new

    @property
    def num_charges(self) -> int:
        """The number of charges associated with this index."""
        return self._num_charges

    @property
    def charge_size(self) -> int:
        """The size of the charges associated with this index."""
        return self._charge_size

    # @property
    # def linearmap(self):
    #     """The linear map from charge and offset to linear index."""
    #     # XXX: this should be an array compat operation
    #     if self._linearmap is None:
    #         linearmap = []
    #         for c in range(self._num_charges):
    #             for ci in range(self._charge_size):
    #                 linearmap.append((c, ci))
    #         self._linearmap = tuple(linearmap)
    #     return self._linearmap

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

    def select_charge(self, charge, subselect=None) -> "FlatIndex":
        """Drop all but the specified charge from this index.

        Parameters
        ----------
        charge : int
            The charge to keep.
        subselect : slice or array_like, optional
            If provided, a range of indices within the selected charge block
            to keep. If not provided, the entire block is kept.

        Returns
        -------
        FlatIndex
        """
        updates = {"num_charges": 1}

        if subselect is not None:
            if isinstance(subselect, slice):
                start, stop, step = subselect.indices(self.charge_size)
                updates["charge_size"] = len(range(start, stop, step))
            elif hasattr(subselect, "size"):  # numpy array or similar
                updates["charge_size"] = subselect.size
            else:
                updates["charge_size"] = len(subselect)

            # drop subinfo if we slice blocks
            updates["subinfo"] = None

        elif self._subinfo is not None:
            updates["subinfo"] = self._subinfo.select_charge(charge)

        return self.copy_with(**updates)

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
        if self._linearmap is None:
            # default mapping is sorted charges
            return divmod(i, self._charge_size)
        return self._linearmap[i]

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
        new_subkeys = self.subkeys[(charge,), ...]
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
