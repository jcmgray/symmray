"""Abelian symmetric arrays with packed backend."""

from ..array_common import ArrayCommon
from ..bosonic_common import BosonicCommon
from ..common import SymmrayCommon
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .packed_array_common import PackedArrayCommon
from .packed_data_common import PackedCommon


class AbelianArrayPacked(
    PackedArrayCommon,
    PackedCommon,
    BosonicCommon,
    ArrayCommon,
    SymmrayCommon,
):
    """Block sparse array with symmetry constraints and packed storage. Blocks
    are stored C contiguous, end to end, in one data buffer. Each block is
    labelled by the position of its charges in each index.

    Parameters
    ----------
    sectors : array_like
        Integer array, ``(num_blocks, ndim)``, giving the charge id (position
        within ``indices[ax].charges``) of each block on each axis. Rows must
        be unique and lexicographically sorted.
    data : array_like
        The 1D buffer of all blocks, in the order of ``sectors``.
    indices : tuple[PackedIndex]
        The indices of the array.
    charge : hashable, optional
        The total charge of the array, inferred from the first sector if not
        given.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    label : hashable, optional
        An optional label for the array.
    """

    __slots__ = (
        "_charge",
        "_data",
        "_indices",
        "_label",
        "_offsets",
        "_sectors",
        "_shapes",
        "_skey",
        "_symmetry",
    )
    fermionic = False
    static_symmetry = None

    def __init__(
        self,
        sectors,
        data,
        indices,
        charge=None,
        symmetry=None,
        label=None,
    ):
        self._init_abelian(
            sectors=sectors,
            data=data,
            indices=indices,
            charge=charge,
            symmetry=symmetry,
            label=label,
        )
        if DEBUG:
            self.check()

    def check(self):
        """Check the internal consistency of the array."""
        self._check_abelian()

    def new_with(
        self, indices, charge, data, sectors, shapes, offsets, label=None
    ) -> "AbelianArrayPacked":
        """Create a new packed abelian array of the same class as this one."""
        return self._new_with_abelian(
            indices=indices,
            charge=charge,
            data=data,
            sectors=sectors,
            shapes=shapes,
            offsets=offsets,
            label=label,
        )

    def copy(self) -> "AbelianArrayPacked":
        """Copy this array. The data is shared and treated as immutable."""
        return self._copy_abelian()

    def copy_with(self, **kwargs) -> "AbelianArrayPacked":
        """Copy this array with some attributes replaced. For internal use, new
        values are only checked if DEBUG is set.
        """
        new = self._copy_with_abelian(**kwargs)
        if DEBUG:
            new.check()
        return new

    def modify(self, **kwargs) -> "AbelianArrayPacked":
        """Modify this array inplace with some attributes replaced. For
        internal use, new values are only checked if DEBUG is set.
        """

        self._modify_abelian(**kwargs)
        if DEBUG:
            self.check()
        return self

    @classmethod
    def from_blocksparse(cls, x, symmetry=None) -> "AbelianArrayPacked":
        """Create a packed array from a block sparse `AbelianArray`."""
        indices, sectors, data, _ = cls._parts_from_blocksparse(x)
        return cls(
            sectors=sectors,
            data=data,
            indices=indices,
            charge=x.charge,
            symmetry=symmetry if symmetry is not None else x.symmetry,
            label=x.label,
        )

    def to_blocksparse(self):
        """Convert to a block sparse `AbelianArray`."""
        from ..utils import get_array_cls

        indices, blocks, _ = self._to_blocksparse_parts()
        cls = get_array_cls(self.symmetry, fermionic=False, layout="sparse")
        return cls(
            indices=indices,
            charge=self.charge,
            blocks=blocks,
            symmetry=self.symmetry,
            label=self.label,
        )

    def _to_pytree_abelian(self):
        return self.to_blocksparse()._to_pytree_abelian()

    def solve(self, b, **kwargs) -> "AbelianArrayPacked":
        """Solve ``A @ x == b`` for ``x``, where ``A`` is this array."""

        return self._solve_abelian(b, **kwargs)


# --------------------------------------------------------------------------- #


class Z2ArrayPacked(AbelianArrayPacked):
    """A packed array with Z2 symmetry."""

    static_symmetry = get_symmetry("Z2")


class U1ArrayPacked(AbelianArrayPacked):
    """A packed array with U1 symmetry."""

    static_symmetry = get_symmetry("U1")


class Z2Z2ArrayPacked(AbelianArrayPacked):
    """A packed array with Z2 x Z2 symmetry."""

    static_symmetry = get_symmetry("Z2Z2")


class U1U1ArrayPacked(AbelianArrayPacked):
    """A packed array with U1 x U1 symmetry."""

    static_symmetry = get_symmetry("U1U1")
