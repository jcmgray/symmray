"""Abelian symmetric arrays with flat backend."""

import autoray as ar

from ..array_common import ArrayCommon
from ..bosonic_common import BosonicCommon
from ..common import SymmrayCommon
from ..sparse.sparse_abelian_array import AbelianArray
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .flat_array_common import FlatArrayCommon
from .flat_data_common import FlatCommon
from .flat_index import FlatIndex


class AbelianArrayFlat(
    FlatArrayCommon,
    FlatCommon,
    BosonicCommon,
    ArrayCommon,
    SymmrayCommon,
):
    """Base class for abelian arrays with flat storage and cyclic symmetry.

    Parameters
    ----------
    sectors : array_like
        The stack of sector keys, with shape (num_blocks, ndim). Each row
        represents a sector of a corresponding block, and each column
        represents a charge in a given axis.
    blocks : array_like
        The stack of array blocks, with shape (num_blocks, *shape_block), i.e.
        `ndim + 1` dimensions, where the first dimension is the block index,
        which should match the first dimension of `sectors`, and the rest are
        the dimensions of individual blocks.
    indices : sequence[FlatIndex]
        Indices describing the dualness and any subindex information for each
        dimension of the array. If bools are supplied, they will be converted
        to a FlatIndex with the corresponding dualness, and no subindex
        information.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = (
        "_blocks",
        "_indices",
        "_label",
        "_sectors",
        "_symmetry",
        "backend",
    )

    fermionic = False
    static_symmetry = None

    def __init__(
        self,
        sectors,
        blocks,
        indices,
        symmetry=None,
        label=None,
    ):
        self._init_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
            symmetry=symmetry,
            label=label,
        )

        if DEBUG:
            # might not have completed fermionic setup yet
            self.check()

    def check(self):
        """Check the internal consistency of the array."""
        self._check_abelian()

    def new_with(
        self,
        sectors,
        blocks,
        indices,
        label=None,
    ) -> "AbelianArrayFlat":
        """Create a new flat abelian array of the same class as this one.
        Unlike `copy`, this does not copy over any existing data and drops
        for example `label`.
        """
        return self._new_with_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
            label=label,
        )

    def copy(self, deep=False) -> "AbelianArrayFlat":
        """Create a copy of the array."""
        return self._copy_abelian(deep=deep)

    def copy_with(
        self,
        sectors=None,
        blocks=None,
        indices=None,
    ) -> "AbelianArrayFlat":
        """A copy of this block array with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self._copy_with_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
        )
        if DEBUG:
            new.check()
        return new

    def modify(
        self,
        sectors=None,
        blocks=None,
        indices=None,
    ) -> "AbelianArrayFlat":
        """Modify this flat array in place with some attributes replaced. Note
        that checks are not performed on the new properties, this is intended
        for internal use.
        """
        self._modify_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
        )
        if DEBUG:
            self.check()
        return self

    def set_params(self, params):
        """Set the underlying array blocks."""
        self._set_params_abelian(params)
        if DEBUG:
            self.check()

    @classmethod
    def from_pytree(cls, pytree) -> "AbelianArrayFlat":
        """Create a flat abelian array from a pytree purely of non-symmray
        containers and objects.
        """
        indices = tuple(map(FlatIndex.from_pytree, pytree["indices"]))
        return cls(
            sectors=pytree["sectors"],
            blocks=pytree["blocks"],
            indices=indices,
            symmetry=pytree["symmetry"],
            label=pytree["label"],
        )

    @classmethod
    def from_blocks(
        cls, blocks, indices, symmetry=None, label=None
    ) -> "AbelianArrayFlat":
        """Create a flat array from an explicit dictionary of blocks, and
        sequence of indices or duals.

        Parameters
        ----------
        blocks : dict[tuple[int, ...], array_like]
            A dictionary mapping sector keys (tuples of charges) to blocks
            (arrays).
        indices : sequence[FlatIndex] | sequence[bool]
            A sequence of indices describing the dualness and any subindex
            information for each dimension of the array. If bools are supplied,
            they will be converted to a FlatIndex with the corresponding
            dualness, and no subindex information.
        symmetry : str or Symmetry, optional
            The symmetry of the array, if not using a specific symmetry class.
        label : hashable, optional
            An optional label for the array, potentially needed for ordering
            dummy odd fermionic modes.


        Returns
        -------
        AbelianArrayFlat
        """
        sectors = list(map(list, blocks.keys()))

        if blocks:
            blocks = ar.do("stack", tuple(blocks.values()))
        else:
            blocks = []
        return cls(
            sectors,
            blocks,
            indices,
            symmetry=symmetry,
            label=label,
        )

    @classmethod
    def from_blocksparse(
        cls,
        x: AbelianArray,
        symmetry=None,
    ) -> "AbelianArrayFlat":
        """Create a flat abelian array from a blocksparse abelian array.

        Parameters
        ----------
        x : AbelianArray
            The blocksparse abelian array to convert.
        symmetry : str or Symmetry, optional
            The symmetry to use. If not supplied, the symmetry of `x` is used.
        """
        # ensure we only have index charges present on blocks
        x = x.sync_charges()

        return cls.from_blocks(
            blocks=x.blocks,
            indices=[ix.to_flat() for ix in x.indices],
            symmetry=symmetry or x.symmetry,
            label=x.label,
        )

    def to_blocksparse(self) -> AbelianArray:
        """Create a blocksparse abelian array from this flat abelian array."""
        return self._to_blocksparse_abelian()

    def to_dense(self):
        """Convert this abelian array to a dense array, by combining all the
        blocks into a single large array, filling in zeros where necessary.

        Returns
        -------
        array_like
            A dense array with the same shape as this abelian array.
        """
        return self._to_dense_abelian()

    def _map_blocks(self, fn_sector=None, fn_block=None):
        self._map_blocks_abelian(fn_sector=fn_sector, fn_block=fn_block)

    def sort_stack(
        self,
        axes=None,
        all_axes=None,
        inplace=False,
    ) -> "AbelianArrayFlat":
        """Lexicgraphic sort the stack of sectors and blocks according to the
        values of charges in the specified axes, optionally filling in the rest
        of the axes with the remaining axes in the order they appear.

        Parameters
        ----------
        axes : int | tuple[int, ...], optional
            The axes to sort by. If a single integer is given, it will be
            interpreted as the axis to sort by. If a tuple of integers is
            given, it will be interpreted as the axes to sort by in order.
            Default is None, if all_axes is also None or True, this will sort
            all axes in their current order.
        all_axes : bool, optional
            Whether to include all non-specified axes as tie-breakers, after
            the specified axes. If ``None``, the default, this will be True
            if `axes` is not supplied explicitly, and False otherwise.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.
            Default is False, which returns a new array.
        """
        kord = self.get_sorting_indices(axes=axes, all_axes=all_axes)
        return self._modify_or_copy(
            sectors=self._sectors[kord],
            blocks=self._blocks[kord],
            inplace=inplace,
        )


class Z2ArrayFlat(AbelianArrayFlat):
    static_symmetry = get_symmetry("Z2")
