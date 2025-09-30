"""Abelian symmetric arrays with flat backend.

Flat backend with branchless design to allow static
computational graphs that can be easily compiled and vectorized etc.

TODO:

- [ ] cache patterns and reshapers/slicers
- [ ] cache properties, funcs
- [ ] roll our own repeat and rearrange to avoid einops dependency

"""

import autoray as ar

from ..abelian_common import AbelianCommon
from ..common import SymmrayCommon
from ..sparse.sparse_abelian_array import AbelianArray
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .flat_array import FlatArrayCommon
from .flat_base import FlatCommon


class AbelianArrayFlat(
    FlatArrayCommon,
    FlatCommon,
    AbelianCommon,
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
    """

    __slots__ = (
        "_blocks",
        "_sectors",
        "_indices",
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
    ):
        self._init_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
            symmetry=symmetry,
        )

        if DEBUG:
            # might not have completed fermionic setup yet
            self.check()

    def check(self):
        """Check the internal consistency of the array."""
        self._check_abelian()

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

    @classmethod
    def from_blocks(cls, blocks, indices, symmetry=None) -> "AbelianArrayFlat":
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

        Returns
        -------
        AbelianArrayFlat
        """
        sectors = list(map(list, blocks.keys()))

        if blocks:
            blocks = ar.do("stack", tuple(blocks.values()))
        else:
            blocks = []
        return cls(sectors, blocks, indices, symmetry=symmetry)

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
        """
        return cls.from_blocks(
            blocks=x.blocks,
            indices=x.duals,
            symmetry=symmetry or x.symmetry,
        )

    def to_blocksparse(self) -> AbelianArray:
        """Create a blocksparse abelian array from this flat abelian array."""
        return self._to_blocksparse_abelian()

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

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        return self._binary_blockwise_op_abelian(
            other, fn, missing=missing, inplace=inplace
        )

    def transpose(
        self,
        axes=None,
        inplace=False,
    ) -> "AbelianArrayFlat":
        """Transpose this flat abelian array.

        Parameters
        ----------
        axes : tuple[int, ...] | None, optional
            A permutation of the axes to transpose the array by. If None,
            the axes will be reversed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArrayFlat
        """
        return self._transpose_abelian(axes=axes, inplace=inplace)

    def conj(self, inplace=False) -> "AbelianArrayFlat":
        """Return the complex conjugate of this flat abelian array, including
        the indices and any subindex fusing information.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArrayFlat
        """
        return self._conj_abelian(inplace=inplace)

    def _fuse_core(
        self,
        *axes_groups,
        inplace=False,
    ) -> "AbelianArrayFlat":
        """The core implementation of the fuse operation, which fuses
        multiple axes into a single group, and returns a new array with
        the new sectors and blocks. The new axes are inserted at the minimum
        axis of any of the groups.
        """
        return self._fuse_core_abelian(*axes_groups, inplace=inplace)

    def unfuse(self, axis, inplace=False) -> "AbelianArrayFlat":
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.

        Parameters
        ----------
        axis : int
            The axis to unfuse. It must have fuse information
            (`.indices[axis].subinfo`), typically from a previous fusing
            operation.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArrayFlat
        """
        return self._unfuse_abelian(axis, inplace=inplace)

    def __matmul__(
        self: "AbelianArrayFlat",
        other: "AbelianArrayFlat",
        preserve_array=False,
    ):
        return self._matmul_abelian(other=other, preserve_array=preserve_array)

    def tensordot(
        self,
        other,
        axes=2,
        mode="auto",
        preserve_array=False,
    ):
        return self._tensordot_abelian(
            other, axes=axes, mode=mode, preserve_array=preserve_array
        )

    def squeeze(self, axis, inplace=False):
        """Assuming `axis` has total size 1, remove it from this array."""
        axs_rem = tuple(i for i in range(self.ndim) if i != axis)

        new_sectors = self.sectors[:, axs_rem]
        new_indices = tuple(self._indices[i] for i in axs_rem)
        block_selector = tuple(
            slice(None) if i != axis + 1 else 0 for i in range(self.ndim + 1)
        )
        new_blocks = self.blocks[block_selector]

        return self._modify_or_copy(
            sectors=new_sectors,
            indices=new_indices,
            blocks=new_blocks,
            inplace=inplace,
        )

    def isel(self, axis, idx, inplace=False):
        """Select a single index along the specified axis."""
        if axis < 0:
            axis += self.ndim
        new = self.select_charge(axis, idx, inplace=inplace)
        return new.squeeze(axis, inplace=True)

    def __getitem__(self, item):
        axis = None
        idx = None

        if not isinstance(item, tuple):
            raise TypeError(
                f"Expected a tuple for indexing, got {type(item)}: {item}"
            )

        for i, s in enumerate(item):
            if isinstance(s, slice):
                if not s.start is s.stop is s.step is None:
                    raise NotImplementedError("Can only slice whole axes.")
            else:
                if axis is not None:
                    raise ValueError(
                        "Can only index one axis at a time, "
                        f"got {item} with multiple indices."
                    )
                axis = i
                idx = s

        return self.isel(axis, idx)

    def allclose(self, other: "AbelianArrayFlat", **allclose_opts) -> bool:
        """Check if two flat abelian arrays are equal to within some tolerance,
        including their sectors and signature.
        """
        return self._allclose_abelian(other, **allclose_opts)

    def test_allclose(self, other: "AbelianArrayFlat", **allclose_opts):
        """Assert that this ``AbelianArrayFlat`` is close to another,
        that is, has all the same sectors, and the corresponding arrays are
        close. Unlike `allclose`, this raises an AssertionError with details
        if not.

        Parameters
        ----------
        other : AbelianArrayFlat
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Raises
        ------
        AssertionError
            If the arrays are not close.
        """
        return self._test_allclose_abelian(other, **allclose_opts)


class Z2ArrayFlat(AbelianArrayFlat):
    static_symmetry = get_symmetry("Z2")
