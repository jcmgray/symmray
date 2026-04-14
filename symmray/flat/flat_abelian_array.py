"""Abelian symmetric arrays with flat backend."""

import autoray as ar

from ..array_common import ArrayCommon
from ..bosonic_common import BosonicCommon
from ..common import SymmrayCommon
from ..linalg_common import Absorb
from ..sparse.sparse_abelian_array import AbelianArray
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .flat_array_common import FlatArrayCommon
from .flat_data_common import FlatCommon
from .flat_index import FlatIndex
from .flat_vector import FlatVector


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

    def to_pytree(self):
        """Convert this flat abelian array to a pytree purely of non-symmray
        containers and objects.
        """
        return self._to_pytree_abelian()

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

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        return self._binary_blockwise_op_abelian(
            other, fn, missing=missing, inplace=inplace
        )

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

    def trace(self):
        """Compute trace of the flat array, assuming it is a square matrix."""
        return self._trace_abelian()

    def einsum(self, eq, preserve_array=False):
        """Einsum for flat abelian arrays, currently only single term.

        Parameters
        ----------
        eq : str
            The einsum equation, e.g. "abcb->ca". The output indices must be
            specified and only trace and permutations are allowed.
        preserve_array : bool, optional
            If tracing to a scalar, whether to return an AbelainArray object
            with no indices, or simply scalar itself (the default).

        Returns
        -------
        FlatAbelianArray or scalar
        """
        return self._einsum_abelian(eq, preserve_array=preserve_array)

    def squeeze(self, axis, inplace=False):
        """Squeeze this flat abelian array, removing axes of size 1.

        Parameters
        ----------
        axis : int or sequence of int, optional
            The axes to squeeze. If not given, all axes of size 1 will be
            removed.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        AbelianArrayFlat
        """
        return self._squeeze_abelian(axis, inplace=inplace)

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

    # --------------------------- linalg methods ---------------------------- #

    def cholesky(self, *, upper=False) -> "AbelianArrayFlat":
        """Cholesky decomposition of this assumed positive-definite flat
        abelian array.

        Parameters
        ----------
        upper : bool, optional
            Whether to return the upper triangular Cholesky factor.
            Default is False, returning the lower triangular factor.

        Returns
        -------
        l_or_r : AbelianArrayFlat
            The Cholesky factor. Lower triangular if ``upper=False``,
            upper triangular if ``upper=True``.
        """
        return self._cholesky_abelian(upper=upper, shift=0)

    def cholesky_regularized(
        self, absorb=0, shift=True
    ) -> tuple["AbelianArrayFlat", None, "AbelianArrayFlat"]:
        """Cholesky decomposition with optional diagonal regularization,
        returning results in an SVD-like ``(left, None, right)`` format
        for compatibility with tensor network split drivers.

        Parameters
        ----------
        absorb : {-12, 0, 12}, optional
            How to return the factors:

            - ``0`` (``'both'``): return ``(L, None, L^H)``.
            - ``-12`` (``'lsqrt'``): return ``(L, None, None)``.
            - ``12`` (``'rsqrt'``): return ``(None, None, L^H)``.

        shift : float, optional
            Diagonal regularization shift. If True or negative, auto-compute
            from dtype machine epsilon. The shift is always applied as a
            relative shift scaled by the trace of each block. Default is True.

        Returns
        -------
        left : AbelianArrayFlat or None
            The lower Cholesky factor, or None.
        s : None
            Always None (no singular values).
        right : AbelianArrayFlat or None
            The conjugate transpose of the Cholesky factor, or None.
        """
        absorb = Absorb.parse(absorb)

        if absorb == Absorb.sqVH:
            r = self._cholesky_abelian(shift=shift, upper=True)
            return None, None, r

        l = self._cholesky_abelian(shift=shift, upper=False)
        if absorb == Absorb.Usq:
            return l, None, None

        # absorb == Absorb.Usq_sqVH (0 or 'both')
        return l, None, l.H

    def eigh(self) -> tuple[FlatVector, "AbelianArrayFlat"]:
        """Hermitian eigen-decomposition of this flat abelian array.

        Returns
        -------
        eigenvalues : FlatVector
            The eigenvalues.
        eigenvectors : AbelianArrayFlat
            The abelian array of right eigenvectors.
        """
        return self._eigh_abelian()


class Z2ArrayFlat(AbelianArrayFlat):
    static_symmetry = get_symmetry("Z2")
