"""Basic classes and functions for flat backend objects."""

import autoray as ar

from ..common import SymmrayCommon
from ..utils import DEBUG


class FlatCommon:
    """Mixin class for flat arrays."""

    __slots__ = ("_blocks", "_sectors", "backend")

    @property
    def sectors(self):
        """The stack of sector keys, with shape (num_blocks, ndim). Each row
        represents a sector of a corresponding block, and each column
        represents a charge in a given axis."""
        return self._sectors

    @property
    def blocks(self):
        """The stack of array blocks, with shape (num_blocks, *shape_block),
        i.e. `ndim + 1` dimensions, where the first dimension is the block
        index, which should match the first dimension of `sectors`, and the
        rest are the dimensions of individual blocks."""
        return self._blocks

    @property
    def dtype(self):
        """Get the dtype name for the blocks."""
        return ar.get_dtype_name(self._blocks)

    def _get_shape_blocks_full(self) -> tuple[int, ...]:
        """Get the full shape of the stacked blocks, including the number of
        blocks."""
        return ar.do("shape", self._blocks, like=self.backend)

    @property
    def num_blocks(self) -> int:
        """Get the number of blocks in the array."""
        return self._get_shape_blocks_full()[0]

    def get_params(self):
        """Interface for getting underlying arrays."""
        return self._blocks

    def set_params(self, params):
        """Interface for setting underlying arrays."""
        self._blocks = params
        self.backend = ar.infer_backend(self._blocks)
        try:
            self._sectors = ar.do("array", self._sectors, like=params)
        except ImportError:
            # params is possibly a placeholder of some kind
            pass

    def item(self):
        """Convert the block array to a scalar if it is a scalar block array."""
        return self._blocks.item()

    def __mul__(self, other):
        return self.copy_with(blocks=self._blocks * other)

    def __rmul__(self, other):
        return self.copy_with(blocks=other * self._blocks)

    def __truediv__(self, other):
        return self.copy_with(blocks=self._blocks / other)

    def __neg__(self):
        return self.copy_with(blocks=-self._blocks)

    def __float__(self):
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def _do_unary_op(self, fn, inplace=False):
        """Perform a unary operation on blocks of the array."""
        new = self if inplace else self.copy()
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
        new._blocks = fn(new._blocks)
        return new

    def abs(self):
        """Get the absolute value of all elements in the array."""
        return self._do_unary_op("abs")

    def isfinite(self):
        """Check if all elements in the array are finite."""
        return self._do_unary_op("isfinite")

    def sqrt(self):
        """Get the square root of all elements in the array."""
        return self._do_unary_op("sqrt")

    def clip(self, a_min, a_max):
        """Clip the values in the array."""
        new = self.copy()
        _clip = ar.get_lib_fn(self.backend, "clip")
        new._blocks = _clip(new._blocks, a_min, a_max)
        return new

    def max(self):
        """Get the maximum element from any block in the array."""
        _max = ar.get_lib_fn(self.backend, "max")
        return _max(self._blocks)

    def min(self):
        """Get the minimum element from any block in the array."""
        _min = ar.get_lib_fn(self.backend, "min")
        return _min(self._blocks)

    def sum(self):
        """Get the sum of all elements in the array."""
        _sum = ar.get_lib_fn(self.backend, "sum")
        return _sum(self._blocks)

    def all(self):
        """Check if all elements in the array are True."""
        _all = ar.get_lib_fn(self.backend, "all")
        return _all(self._blocks)

    def any(self):
        """Check if any element in the array is True."""
        _any = ar.get_lib_fn(self.backend, "any")
        return _any(self._blocks)

    def norm(self):
        _norm = ar.get_lib_fn(self.backend, "linalg.norm")
        return _norm(self._blocks)


class FlatVector(FlatCommon, SymmrayCommon):
    """Class for storing block vectors with flat storage, e.g. for the
    singular- or eigen- values of a matrix.

    Parameters
    ----------
    sectors : array_like
        The vector of charges, with shape (num_blocks,).
    blocks : array_like
        The stack of vectors, with shape (num_blocks, charge_subsize).
    """

    __slots__ = ("_blocks", "_sectors", "backend")

    def __init__(self, sectors, blocks):
        self._blocks = (
            blocks if hasattr(blocks, "shape") else ar.do("array", blocks)
        )
        self._sectors = (
            sectors
            if hasattr(sectors, "shape")
            else ar.do("array", sectors, like=blocks)
        )
        self.backend = ar.infer_backend(self._blocks)

        if DEBUG:
            self.check()

    @property
    def size(self):
        """The total size of all elements in the vector."""
        db, dv = self._get_shape_blocks_full()
        return db * dv

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the effective shape of the vector."""
        return (self.size,)

    def copy(self, deep=False) -> "FlatVector":
        if deep:
            blocks = ar.do("copy", self._blocks, like=self.backend)
            sectors = ar.do("copy", self._sectors, like=self.backend)
        else:
            blocks = self._blocks
            sectors = self._sectors

        return self.__class__(sectors, blocks)

    def copy_with(self, sectors=None, blocks=None):
        """Create a copy of the vector with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self.__new__(self.__class__)
        new._sectors = self._sectors if sectors is None else sectors
        new._blocks = self._blocks if blocks is None else blocks
        new.backend = self.backend

        if DEBUG:
            new.check()

        return new

    def modify(self, sectors=None, blocks=None):
        """Modify the vector in place with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        if sectors is not None:
            self._sectors = sectors
        if blocks is not None:
            self._blocks = blocks

        if DEBUG:
            self.check()

        return self

    def check(self):
        assert ar.do("ndim", self._blocks, like=self.backend) == 2

    def to_blockvector(self):
        from ..sparse.sparse_base import BlockVector

        return BlockVector(
            {k.item(): b for k, b in zip(self.sectors, self.blocks)}
        )

    def to_dense(self):
        return ar.do("reshape", self._blocks, (-1,), like=self.backend)

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"total_size={self.size}, ",
                f"num_blocks={self.num_blocks})",
            ]
        )
