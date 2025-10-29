"""Basic classes and functions for flat backend objects."""

import autoray as ar


class FlatCommon:
    """Mixin class for flat arrays."""

    __slots__ = ("_blocks", "_sectors", "backend")

    def _init_flatcommon(self, sectors, blocks):
        self._blocks = (
            blocks if hasattr(blocks, "shape") else ar.do("array", blocks)
        )
        # infer the backend to reuse for efficiency
        self.backend = ar.infer_backend(self._blocks)

        self._sectors = (
            sectors
            if hasattr(sectors, "shape")
            else ar.do("array", sectors, like=self._blocks)
        )

    def _new_with_flatcommon(self, sectors, blocks):
        new = self.__new__(self.__class__)
        new._sectors = sectors
        new._blocks = blocks
        new.backend = self.backend
        return new

    def _copy_flatcommon(self, deep=False):
        new = self.__new__(self.__class__)
        if deep:
            new._sectors = ar.do("copy", self._sectors, like=self.backend)
            new._blocks = ar.do("copy", self._blocks, like=self.backend)
        else:
            new._sectors = self._sectors
            new._blocks = self._blocks
        new.backend = self.backend
        return new

    def _copy_with_flatcommon(self, sectors=None, blocks=None):
        new = self.__new__(self.__class__)
        new._sectors = self._sectors if sectors is None else sectors
        new._blocks = self._blocks if blocks is None else blocks
        new.backend = self.backend
        return new

    def _modify_flatcommon(self, sectors=None, blocks=None):
        if sectors is not None:
            self._sectors = sectors
        if blocks is not None:
            self._blocks = blocks
        return self

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
    def shape_block(self) -> tuple[int, ...]:
        """Get the shape of an individual block."""
        return self._get_shape_blocks_full()[1:]

    @property
    def ndim(self) -> int:
        """Get the number of effective dimensions of the array."""
        return len(self.shape_block)

    @property
    def num_blocks(self) -> int:
        """Get the number of blocks in the array."""
        return self._get_shape_blocks_full()[0]

    def get_any_array(self):
        """Get an arbitrary (the first) block from the stack."""
        return self._blocks[0]

    def get_scalar_element(self):
        """Get the scalar element from a scalar block array."""
        if self.shape_block != ():
            raise ValueError("Array does not have scalar blocks.")
        return self._blocks[0]

    def is_zero(self, tol=1e-12):
        """Check if all blocks are zero up to a tolerance."""
        return ar.do("allclose", self.blocks, 0.0, atol=tol, like=self.backend)

    def get_params(self):
        """Interface for getting underlying arrays."""
        return self._blocks

    def _set_params_flatcommon(self, params):
        """Interface for setting underlying arrays."""
        self._blocks = params
        self.backend = ar.infer_backend(self._blocks)
        try:
            self._sectors = ar.do("array", self._sectors, like=params)
        except ImportError:
            # params is possibly a placeholder of some kind
            pass

    def apply_to_arrays(self, fn):
        self._blocks = fn(self._blocks)

    def item(self):
        """Convert the block array to a scalar if it is a scalar block array."""
        return self._blocks.item()

    def _binary_blockwise_op_abelian(
        self, other, fn, missing=None, inplace=False
    ):
        xy = self.sort_stack(inplace=inplace)
        other = other.sort_stack()

        # XXX: check matching sectors
        xy.modify(blocks=fn(xy._blocks, other._blocks))
        return xy

    def _do_reduction(self, fn):
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
        return fn(self._blocks)

    def norm(self):
        return ar.do("linalg.norm", self._blocks, like=self.backend)
