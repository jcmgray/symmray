"""Sparse backend block vector class."""

import autoray as ar

from ..common import SymmrayCommon
from ..vector_common import VectorCommon
from .sparse_data_common import BlockCommon


class BlockVector(BlockCommon, VectorCommon, SymmrayCommon):
    """Class for storing block vectors with sparse storage, e.g. for the
    singular- or eigen- values of a matrix.

    This is equivalent to the diagonal of a zero charge abelian matrix.

    Parameters
    ----------
    blocks : dict
        A dictionary mapping sector keys to array blocks.
    """

    __slots__ = ("_blocks",)
    ndim = 1

    def __init__(self, blocks):
        self._init_blockcommon(blocks)

    def copy(self):
        return self._copy_blockcommon()

    @property
    def size(self):
        """The total size of all elements in the vector."""
        # compute lazily
        _size = ar.get_lib_fn(self.backend, "size")
        return sum(_size(x) for x in self.get_all_blocks())

    @property
    def shape(self):
        """Get the effective shape of the vector."""
        return (self.size,)

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        return self._binary_blockwise_op_abelian(
            other, fn, missing=missing, inplace=inplace
        )

    def check(self):
        """Check that the block vector is well formed."""
        ndims = {ar.ndim(x) for x in self.get_all_blocks()}
        if len(ndims) != 1:
            raise ValueError(f"blocks have different ndims: {ndims}")
        assert self.size == sum(ar.size(s) for s in self.get_all_blocks())

    def to_flat(self):
        """Convert the block sparse vector to a flat vector."""
        from ..flat.flat_vector import FlatVector

        return FlatVector.from_blocks(self.blocks)

    def to_dense(self):
        """Convert the block vector to a dense array."""
        arrays = tuple(map(self.get_block, sorted(self.sectors)))
        _concatenate = ar.get_lib_fn(self.backend, "concatenate")
        return _concatenate(arrays, axis=0)

    def allclose(self, other, **allclose_opts):
        return self._allclose_blockcommon(other, **allclose_opts)

    def test_allclose(self, other, **allclose_opts):
        """Like `allclose` but raises an AssertionError with details if not
        close."""
        return self._test_allclose_blockcommon(other, **allclose_opts)

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"total_size={self.size}, ",
                f"num_blocks={self.num_blocks})",
            ]
        )
