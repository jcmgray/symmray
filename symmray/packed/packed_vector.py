"""Packed backend block vector class."""

import autoray as ar

from ..common import SymmrayCommon
from ..vector_common import VectorCommon


class PackedVector(VectorCommon, SymmrayCommon):
    """Block vector with packed storage, e.g. for the singular values or
    eigenvalues of a matrix. Blocks are stored end to end in one buffer, in
    sorted charge order.

    Parameters
    ----------
    blocks : dict
        A mapping of each charge to a 1D block.
    """

    __slots__ = ("_charges", "_data", "_offsets", "_skey")
    ndim = 1

    def __init__(self, blocks):
        import numpy as np

        blocks = dict(blocks)
        charges = sorted(blocks)
        arrays = [ar.do("reshape", blocks[c], (-1,)) for c in charges]
        self._charges = tuple(charges)
        if arrays:
            self._data = np.concatenate(arrays)
        else:
            self._data = np.zeros(0)
        self._offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([ar.size(x) for x in arrays], out=self._offsets[1:])
        self._skey = None

    @classmethod
    def from_parts(cls, charges, data, offsets):
        """Create directly from ``charges``, the ``data`` buffer and the
        ``offsets`` of each block within it. ``charges`` must be sorted.
        """
        new = cls.__new__(cls)
        new._charges = tuple(charges)
        new._data = data
        new._offsets = offsets
        new._skey = None
        return new

    @classmethod
    def from_blockvector(cls, v):
        """Create from a block sparse `BlockVector`."""
        return cls(v.blocks)

    def to_blockvector(self):
        """Convert to a block sparse `BlockVector`."""
        from ..sparse.sparse_vector import BlockVector

        return BlockVector(self.blocks)

    def copy(self):
        new = self.from_parts(self._charges, self._data, self._offsets)
        new._skey = self._skey
        return new

    def structure_key(self):
        """A hashable key describing the block structure."""
        if self._skey is None:
            self._skey = (self._charges, self._offsets.tobytes())
        return self._skey

    @property
    def data(self):
        """The flat buffer holding every block."""
        return self._data

    @property
    def offsets(self):
        """Where each block starts within the data buffer."""
        return self._offsets

    @property
    def charges(self):
        """The charge of each block, sorted."""
        return self._charges

    @property
    def sectors(self):
        """The charge of each block, sorted."""
        return self._charges

    @property
    def blocks(self):
        """A mapping of each charge to a view of its block."""
        offsets = self._offsets.tolist()
        return {
            c: self._data[offsets[i] : offsets[i + 1]]
            for i, c in enumerate(self._charges)
        }

    def get_block(self, charge):
        i = self._charges.index(charge)
        return self._data[self._offsets[i] : self._offsets[i + 1]]

    def has_sector(self, charge):
        return charge in self._charges

    def get_sector_block_pairs(self):
        return self.blocks.items()

    def get_all_blocks(self):
        return self.blocks.values()

    @property
    def num_blocks(self):
        return len(self._charges)

    @property
    def size(self):
        """The total size of all elements in the vector."""
        return int(self._offsets[-1])

    @property
    def shape(self):
        return (self.size,)

    @property
    def dtype(self):
        return ar.get_dtype_name(self._data)

    @property
    def backend(self):
        return ar.infer_backend(self._data)

    def get_any_array(self):
        return self._data

    def get_namespace(self):
        return ar.get_namespace(self._data)

    def get_params(self):
        return {"data": self._data}

    def set_params(self, params):
        data = params["data"]
        if ar.infer_backend(data) != "numpy":
            raise TypeError(
                "The packed backend only supports numpy data, "
                f"got {ar.infer_backend(data)}."
            )
        self._data = data

    def apply_to_arrays(self, fn):
        """Apply ``fn`` inplace to all the data at once. ``fn`` should be
        elementwise, use `apply_to_blocks` otherwise.
        """
        self._data = fn(self._data)

    def apply_to_blocks(self, fn):
        """Apply ``fn`` inplace to the array of each block. ``fn`` need not be
        elementwise.
        """

        import numpy as np

        arrays = [np.asarray(fn(b)).reshape(-1) for b in self.blocks.values()]
        if arrays:
            self._data = np.concatenate(arrays)
            self._offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
            np.cumsum([x.size for x in arrays], out=self._offsets[1:])
            self._skey = None

    def _do_reduction_blockwise(self, fn):
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
        return fn(self._data)

    def _get_scalar_element_blockwise(self):
        if self.size == 1:
            return self._data[0]
        if self.size == 0:
            return 0.0
        raise ValueError("Vector does not seem to be a scalar.")

    def _item_blockwise(self):
        return self._data.item()

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        import numpy as np

        new = self if inplace else self.copy()
        if self._charges == other._charges and np.array_equal(
            self._offsets, other._offsets
        ):
            new._data = fn(self._data, other._data)
            return new

        # mismatched blocks, go via dicts
        from ..sparse.sparse_vector import BlockVector

        res = BlockVector(self.blocks)._binary_blockwise_op(
            BlockVector(other.blocks), fn, missing=missing
        )
        res = PackedVector(res.blocks)
        new._charges, new._data, new._offsets = (
            res._charges,
            res._data,
            res._offsets,
        )
        new._skey = None
        return new

    def norm(self):
        """Get the frobenius norm of the vector."""
        return ar.do("linalg.norm", self._data)

    def is_zero(self, tol=1e-12):
        return bool(ar.do("allclose", self._data, 0.0, atol=tol))

    def check(self):
        """Check that the vector is well formed."""
        assert list(self._charges) == sorted(self._charges)
        assert len(self._offsets) == len(self._charges) + 1
        assert self._offsets[-1] == ar.size(self._data)
        assert ar.ndim(self._data) == 1

    def to_dense(self):
        """Convert the block vector to a dense array, in sorted charge order."""
        return self._data

    def allclose(self, other, **allclose_opts):
        if isinstance(other, PackedVector):
            other = other.to_blockvector()
        return self.to_blockvector().allclose(other, **allclose_opts)

    def test_allclose(self, other, **allclose_opts):
        """Like `allclose` but raises an AssertionError with details if not
        close.
        """
        if isinstance(other, PackedVector):
            other = other.to_blockvector()
        return self.to_blockvector().test_allclose(other, **allclose_opts)

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"total_size={self.size}, ",
                f"num_blocks={self.num_blocks})",
            ]
        )
