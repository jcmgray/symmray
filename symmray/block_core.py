"""Basic classes and functions for array like objects consisting of blocks."""

import functools
import operator

import autoray as ar


def _identity(x):
    return x


class BlockBase:
    """Mixin class for arrays consisting of dicts of blocks."""

    __slots__ = ("_blocks",)

    def __init__(self, blocks):
        self._blocks = dict(blocks)

    def copy(self):
        new = self.__class__(self.blocks)
        return new

    def copy_with(self, blocks=None):
        if blocks is None:
            return self.copy()
        else:
            return self.__class__(blocks)

    @property
    def blocks(self):
        """The blocks of the array."""
        return self._blocks

    def _map_blocks(self, fn_block=None, fn_sector=None):
        """Map the blocks and their keys (sectors) of the array inplace."""
        if fn_block is None:
            fn_block = _identity

        if fn_sector is None:
            fn_sector = _identity

        self._blocks = {
            fn_sector(sector): fn_block(block)
            for sector, block in self._blocks.items()
        }

    def get_any_array(self):
        """Get any array from the blocks, to check type and backend for
        example.
        """
        return next(iter(self._blocks.values()), 0.0)

    @property
    def dtype(self):
        """Get the dtype name from an arbitrary block in the array."""
        return ar.get_dtype_name(self.get_any_array())

    @property
    def backend(self):
        """Get the backend name from an arbitrary block in the array."""
        return ar.infer_backend(self.get_any_array())

    @property
    def num_blocks(self):
        """The number of blocks in the array."""
        return len(self._blocks)

    @property
    def sectors(self):
        """Get the sectors, i.e. keys of the blocks."""
        return tuple(self._blocks.keys())

    def get_params(self):
        """Get the parameters of this block array as a pytree (dict).

        Returns
        -------
        dict[tuple, array_like]
        """
        return self.blocks.copy()

    def set_params(self, params):
        """Set the parameters of this block array from a pytree (dict).

        Parameters
        ----------
        params : dict[tuple, array_like]
        """
        self.blocks.update(params)

    def apply_to_arrays(self, fn):
        """Apply the ``fn`` inplace to the array of every block."""
        for sector, array in self._blocks.items():
            self._blocks[sector] = fn(array)

    def item(self):
        """Convert the block array to a scalar if it is a scalar block array."""
        (array,) = self.blocks.values()
        return array.item()

    def __float__(self):
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    def _binary_blockwise_op(self, other, fn, inplace=False):
        """Apply a binary blockwise operation to two block arrays, which must
        have exactly the same sectors/keys.

        Parameters
        ----------
        fn : callable
            Function to apply to the blocks of the arrays, with signature
            ``fn(x_block, y_block) -> result_block``.
        x : BlockBase
            First block array.
        y : BlockBase
            Second block array.
        inplace : bool, optional
            Whether to modify the first array in place. Default is False.

        Returns
        -------
        BlockBase
        """
        xy = self if inplace else self.copy()

        xy_blocks = xy.blocks
        other_blocks = other.blocks.copy()

        for sector, x_block in xy_blocks.items():
            other_block = other_blocks.pop(sector)
            xy_blocks[sector] = fn(x_block, other_block)

        if other_blocks:
            raise ValueError(
                f"other has blocks not present in x: {other_blocks.keys()}"
            )

        return xy

    def __add__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(other, fn=operator.add)

        # addition with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Addition with {type(other)} not implemented."
        )

    def __iadd__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(
                other, fn=operator.add, inplace=True
            )

        # addition with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Addition with {type(other)} not implemented."
        )

    def __sub__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(other, fn=operator.sub)

        # subtraction with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Subtraction with {type(other)} not implemented."
        )

    def __isub__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(
                other, fn=operator.sub, inplace=True
            )

        # subtraction with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Subtraction with {type(other)} not implemented."
        )

    def __mul__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(other, fn=operator.mul)
        new = self.copy()
        new.apply_to_arrays(lambda x: x * other)
        return new

    def __imul__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(
                other, fn=operator.mul, inplace=True
            )
        self.apply_to_arrays(lambda x: x * other)
        return self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, BlockBase):
            if self.shape == other.shape and all(d == 1 for d in self.shape):
                return self._binary_blockwise_op(other, fn=operator.truediv)
            # deviding by implicit zeros not defined
            return NotImplemented

        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: x / other)
        return new

    def __itruediv__(self, other):
        if isinstance(other, BlockBase):
            # deviding by implicit zeros not defined
            return NotImplemented

        # assume scalar
        self.apply_to_arrays(lambda x: x / other)
        return self

    def __neg__(self):
        new = self.copy()
        new.apply_to_arrays(lambda x: -x)
        return new

    def _do_reduction(self, fn):
        """Perform an (associative) reduction operation on blocks of the array."""
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
            _stack = ar.get_lib_fn(self.backend, "stack")
        block_results = tuple(map(fn, self.blocks.values()))
        return fn(_stack(block_results))

    def max(self):
        """Get the maximum element from any block in the array."""
        return self._do_reduction("max")

    def min(self):
        """Get the minimum element from any block in the array."""
        return self._do_reduction("min")

    def sum(self):
        """Get the sum of all elements in the array."""
        return self._do_reduction("sum")

    def all(self):
        """Check if all elements in the array are True."""
        return self._do_reduction("all")

    def any(self):
        """Check if any element in the array is True."""
        return self._do_reduction("any")

    def _do_unary_op(self, fn, inplace=False):
        """Perform a unary operation on blocks of the array."""
        new = self if inplace else self.copy()
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
        new.apply_to_arrays(fn)
        return new

    def isfinite(self):
        """Check if all elements in the array are finite."""
        return self._do_unary_op("isfinite")

    def abs(self):
        """Get the absolute value of all elements in the array."""
        return self._do_unary_op("abs")

    def norm(self):
        """Get the frobenius norm of the block array."""
        backend = self.backend
        _sum = ar.get_lib_fn(backend, "sum")
        _abs = ar.get_lib_fn(backend, "abs")
        return (
            functools.reduce(
                operator.add,
                (_sum(_abs(x) ** 2) for x in self.blocks.values()),
            )
            ** 0.5
        )

    def __repr__(self):
        return "".join(
            [
                f"{self.__class__.__name__}(",
                f"total_size={self.size}, ",
                f"num_blocks={self.num_blocks})",
            ]
        )


class BlockVector(BlockBase):
    """A vector stored as a dict of blocks."""

    __slots__ = BlockBase.__slots__

    ndim = 1

    @property
    def size(self):
        """The total size of the arrays blocks."""
        # compute lazily
        _size = ar.get_lib_fn(self.backend, "size")
        return sum(_size(x) for x in self.blocks.values())

    @property
    def shape(self):
        return (self.size,)

    def __add__(self, other):
        if isinstance(other, BlockVector):
            return self._binary_blockwise_op(other, fn=operator.add)

        if isinstance(other, BlockBase):
            # block structure not preserved
            return NotImplemented

        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: x + other)
        return new

    def __iadd__(self, other):
        if isinstance(other, BlockVector):
            return self._binary_blockwise_op(
                other, fn=operator.add, inplace=True
            )

        if isinstance(other, BlockBase):
            # block structure not preserved
            return NotImplemented

        # assume scalar
        self.apply_to_arrays(lambda x: x + other)
        return self

    def __radd__(self, other):
        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: other + x)
        return new

    def __sub__(self, other):
        if isinstance(other, BlockVector):
            return self._binary_blockwise_op(other, fn=operator.sub)

        if isinstance(other, BlockBase):
            # block structure not preserved
            return NotImplemented

        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: x - other)
        return new

    def __isub__(self, other):
        if isinstance(other, BlockVector):
            return self._binary_blockwise_op(
                other, fn=operator.sub, inplace=True
            )

        if isinstance(other, BlockBase):
            # block structure not preserved
            return NotImplemented

        # assume scalar
        self.apply_to_arrays(lambda x: x - other)
        return self

    def __rsub__(self, other):
        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: other - x)
        return new

    def __truediv__(self, other):
        if isinstance(other, BlockVector):
            return self._binary_blockwise_op(other, fn=operator.truediv)

        if isinstance(other, BlockBase):
            # deviding by implicit zeros not defined
            return NotImplemented

        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: x / other)
        return new

    def __itruediv__(self, other):
        if isinstance(other, BlockVector):
            return self._binary_blockwise_op(
                other, fn=operator.truediv, inplace=True
            )

        if isinstance(other, BlockBase):
            # deviding by implicit zeros not defined
            return NotImplemented

        # assume scalar
        self.apply_to_arrays(lambda x: x / other)
        return self

    def __rtruediv__(self, other):
        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: other / x)
        return new

    def __pow__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(other, fn=operator.pow)

        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: x**other)
        return new

    def __ipow__(self, other):
        if isinstance(other, BlockBase):
            return self._binary_blockwise_op(
                other, fn=operator.pow, inplace=True
            )

        # assume scalar
        self.apply_to_arrays(lambda x: x**other)
        return self

    def __rpow__(self, other):
        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: other**x)
        return new

    def check(self):
        """Check that the block vector is well formed."""
        ndims = {ar.ndim(x) for x in self.blocks.values()}
        if len(ndims) != 1:
            raise ValueError(f"blocks have different ndims: {ndims}")
        assert self.size == sum(ar.size(s) for s in self.blocks.values())

    def to_dense(self):
        """Convert the block vector to a dense array."""
        arrays = tuple(self.blocks[k] for k in sorted(self.blocks))
        _concatenate = ar.get_lib_fn(self.backend, "concatenate")
        return _concatenate(arrays, axis=0)
