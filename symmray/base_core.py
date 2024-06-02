import functools
import operator

import autoray as ar


class DictArray:
    """Mixin class for arrays consisting of dicts of blocks."""

    __slots__ = ("_blocks", "_size")

    def __init__(self, blocks):
        self._blocks = dict(blocks)
        self._size = None

    def copy(self):
        new = self.__class__(self.blocks)
        new._size = self._size
        return new

    @property
    def blocks(self):
        """The blocks of the array."""
        return self._blocks

    @property
    def size(self):
        """The total size of the arrays blocks."""
        # compute lazily
        if self._size is None:
            self._size = sum(ar.size(s) for s in self._blocks.values())
        return self._size

    def get_any_array(self):
        """Get any array from the blocks, to check type and backend for
        example.
        """
        return next(iter(self._blocks.values()))

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

    def __mul__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: x * other)
        return new

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: x / other)
        return new

    def __rtruediv__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: other / x)
        return new

    def __pow__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: x ** other)
        return new

    def __rpow__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: other ** x)
        return new

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


class BlockVector(DictArray):
    """A vector stored as a dict of blocks."""

    __slots__ = DictArray.__slots__

    ndim = 1

    @property
    def shape(self):
        return (self.size,)

    # def reshape(self, newshape):
    #     new = self.copy()
    #     new.apply_to_arrays(lambda x: ar.reshape(x, newshape))
    #     return new

    def __add__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: x + other)
        return new

    def __sub__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: x - other)
        return new

    def __radd__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: other + x)
        return new

    def __rsub__(self, other):
        new = self.copy()
        new.apply_to_arrays(lambda x: other - x)
        return new

    def to_dense(self):
        """Convert the block vector to a dense array."""
        arrays = tuple(self.blocks[k] for k in sorted(self.blocks))
        return ar.do("concatenate", arrays, axis=0)