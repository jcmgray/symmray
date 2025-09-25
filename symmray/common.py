"""Common interfaces for all symmray array objects."""

import operator

import autoray as ar

from .utils import lazyabstractmethod

_symmray_namespace = None


class SymmrayCommon:
    """Common functionality for all symmray array like objects."""

    def __array_namespace__(self, api_version=None):
        """Return the namespace for the symmray module."""
        global _symmray_namespace
        if _symmray_namespace is None:
            import symmray

            _symmray_namespace = symmray
        return _symmray_namespace

    @lazyabstractmethod
    def copy(self) -> "SymmrayCommon":
        pass

    @lazyabstractmethod
    def copy_with(self, **kwargs) -> "SymmrayCommon":
        pass

    @lazyabstractmethod
    def modify(self, **kwargs) -> "SymmrayCommon":
        pass

    def _modify_or_copy(self, inplace=False, **kwargs) -> "SymmrayCommon":
        if inplace:
            return self.modify(**kwargs)
        else:
            return self.copy_with(**kwargs)

    @lazyabstractmethod
    def item(self):
        pass

    def __float__(self):
        return float(self.item())

    def __complex__(self):
        return complex(self.item())

    def __int__(self):
        return int(self.item())

    def __bool__(self):
        return bool(self.item())

    @lazyabstractmethod
    def _binary_blockwise_op(
        self, other, fn, missing=None, inplace=False
    ) -> "SymmrayCommon":
        pass

    def __add__(self, other):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other,
                fn=operator.add,
                missing="outer",
                inplace=False,
            )

        # addition with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Addition with {type(self)} and {type(other)} not implemented."
        )

    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other, fn=operator.add, missing="outer", inplace=True
            )

        # addition with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Addition with {type(self)} and {type(other)} not implemented."
        )

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other,
                fn=operator.sub,
                inplace=False,
            )

        # subtraction with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Subtraction with {type(self)} and {type(other)} not implemented."
        )

    def __isub__(self, other):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other, fn=operator.sub, inplace=True
            )

        # subtraction with non-matching block array breaks sparsity
        raise NotImplementedError(
            f"Subtraction with {type(self)} and {type(other)} not implemented."
        )

    @lazyabstractmethod
    def apply_to_arrays(self, fn):
        pass

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other, fn=operator.mul, missing="inner"
            )
        new = self.copy()
        new.apply_to_arrays(lambda x: x * other)
        return new

    def __imul__(self, other):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other, fn=operator.mul, missing="inner", inplace=True
            )
        self.apply_to_arrays(lambda x: x * other)
        return self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            if self.shape == other.shape and all(d == 1 for d in self.shape):
                return self._binary_blockwise_op(other, fn=operator.truediv)
            # deviding by implicit zeros not defined
            return NotImplemented

        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: x / other)
        return new

    def __itruediv__(self, other):
        if isinstance(other, self.__class__):
            # deviding by implicit zeros not defined
            return NotImplemented

        # assume scalar
        self.apply_to_arrays(lambda x: x / other)
        return self

    def __neg__(self) -> "SymmrayCommon":
        new = self.copy()
        new.apply_to_arrays(operator.neg)
        return new

    def _do_unary_op(self, fn, inplace=False) -> "SymmrayCommon":
        """Perform a unary operation on blocks of the array."""
        new = self if inplace else self.copy()
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
        new.apply_to_arrays(fn)
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
        new.apply_to_arrays(lambda x: _clip(x, a_min, a_max))
        return new

    @lazyabstractmethod
    def _do_reduction(self, fn):
        pass

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
