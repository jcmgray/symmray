import operator

from .common import SymmrayCommon
from .utils import DEBUG


class VectorCommon:
    def __add__(self, other, inplace=False):
        if isinstance(other, self.__class__):
            # can directly add matching vectors
            return self._binary_blockwise_op(
                other, fn=operator.add, missing="outer", inplace=inplace
            )

        if isinstance(other, SymmrayCommon):
            # cannot add to symmray objects:
            # sparsity not preserved or incompatible backend
            # XXX: possibly support 0-dimensional symmray objects?
            return NotImplemented

        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Addition {self} + {other} not supported.")

        # assume scalar
        new = self if inplace else self.copy()
        new.apply_to_arrays(lambda x: x + other)
        return new

    def __iadd__(self, other):
        return self.__add__(other, inplace=True)

    def __radd__(self, other):
        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Addition {other} + {self} not supported.")
        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: other + x)
        return new

    def __sub__(self, other, inplace=False):
        if isinstance(other, self.__class__):
            # can directly subtract matching vectors
            return self._binary_blockwise_op(
                other, fn=operator.sub, inplace=inplace
            )

        if isinstance(other, SymmrayCommon):
            # block structure not preserved
            # XXX: possibly support 0-dimensional symmray objects?
            return NotImplemented

        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Subtraction {self} - {other} not supported.")

        # assume scalar
        new = self if inplace else self.copy()
        new.apply_to_arrays(lambda x: x - other)
        return new

    def __isub__(self, other):
        return self.__sub__(other, inplace=True)

    def __rsub__(self, other):
        # assume scalar
        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Subtraction {other} - {self} not supported.")
        new = self.copy()
        new.apply_to_arrays(lambda x: other - x)
        return new

    def __truediv__(self, other, inplace=False):
        if isinstance(other, self.__class__):
            # can directly divide matching vectors
            return self._binary_blockwise_op(
                other, fn=operator.truediv, inplace=inplace
            )

        if isinstance(other, SymmrayCommon):
            # deviding by implicit zeros not defined
            # XXX: possibly support 0-dimensional symmray objects?
            return NotImplemented

        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Division {self} / {other} not supported.")

        # assume scalar
        new = self if inplace else self.copy()
        new.apply_to_arrays(lambda x: x / other)
        return new

    def __itruediv__(self, other):
        return self.__truediv__(other, inplace=True)

    def __rtruediv__(self, other):
        # assume scalar
        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Division {other} / {self} not supported.")
        new = self.copy()
        new.apply_to_arrays(lambda x: other / x)
        return new

    def __pow__(self, other, inplace=False):
        if isinstance(other, self.__class__):
            return self._binary_blockwise_op(
                other, fn=operator.pow, inplace=inplace
            )

        if isinstance(other, SymmrayCommon):
            # exponentiation with symmray objects not defined
            # XXX: possibly support 0-dimensional symmray objects?
            return NotImplemented

        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Exponential {self} ** {other} not supported.")

        # assume scalar
        new = self if inplace else self.copy()
        new.apply_to_arrays(lambda x: x**other)
        return new

    def __ipow__(self, other):
        return self.__pow__(other, inplace=True)

    def __rpow__(self, other):
        if DEBUG and getattr(other, "ndim", 0) != 0:
            raise ValueError(f"Exponential {other} ** {self} not supported.")
        # assume scalar
        new = self.copy()
        new.apply_to_arrays(lambda x: other**x)
        return new
