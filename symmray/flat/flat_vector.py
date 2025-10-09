"""Flat backend block vector class."""

import autoray as ar

from ..common import SymmrayCommon
from ..utils import DEBUG
from ..vector_common import VectorCommon
from .flat_data_common import FlatCommon


class FlatVector(FlatCommon, VectorCommon, SymmrayCommon):
    """Class for storing block vectors with flat storage, e.g. for the
    singular- or eigen- values of a matrix.

    This is equivalent to the diagonal of a zero charge abelian matrix.

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

    @classmethod
    def from_fill_fn(cls, fill_fn, sectors, charge_size):
        """Create a flat vector by filling blocks using a function that takes
        a shape and returns an array.

        Parameters
        ----------
        fill_fn : callable
            Function that takes a shape tuple and returns an array of that
            shape.
        sectors : array_like or int or str
            The vector of charges, with shape (num_blocks,). If an integer
            is given, it is assumed to be Z{N} symmetry and the sectors are
            set to range(N). If a string is given, it is assumed to be of the
            form "ZN" for some integer N.
        charge_size : int
            The size of each charge block.
        like : str or array_like, optional

        """
        if isinstance(sectors, str):
            # assume {ZN} symmetry -> turn into int
            import re

            m = re.match(r"Z(\d+)", sectors)
            if m is None:
                raise ValueError(f"Invalid symmetry string: {sectors}")
            sectors = int(m[1])

        if isinstance(sectors, int):
            # integer specifying Z{N} symmetry
            n = sectors
            create_sectors = True
        else:
            n = len(sectors)
            create_sectors = False

        block_shape = (n, charge_size)
        blocks = fill_fn(block_shape)

        if create_sectors:
            sectors = ar.do("arange", n, like=blocks)

        return cls(sectors, blocks)

    @classmethod
    def rand(
        cls, sectors, charge_size, dist="normal", seed=None, like="numpy"
    ):
        rng = ar.do("random.default_rng", seed, like=like)

        if dist == "normal":

            def fill_fn(shape):
                return rng.normal(size=shape)

        elif dist == "uniform":

            def fill_fn(shape):
                return rng.uniform(size=shape)

        else:
            raise ValueError(f"Invalid distribution: {dist}")

        return cls.from_fill_fn(fill_fn, sectors, charge_size)

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
        from ..sparse.sparse_vector import BlockVector

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
