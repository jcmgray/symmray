"""Flat backend block vector class."""

import numbers

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
        self._init_flatcommon(sectors, blocks)
        if DEBUG:
            self.check()

    @classmethod
    def from_blocks(cls, blocks):
        """Create a flat vector from a dictionary of blocks.

        Parameters
        ----------
        blocks : dict
            Dictionary mapping sector (charge) to block (array).
        """
        new_blocks = ar.do("stack", tuple(blocks.values()))
        sectors = ar.do("array", tuple(blocks.keys()), like=new_blocks)
        return cls(sectors, new_blocks)

    @classmethod
    def from_blockvector(cls, blockvector):
        """Create a flat backend vector from a sparse backend BlockVector.

        Parameters
        ----------
        blockvector : BlockVector
            The BlockVector to convert.
        """
        return cls.from_blocks(blockvector.blocks)

    def to_blockvector(self):
        """Convert to a sparse backend BlockVector."""
        from ..sparse.sparse_vector import BlockVector

        return BlockVector(
            {k.item(): b for k, b in zip(self.sectors, self.blocks)}
        )

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

        if isinstance(sectors, numbers.Integral):
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
        return self._copy_flatcommon(deep=deep)

    def copy_with(self, sectors=None, blocks=None) -> "FlatVector":
        """Create a copy of the vector with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self._copy_with_flatcommon(sectors=sectors, blocks=blocks)
        if DEBUG:
            new.check()
        return new

    def modify(self, sectors=None, blocks=None):
        """Modify the vector in place with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        self._modify_flatcommon(sectors=sectors, blocks=blocks)
        if DEBUG:
            self.check()
        return self

    def check(self):
        assert ar.do("ndim", self._blocks, like=self.backend) == 2

    def sort_stack(self, inplace=False):
        """Sort the sectors and associated blocks in ascending order of
        charge.
        """
        k = ar.do("argsort", self._sectors, like=self.backend)
        return self._modify_or_copy(
            sectors=self._sectors[k],
            blocks=self._blocks[k],
            inplace=inplace,
        )

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        return self._binary_blockwise_op_abelian(
            other, fn, missing=missing, inplace=inplace
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
