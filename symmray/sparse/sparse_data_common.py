"""Basic classes and functions for block sparse backend objects."""

import functools
import operator

import autoray as ar

from ..utils import lazyabstractmethod


def _identity(x):
    return x


class BlockCommon:
    """Mixin class for arrays consisting of dicts of blocks."""

    __slots__ = ("_blocks",)

    def _init_blockcommon(self, blocks):
        self._blocks = dict(blocks)

    def _new_with_blockcommon(self, blocks):
        new = self.__new__(self.__class__)
        new._blocks = blocks
        return new

    def _copy_blockcommon(self):
        new = self.__new__(self.__class__)
        new._blocks = self._blocks.copy()
        return new

    @lazyabstractmethod
    def copy(self):
        pass

    def _copy_with_blockcommon(self, blocks=None):
        new = self.__new__(self.__class__)
        new._blocks = self._blocks.copy() if blocks is None else blocks
        return new

    @lazyabstractmethod
    def copy_with(self, blocks=None):
        pass

    def _modify_blockcommon(self, blocks=None):
        if blocks is not None:
            self._blocks = blocks
        return self

    @lazyabstractmethod
    def modify(self, blocks=None):
        pass

    @property
    def blocks(self):
        """The blocks of the array."""
        return self._blocks

    def get_block(self, sector):
        """Get the block for the given sector."""
        return self._blocks[sector]

    def set_block(self, sector, block):
        """Set the block for the given sector."""
        self._blocks[sector] = block

    def del_block(self, sector):
        """Delete the block for the given sector."""
        del self._blocks[sector]

    def get_sector_block_pairs(self):
        """Get an iterator over all `(sector, block)` pairs."""
        return self._blocks.items()

    def get_all_blocks(self):
        """Get an iterator over all blocks of the array."""
        return self._blocks.values()

    def _map_blocks_blockcommon(
        self,
        fn_block=None,
        fn_sector=None,
        fn_filter=None,
    ):
        """Map the blocks and their keys (sectors) of the array inplace."""
        if fn_block is None:
            fn_block = _identity

        if fn_sector is None:
            fn_sector = _identity

        if fn_filter is None:

            def fn_filter(sector):
                return True

        self._blocks = {
            fn_sector(sector): fn_block(block)
            for sector, block in self.get_sector_block_pairs()
            if fn_filter(sector)
        }

    @lazyabstractmethod
    def _map_blocks(self, fn_block=None, fn_sector=None, fn_filter=None):
        pass

    def get_any_sector(self):
        return next(iter(self._blocks.keys()), ())

    def get_any_array(self):
        """Get any array from the blocks, to check type and backend for
        example.
        """
        return next(iter(self.get_all_blocks()), 0.0)

    def get_scalar_element(self):
        """Assuming the block array is a scalar, get that scalar element."""
        sectors = self.sectors

        if sectors == ((),):
            # single scalar block, usual case
            return self.get_block(())
        elif sectors == ():
            # no aligned blocks, return zero
            return 0.0
        else:
            raise ValueError("Block array does not seem to be a scalar.")

    def is_zero(self, tol=1e-12):
        """Check if all blocks are zero up to a tolerance."""
        return all(
            ar.do("allclose", b, 0.0, atol=tol) for b in self.get_all_blocks()
        )

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

    def has_sector(self, sector):
        """Check if the array has a block for the given sector."""
        return sector in self._blocks

    def get_params(self):
        """Get the parameters of this block array as a pytree (dict).

        Returns
        -------
        dict[tuple, array_like]
        """
        return self._blocks.copy()

    def set_params(self, params):
        """Set the parameters of this block array from a pytree (dict).

        Parameters
        ----------
        params : dict[tuple, array_like]
        """
        self._blocks.update(params)

    def apply_to_arrays(self, fn):
        """Apply the ``fn`` inplace to the array of every block."""
        for sector, array in self.get_sector_block_pairs():
            self.set_block(sector, fn(array))

    def item(self):
        """Convert the block array to a scalar if it is a scalar block array."""
        (array,) = self.get_all_blocks()
        return array.item()

    def _binary_blockwise_op_abelian(
        self, other, fn, missing=None, inplace=False
    ):
        """Apply a binary blockwise operation to two block arrays, which must
        have exactly the same sectors/keys, depending on `missing`.

        Parameters
        ----------
        fn : callable
            Function to apply to the blocks of the arrays, with signature
            ``fn(x_block, y_block) -> result_block``.
        x : BlockCommon
            First block array.
        y : BlockCommon
            Second block array.
        missing : str, optional
            How to handle missing blocks. Can be "outer", "inner" or None.
            Default is None which requires 1:1 matching blocks. If missing
            "outer" is specified, blocks present in only one of either array
            are simply kept. If missing "inner" is specified, blocks present
            in only one of either array are simply dropped.
        inplace : bool, optional
            Whether to modify the first array in place. Default is False.

        Returns
        -------
        BlockCommon
        """
        xy = self if inplace else self.copy()

        xy_blocks = xy.blocks
        other_blocks = other.blocks.copy()

        if missing is None:
            # by default require 1:1 matching blocks

            for sector, x_block in xy_blocks.items():
                try:
                    other_block = other_blocks.pop(sector)
                except KeyError:
                    raise ValueError(
                        f"Not all left blocks present in right: {sector}"
                    )
                xy_blocks[sector] = fn(x_block, other_block)

            if other_blocks:
                raise ValueError(
                    f"Not all right blocks present in left: {other_blocks}."
                )

        elif missing == "outer":
            # blocks present in only one of either array are simply kept

            for sector, x_block in xy_blocks.items():
                if sector in other_blocks:
                    # both blocks present -> combine with fn
                    other_block = other_blocks.pop(sector)
                    xy_blocks[sector] = fn(x_block, other_block)
                else:
                    # only left present
                    xy_blocks[sector] = x_block

            # add remaining blocks only present in right
            xy_blocks.update(other_blocks)

        elif missing == "inner":
            # blocks present in only one of either array are simply dropped

            for sector, x_block in xy_blocks.items():
                if sector in other_blocks:
                    other_block = other_blocks.pop(sector)
                    xy_blocks[sector] = fn(x_block, other_block)

        return xy

    def _do_reduction(self, fn):
        """Perform an (associative) reduction operation on blocks of the array."""
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
            _stack = ar.get_lib_fn(self.backend, "stack")
        block_results = tuple(map(fn, self.get_all_blocks()))
        return fn(_stack(block_results))

    def norm(self):
        """Get the frobenius norm of the block array."""
        backend = self.backend
        _sum = ar.get_lib_fn(backend, "sum")
        _abs = ar.get_lib_fn(backend, "abs")
        return (
            functools.reduce(
                operator.add,
                (_sum(_abs(x) ** 2) for x in self.get_all_blocks()),
            )
            ** 0.5
        )

    def _allclose_blockcommon(self, other, **allclose_opts):
        _allclose = ar.get_lib_fn(self.backend, "allclose")

        # all shared blocks must be close
        shared = self.blocks.keys() & other.blocks.keys()
        for sector in shared:
            if not _allclose(
                self.get_block(sector),
                other.get_block(sector),
                **allclose_opts,
            ):
                return False

        # all missing blocks must be zero
        left = self.blocks.keys() - other.blocks.keys()
        right = other.blocks.keys() - self.blocks.keys()
        for sector in left:
            if not _allclose(self.get_block(sector), 0.0, **allclose_opts):
                return False
        for sector in right:
            if not _allclose(other.get_block(sector), 0.0, **allclose_opts):
                return False

        return True

    @lazyabstractmethod
    def allclose(self, other, **allclose_opts):
        pass

    def _test_allclose_blockcommon(self, other, **allclose_opts):
        """Like `allclose` but raises an AssertionError with details if not
        close."""
        _allclose = ar.get_lib_fn(self.backend, "allclose")

        # all shared blocks must be close
        shared = self.blocks.keys() & other.blocks.keys()
        for sector in shared:
            if not _allclose(
                self.get_block(sector),
                other.get_block(sector),
                **allclose_opts,
            ):
                raise AssertionError(
                    f"Block arrays are not allclose for sector {sector}:\n"
                    f" Left  block:\n{self.get_block(sector)}\n"
                    f" Right block:\n{other.get_block(sector)}\n"
                )

        # all missing blocks must be zero
        left = self.blocks.keys() - other.blocks.keys()
        right = other.blocks.keys() - self.blocks.keys()
        for sector in left:
            if not _allclose(self.get_block(sector), 0.0, **allclose_opts):
                raise AssertionError(
                    f"Left block only is not zero for {sector}:\n"
                    f" Left  block:\n{self.get_block(sector)}\n"
                    f" Right block:\n<missing>\n"
                )
        for sector in right:
            if not _allclose(other.get_block(sector), 0.0, **allclose_opts):
                raise AssertionError(
                    f"Right block only is not zero for {sector}:\n"
                    f" Left  block:\n<missing>\n"
                    f" Right block:\n{other.get_block(sector)}\n"
                )

        return True

    @lazyabstractmethod
    def test_allclose(self, other, **allclose_opts):
        pass
