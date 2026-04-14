"""Abelian symmetric arrays with block sparse backend."""

from ..array_common import ArrayCommon
from ..bosonic_common import BosonicCommon
from ..common import SymmrayCommon
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .sparse_array_common import SparseArrayCommon
from .sparse_data_common import BlockCommon
from .sparse_index import BlockIndex


class AbelianArray(
    SparseArrayCommon,
    BlockCommon,
    BosonicCommon,
    ArrayCommon,
    SymmrayCommon,
):
    """A block sparse array with symmetry constraints.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices of the array.
    charge : hashable, optionals
        The total charge of the array, if not given it will be inferred from
        either the first sector or set to the identity charge, if no sectors
        are given.
    blocks : dict[tuple[hashable], array_like]
        A mapping of each 'sector' (tuple of charges) to the data array.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = (
        "_blocks",
        "_charge",
        "_indices",
        "_label",
        "_symmetry",
    )
    fermionic = False
    static_symmetry = None

    def __init__(
        self,
        indices,
        charge=None,
        blocks=(),
        symmetry=None,
        label=None,
    ):
        self._init_abelian(
            indices=indices,
            charge=charge,
            blocks=blocks,
            symmetry=symmetry,
            label=label,
        )

        if DEBUG:
            self.check()

    def new_with(self, indices, charge, blocks, label=None) -> "AbelianArray":
        """Create a new block sparse abelian array of the same class as this
        one. Unlike `copy`, this does not copy over any existing data and drops
        for example `label`.
        """
        return self._new_with_abelian(
            indices=indices,
            charge=charge,
            blocks=blocks,
            label=label,
        )

    def copy(self):
        """Copy this abelian block sparse array."""
        return self._copy_abelian()

    def copy_with(self, indices=None, charge=None, blocks=None):
        """A copy of this block array with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self._copy_with_abelian(
            indices=indices,
            charge=charge,
            blocks=blocks,
        )
        if DEBUG:
            new.check()
        return new

    def modify(self, indices=None, charge=None, blocks=None):
        """Modify this block array in place with some attributes replaced. Note
        that checks are not performed on the new properties, this is intended
        for internal use.
        """
        self._modify_abelian(indices=indices, charge=charge, blocks=blocks)
        if DEBUG:
            self.check()
        return self

    @classmethod
    def from_pytree(cls, pytree):
        """Create a sparse abelian array from a pytree purely of non-symmray
        containers and objects.
        """
        indices = tuple(map(BlockIndex.from_pytree, pytree["indices"]))
        return cls(
            indices=indices,
            charge=pytree["charge"],
            blocks=pytree["blocks"],
            symmetry=pytree["symmetry"],
            label=pytree["label"],
        )

    def _map_blocks(self, fn_block=None, fn_sector=None, fn_filter=None):
        """Map the blocks and their keys (sectors) of the array inplace."""
        self._map_blocks_blockcommon(fn_block, fn_sector, fn_filter)

    # --------------------------- linalg methods ---------------------------- #

    def solve(self, b: "AbelianArray", **kwargs) -> "AbelianArray":
        """Solve the linear system `A @ x == b` for x, where A is this array.

        Parameters
        ----------
        b : AbelianArray
            The right-hand side array.

        Returns
        -------
        x : AbelianArray
            The solution array.
        """
        return self._solve_abelian(b, **kwargs)


# --------------------------------------------------------------------------- #


class Z2Array(AbelianArray):
    """A block array with Z2 symmetry."""

    static_symmetry = get_symmetry("Z2")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.core import SparseTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import Z2

        blocks = [
            SubTensor(array, q_labels=tuple(map(Z2, sector)))
            for sector, array in self.get_sector_block_pairs()
        ]

        data = SparseTensor(blocks)

        if flat:
            from pyblock3.algebra.flat import FlatSparseTensor

            data = FlatSparseTensor.from_sparse(data)

        data.shape = self.shape

        return data


class U1Array(AbelianArray):
    """A block array with U1 symmetry."""

    static_symmetry = get_symmetry("U1")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.core import SparseTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import U1

        blocks = [
            SubTensor(array, q_labels=tuple(map(U1, sector)))
            for sector, array in self.get_sector_block_pairs()
        ]

        data = SparseTensor(blocks)

        if flat:
            from pyblock3.algebra.flat import FlatSparseTensor

            data = FlatSparseTensor.from_sparse(data)

        data.shape = self.shape

        return data

    def to_yastn(self, **config_opts):
        import yastn

        t = yastn.Tensor(
            config=yastn.make_config(sym="U1", **config_opts),
            s=tuple(-1 if ix.dual else 1 for ix in self.indices),
            n=self.charge,
        )
        for sector, array in self.get_sector_block_pairs():
            t.set_block(ts=sector, Ds=array.shape, val=array)

        return t


class Z2Z2Array(AbelianArray):
    """A block array with Z2 x Z2 symmetry."""

    static_symmetry = get_symmetry("Z2Z2")


class U1U1Array(AbelianArray):
    """A block array with U1 x U1 symmetry."""

    static_symmetry = get_symmetry("U1U1")
