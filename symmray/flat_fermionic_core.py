import autoray as ar

from .flat_abelian_core import AbelianArrayFlat
from .interface import tensordot
from .sparse_fermionic_core import FermionicArray
from .symmetries import get_symmetry


def perm_to_swaps(perm):
    """Convert a permutation to a list of swaps.

    Parameters
    ----------
    perm : tuple[int, ...]
        The permutation to convert. Should be a permutation of the elements in
        range(len(perm)).

    Returns
    -------
    tuple[tuple[int, int], ...]
        A tuple of swaps that will perform the permutation.
    """
    N = len(perm)
    current = list(range(N))
    desired = list(map(int, perm))
    idxs = {i: i for i in current}
    swaps = []

    for pos in range(N):
        x = desired[pos]
        ix = idxs[x]
        # ix = current.index(x)
        in_position = ix == pos
        while not in_position:
            # swap with item to left
            iy = ix - 1
            y = current[iy]
            # record the swap
            swaps.append((iy, ix))
            current[iy], current[ix] = x, y
            idxs[y], idxs[x] = ix, iy
            # check if we have finished this element
            ix = iy
            in_position = ix == pos

    return tuple(swaps)


class FermionicArrayFlat(AbelianArrayFlat):
    __slots__ = AbelianArrayFlat.__slots__ + ("_phases", "_oddpos")
    fermionic = True
    static_symmetry = None

    def __init__(
        self,
        sectors,
        blocks,
        indices,
        phases=None,
        oddpos=(),
        symmetry=None,
    ):
        AbelianArrayFlat.__init__(
            self,
            sectors=sectors,
            blocks=blocks,
            indices=indices,
            symmetry=symmetry,
        )

        if phases is None:
            self._phases = ar.do("ones", self.num_blocks, like=self._blocks)
        elif hasattr(phases, "shape"):
            self._phases = phases
        else:
            self._phases = ar.do("array", phases, like=self._blocks)

        assert not oddpos
        self._oddpos = ()

    @property
    def phases(self):
        """The lazy phases for each block."""
        return self._phases

    def check(self):
        """Check the internal consistency of the array."""
        AbelianArrayFlat.check(self)
        assert self._phases.shape == (self.num_blocks,)
        assert ar.do("all", ar.do("isin", self._phases, [-1, 1]))

    def copy(self, deep=False) -> "FermionicArrayFlat":
        """Create a copy of the array."""
        new = AbelianArrayFlat.copy(self, deep=deep)

        if deep:
            new._phases = ar.do("copy", self._phases, like=self.backend)
        else:
            new._phases = self._phases

        return new

    def copy_with(
        self,
        sectors=None,
        blocks=None,
        indices=None,
        phases=None,
    ) -> "FermionicArrayFlat":
        """A copy of this fermionic flat array with some attributes replaced.
        Note that checks are not performed on the new properties, this is
        intended for internal use.
        """
        new = AbelianArrayFlat.copy_with(
            self,
            sectors=sectors,
            blocks=blocks,
            indices=indices,
        )
        new._phases = self._phases if phases is None else phases
        return new

    def modify(
        self,
        sectors=None,
        blocks=None,
        indices=None,
        phases=None,
    ) -> "FermionicArrayFlat":
        """Modify this fermionic flat array in place with some attributes
        replaced. Note that checks are not performed on the new properties,
        this is intended for internal use.
        """
        AbelianArrayFlat.modify(self, sectors, blocks, indices)
        if phases is not None:
            self._phases = phases
        return self

    def _modify_or_copy(
        self,
        sectors=None,
        blocks=None,
        indices=None,
        phases=None,
        inplace=False,
    ) -> "FermionicArrayFlat":
        if inplace:
            return self.modify(sectors, blocks, indices, phases)
        else:
            return self.copy_with(sectors, blocks, indices, phases)

    @classmethod
    def from_blocks(
        cls,
        blocks,
        indices,
        phases=None,
        oddpos=None,
        symmetry=None,
    ) -> "FermionicArrayFlat":
        """Create a fermionic flat array from an explicit dictionary of blocks,
        and sequence of indices or duals.

        Parameters
        ----------
        blocks : dict[tuple[int, ...], array_like]
            A dictionary mapping sector keys (tuples of charges) to blocks
            (arrays).
        indices : sequence[FlatIndex] | sequence[bool]
            A sequence of indices describing the dualness and any subindex
            information for each dimension of the array. If bools are supplied,
            they will be converted to a FlatIndex with the corresponding
            dualness, and no subindex information.
        phases : dict[tuple[int, ...], int], optional
            A dictionary mapping sector keys to +/- 1 phases. If not supplied,
            all phases are assumed to be +1.
        """
        sectors = list(map(list, blocks.keys()))

        if phases is not None:
            phases = ar.do(
                "array",
                [phases.get(sector, 1) for sector in blocks.keys()],
                like=blocks,
            )

        if blocks:
            blocks = ar.do("stack", tuple(blocks.values()))
        else:
            blocks = []

        return cls(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
            phases=phases,
            oddpos=oddpos,
            symmetry=symmetry,
        )

    @classmethod
    def from_blocksparse(
        cls,
        x: "FermionicArray",
        symmetry=None,
    ):
        """Create a fermionic flat array from a fermionic blocksparse array.

        Parameters
        ----------
        x : FermionicArray
            The fermionic blocksparse array to convert.
        symmetry : FermionicSymmetry, optional
            The symmetry to use. If not supplied, the symmetry of `x` is used.
        """
        return cls.from_blocks(
            blocks=x.blocks,
            indices=x.duals,
            phases=x.phases,
            oddpos=x.oddpos,
            symmetry=symmetry or x.symmetry,
        )

    def to_blocksparse(self) -> "FermionicArray":
        """Convert this fermionic flat array to a fermionic blocksparse array.

        Returns
        -------
        FermionicArray
            The equivalent fermionic blocksparse array.
        """
        new = AbelianArrayFlat.to_blocksparse(self)
        phases = {
            sector: int(phase)
            for sector, phase in zip(new.sectors, self._phases)
            if phase != 1
        }
        new.modify(phases=phases, oddpos=self._oddpos)
        return new

    def phase_sync(self, inplace=False) -> "FermionicArrayFlat":
        """Multiply all lazy phases into the block arrays.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArrayFlat
            The resolved array, which now has all trivial (+1) phases.
        """
        new = self if inplace else self.copy()

        # do broadcasted multiply to resolve phases
        new._blocks = (
            new._blocks * new._phases[(slice(None),) + (None,) * (new.ndim)]
        )
        new._phases = ar.do("ones", new.num_blocks, like=new._phases)

        return new

    def phase_flip(self, *axs, inplace=False) -> "FermionicArrayFlat":
        """Flip the phase of all sectors with odd parity at the given axis.

        Parameters
        ----------
        ax : int
            The axis along which to flip the phase.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArrayFlat
            The phase-flipped array.
        """
        new = self if inplace else self.copy()
        if not axs:
            # nothing to do
            return new
        flip_phases = (-1) ** ar.do("sum", new._sectors[:, axs], axis=1)
        new.modify(phases=new.phases * flip_phases)
        return new

    def phase_transpose(
        self,
        axes=None,
        inplace=False,
    ) -> "FermionicArrayFlat":
        """Phase this fermionic array as if it were transposed virtually, i.e.
        the actual arrays are not transposed. Useful when one wants the actual
        data layout to differ from the required fermionic mode layout.

        Parameters
        ----------
        axes : tuple of int, optional
            The permutation of axes, by default None.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArrayFlat
        """
        new = self if inplace else self.copy()

        N = new.ndim
        if axes is None:
            axes = tuple(reversed(range(N)))

        # convert permutation to sequence of pairwise neighboring swaps
        swaps = perm_to_swaps(axes)

        # count how many swaps of odd charges there are
        parities = [self.sectors[:, i] % 2 for i in range(N)]
        nswap = 0
        for il, ir in swaps:
            nswap = nswap + (parities[il] * parities[ir])
            parities[il], parities[ir] = parities[ir], parities[il]

        # absorb into current phases
        phase_change = (-1) ** nswap
        new.modify(phases=new.phases * phase_change)

        return new

    def phase_global(self, inplace=False):
        raise NotImplementedError

    def _binary_blockwise_op(self, other, fn, inplace=False, **kwargs):
        raise NotImplementedError

    def transpose(self, axes=None, phase=True, inplace=False):
        raise NotImplementedError

    def conj(self, phase_permutation=True, phase_dual=False, inplace=False):
        raise NotImplementedError

    def dagger(self, phase_dual=False, inplace=False):
        raise NotImplementedError

    @property
    def H(self):
        """Fermionic conjugate transpose."""
        return self.dagger()

    def fuse(self, *axes_groups, expand_empty=True, inplace=False):
        raise NotImplementedError

    def unfuse(self, axis, inplace=False):
        raise NotImplementedError

    def einsum(self, eq, preserve_array=False):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def to_dense(self):
        raise NotImplementedError

    def allclose(self, other, **kwargs):
        raise NotImplementedError

    def trace(self):
        raise NotImplementedError


@tensordot.register(FermionicArrayFlat)
def tensordot_fermionic_flat(a, b, axes=2, preserve_array=False, **kwargs):
    raise NotImplementedError


class Z2FermionicArrayFlat(FermionicArrayFlat):
    static_symmetry = get_symmetry("Z2")
