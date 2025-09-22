"""Fermionic symmetric arrays with flat backend."""

import autoray as ar

from .flat_abelian_array import AbelianArrayFlat
from .interface import tensordot
from .sparse_fermionic_array import FermionicArray, FermionicCommon
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


class FermionicArrayFlat(AbelianArrayFlat, FermionicCommon):
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
            self._phases = None
        elif hasattr(phases, "shape"):
            self._phases = phases
        else:
            self._phases = ar.do("array", phases, like=self._blocks)

        assert not oddpos
        self._oddpos = ()

    @property
    def phases(self):
        """The phases for each block."""
        if self._phases is None:
            self._phases = ar.do("ones", self.num_blocks, like=self._blocks)
        return self._phases

    def check(self):
        """Check the internal consistency of the array."""
        AbelianArrayFlat.check(self)
        if self._phases is not None:
            assert self._phases.shape == (self.num_blocks,)
            assert ar.do("all", ar.do("isin", self._phases, [-1, 1]))

    def copy(self, deep=False) -> "FermionicArrayFlat":
        """Create a copy of the array."""
        new = AbelianArrayFlat.copy(self, deep=deep)

        if self._phases is not None:
            if deep:
                new._phases = ar.do("copy", self._phases, like=self.backend)
            else:
                new._phases = self._phases
        else:
            new._phases = None

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
            if isinstance(phases, str):
                if phases == "reset":
                    self._phases = None
                else:
                    raise ValueError(f"Unknown phases value '{phases}'")
            else:
                self._phases = phases

        if isinstance(self._phases, int):
            raise ValueError

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

        if self._phases is None:
            phases = {}
        else:
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

        if new._phases is not None:
            # do broadcasted multiply to resolve phases
            phases_b = new._phases[(slice(None),) + (None,) * (new.ndim)]
            new.modify(
                blocks=new._blocks * phases_b,
                phases="reset",
            )

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
            # full reversal, shortcut to count the swaps
            nswap = ar.do("sum", new.sectors % 2, axis=1) // 2
        elif all(ax == i for i, ax in enumerate(axes)):
            # identity, nothing to do
            return new
        else:
            # convert permutation to sequence of pairwise neighboring swaps
            swaps = perm_to_swaps(axes)

            # count how many swaps of odd charges there are
            parities = [new.sectors[:, i] % 2 for i in range(N)]
            nswap = None
            for il, ir in swaps:
                if nswap is None:
                    nswap = parities[il] * parities[ir]
                else:
                    nswap = nswap + (parities[il] * parities[ir])
                parities[il], parities[ir] = parities[ir], parities[il]

        if nswap is None:
            raise ValueError("No phase changes required.")

        # absorb into current phases
        phase_change = (-1) ** nswap
        new.modify(
            phases=(
                phase_change
                if new._phases is None
                else new._phases * phase_change
            )
        )

        return new

    def phase_sector(self, sector, inplace=False):
        raise NotImplementedError(
            "Flat fermionic arrays do not support individual sector phasing."
        )

    def phase_global(self, inplace=False):
        """Flip the global phase of the array.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArray
        """
        new = self if inplace else self.copy()
        new.modify(phases=-new.phases)
        return new

    def transpose(
        self,
        axes=None,
        phase=True,
        inplace=False,
    ) -> "FermionicArrayFlat":
        """Transpose this flat fermionic array, by default accounting for the
        phases accumulated from swapping odd charges.

        Parameters
        ----------
        axes : tuple of int, optional
            The permutation of axes, by default None.
        phase : bool, optional
            Whether to flip the phase of sectors whose odd charges undergo a
            odd permutation. By default True.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArrayFlat
            The transposed array.
        """
        new = self if inplace else self.copy()

        if phase:
            # perform the phase accumulation separately first
            new.phase_transpose(axes, inplace=True)

        # transpose the actual arrays
        return AbelianArrayFlat.transpose(new, axes, inplace=True)

    def conj(self, phase_permutation=True, phase_dual=False, inplace=False):
        """Conjugate this flat fermionic array. By default this include phases
        from both the virtual flipping of all axes, but *not* the conjugation
        of dual indices, such that::

            (
                tensordot_fermionic(x.conj(), x, ndim) ==
                tensordot_fermionic(x, x.conj(), ndim)
            )

        If all indices have matching dualness (i.e. all bra or all ket), *or*
        you set `phase_dual=True` then the above contractions will also be
        equal to ``x.norm() ** 2``.

        Parameters
        ----------
        phase_permutation : bool, optional
            Whether to flip the phase of sectors whose odd charges undergo a
            odd permutation due to *virtually* flipping the order of axes, by
            default True.
        phase_dual : bool, optional
            Whether to flip the phase of dual indices, by default False. If a
            FermionicArrayFlat has a mix of dual and non-dual indices, and you
            are explicitly forming the norm, you may want to set this to True.
            But if it is part of a large tensor network you only need to flip
            the phase of true 'outer' dual indices.

        Returns
        -------
        FermionicArrayFlat
        """
        new = self if inplace else self.copy()

        if phase_permutation:
            # perform the phase accumulation separately first
            new.phase_transpose(inplace=True)

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new.indices) if ix.dual
            )
            new.phase_flip(*axs_conj, inplace=True)

        # conjugate the actual arrays
        return AbelianArrayFlat.conj(new, inplace=True)

    def dagger(self, phase_dual=False, inplace=False):
        raise NotImplementedError

    @property
    def H(self):
        """Fermionic conjugate transpose."""
        return self.dagger()

    def fuse(
        self, *axes_groups, expand_empty=True, inplace=False
    ) -> "FermionicArrayFlat":
        """Fermionic fusion of axes groups. This includes three sources of
        phase changes:

        1. Initial fermionic transpose to make each group contiguous.
        2. Flipping of non dual indices, if merged group is overall dual.
        3. Virtual transpose within a group, if merged group is overall dual.

        A grouped axis is overall dual if the first axis in the group is dual.

        Parameters
        ----------
        axes_groups : Sequence[Sequence[int]]
            The axes groups to fuse. See `AbelianArray.fuse` for more details.
        expand_empty : bool, optional
            Whether to expand empty groups into new axes.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FermionicArrayFlat
        """
        return FermionicCommon.fuse(
            self,
            *axes_groups,
            expand_empty=expand_empty,
            inplace=inplace,
        )

    def unfuse(self, axis, inplace=False):
        """Fermionic unfuse, which includes two sources of phase changes:

        1. Flipping of non dual sub indices, if overall index is dual.
        2. Virtual transpose within group, if overall index is dual.

        Parameters
        ----------
        axis : int
            The axis to unfuse.
        """
        return FermionicCommon.unfuse(self, axis, inplace=inplace)

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
