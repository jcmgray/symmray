"""Fermionic symmetric arrays with flat backend."""

import autoray as ar

from ..abelian_common import AbelianCommon
from ..common import SymmrayCommon
from ..fermionic_common import FermionicCommon
from ..sparse.sparse_fermionic_array import FermionicArray
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .flat_array import FlatArrayCommon
from .flat_base import FlatCommon


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


class FermionicArrayFlat(
    FermionicCommon,
    FlatArrayCommon,
    FlatCommon,
    AbelianCommon,
    SymmrayCommon,
):
    __slots__ = (
        "_blocks",
        "_sectors",
        "_indices",
        "_symmetry",
        "backend",
        "_phases",
        "_oddpos",
    )
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
        self._init_abelian(
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

        if DEBUG:
            self.check()

    @property
    def phases(self):
        """The phases for each block."""
        if self._phases is None:
            self._phases = ar.do("ones", self.num_blocks, like=self._blocks)
        return self._phases

    @property
    def oddpos(self):
        """Any labels of dummy fermionic modes for odd parity tensors."""
        return self._oddpos

    def check(self):
        """Check the internal consistency of the array."""
        self._check_abelian()
        if self._phases is not None:
            assert ar.do("shape", self._phases) == (self.num_blocks,)
            assert ar.do("all", ar.do("isin", self._phases, [-1, 1]))

    def copy(self, deep=False) -> "FermionicArrayFlat":
        """Create a copy of the array."""
        new = self._copy_abelian(deep=deep)

        if self._phases is not None:
            if deep:
                new._phases = ar.do("copy", self._phases, like=self.backend)
            else:
                new._phases = self._phases
        else:
            new._phases = None

        new._oddpos = self._oddpos

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
        new = self._copy_with_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
        )
        new._phases = self._phases if phases is None else phases
        new._oddpos = self._oddpos
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
        self._modify_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
        )
        if phases is not None:
            if isinstance(phases, str):
                if phases == "reset":
                    self._phases = None
                else:
                    raise ValueError(f"Unknown phases value '{phases}'")
            else:
                self._phases = phases

        if DEBUG:
            self.check()

        return self

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
        new = self._to_blocksparse_abelian()

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

    def sort_stack(
        self,
        axes=None,
        all_axes=None,
        inplace=False,
    ) -> "FermionicArrayFlat":
        """Lexicgraphic sort the stack of sectors and blocks according to the
        values of charges in the specified axes, optionally filling in the rest
        of the axes with the remaining axes in the order they appear.

        Parameters
        ----------
        axes : int | tuple[int, ...], optional
            The axes to sort by. If a single integer is given, it will be
            interpreted as the axis to sort by. If a tuple of integers is
            given, it will be interpreted as the axes to sort by in order.
            Default is None, if all_axes is also None or True, this will sort
            all axes in their current order.
        all_axes : bool, optional
            Whether to include all non-specified axes as tie-breakers, after
            the specified axes. If ``None``, the default, this will be True
            if `axes` is not supplied explicitly, and False otherwise.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.
            Default is False, which returns a new array.
        """
        kord = self.get_sorting_indices(axes=axes, all_axes=all_axes)

        new_sectors = self._sectors[kord]
        new_blocks = self._blocks[kord]
        if self._phases is None:
            new_phases = None
        else:
            new_phases = self._phases[kord]

        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            phases=new_phases,
            inplace=inplace,
        )

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
        return new._transpose_abelian(axes, inplace=True)

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
            assert not new.oddpos

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new.indices) if ix.dual
            )
            new.phase_flip(*axs_conj, inplace=True)

        # conjugate the actual arrays
        return new._conj_abelian(inplace=True)

    def dagger(self, phase_dual=False, inplace=False):
        """Fermionic conjugate transpose."""
        new = self._conj_abelian(inplace=inplace)
        new._transpose_abelian(inplace=True)
        return new

    def einsum(self, eq: str, preserve_array=False):
        raise NotImplementedError

    def tensordot(
        self, other, axes=2, preserve_array=False, **kwargs
    ) -> "FermionicArrayFlat":
        # XXX: move into FermionicCommon as a generic

        if not isinstance(other, self.__class__):
            if getattr(other, "ndim", 0) == 0:
                # assume scalar
                return self * other
            else:
                raise TypeError(
                    f"Expected {self.__class__.__name__}, got {type(other)}."
                )

        a, b, new_axes_a, new_axes_b = self._prepare_for_tensordot_fermionic(
            other, axes
        )

        # perform blocked contraction!
        c = a._tensordot_abelian(
            b,
            axes=(new_axes_a, new_axes_b),
            # preserve array for resolving oddposs
            preserve_array=True,
            **kwargs,
        )

        # XXX: need flat resolve_combined_oddpos
        # resolve_combined_oddpos(a, b, c)

        if (c.ndim == 0) and (not preserve_array):
            c.phase_sync(inplace=True)
            return c.get_scalar_element()

        return c

    def __matmul__(self, other: "FermionicArrayFlat", preserve_array=False):
        # XXX: move into FermionicCommon as a generic

        # shortcut of matrx/vector products
        if self.ndim > 2 or other.ndim > 2:
            raise ValueError("Matrix multiplication requires 2D arrays.")

        if other.indices[0].dual:
            # have |x><x| -> want <x|x>
            other = other.phase_flip(0)

        a = self.phase_sync()
        b = other.phase_sync()
        c = a._matmul_abelian(b, preserve_array=True)

        # XXX: need to implement
        # resolve_combined_oddpos(a, b, c)

        if c.ndim == 0:
            c.phase_sync(inplace=True)
            return c.get_scalar_element()

        return c

    def to_dense(self):
        raise NotImplementedError

    def trace(self):
        raise NotImplementedError


class Z2FermionicArrayFlat(FermionicArrayFlat):
    static_symmetry = get_symmetry("Z2")
