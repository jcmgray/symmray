"""Fermionic symmetric arrays with flat backend."""

import functools

import autoray as ar

from ..abelian_common import AbelianCommon
from ..common import SymmrayCommon
from ..fermionic_common import FermionicCommon
from ..fermionic_local_operators import FermionicOperator
from ..sparse.sparse_fermionic_array import FermionicArray
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .flat_array_common import FlatArrayCommon, truncate_svd_result_flat
from .flat_data_common import FlatCommon
from .flat_vector import FlatVector


@functools.lru_cache(maxsize=2**14)
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


def oddpos_parse(oddpos, parity):
    if oddpos is None:
        # no dummy indices
        return (), ()

    if isinstance(oddpos, (list, tuple)) and isinstance(
        oddpos[0], FermionicOperator
    ):
        # an explicit sequence of subsumed odd ranks
        oddpos = tuple(oddpos)

        if not isinstance(parity, (list, tuple)):
            parities = (parity,) * len(oddpos)
        else:
            parities = tuple(parity)

        return oddpos, parities

    if not isinstance(oddpos, FermionicOperator):
        oddpos = FermionicOperator(oddpos)

    return (oddpos,), (parity,)


class FermionicArrayFlat(
    FermionicCommon,
    FlatArrayCommon,
    FlatCommon,
    AbelianCommon,
    SymmrayCommon,
):
    """Fermionic abelian symmetric array with flat backend.

    Parameters
    ----------
    sectors : array_like
        The stack of sector keys, with shape (num_blocks, ndim). Each row
        represents a sector of a corresponding block, and each column
        represents a charge in a given axis.
    blocks : array_like
        The stack of array blocks, with shape (num_blocks, *shape_block), i.e.
        `ndim + 1` dimensions, where the first dimension is the block index,
        which should match the first dimension of `sectors`, and the rest are
        the dimensions of individual blocks.
    indices : sequence[FlatIndex]
        Indices describing the dualness and any subindex information for each
        dimension of the array. If bools are supplied, they will be converted
        to a FlatIndex with the corresponding dualness, and no subindex
        information.
    phases : array_like, optional
        An array of +/- 1 phases, with shape (num_blocks,), giving the phase
        of each block. If not supplied, all phases are assumed to be +1.
    label : hashable, optional
        An optional label for the array, potentially needed for ordering dummy
        odd fermionic modes.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = (
        "_blocks",
        "_indices",
        "_odd_parities",
        "_oddpos",
        "_phases",
        "_sectors",
        "_symmetry",
        "backend",
    )
    fermionic = True
    static_symmetry = None

    def __init__(
        self,
        sectors,
        blocks,
        indices,
        phases=None,
        label=None,
        symmetry=None,
        oddpos=None,
        odd_parities=None,
    ):
        self._init_abelian(
            sectors=sectors,
            blocks=blocks,
            indices=indices,
            symmetry=symmetry,
            label=label,
        )

        if phases is None:
            self._phases = None
        elif hasattr(phases, "shape"):
            self._phases = phases
        else:
            self._phases = ar.do("array", phases, like=self._blocks)

        if oddpos is None and self.label is not None:
            # default to the array label
            oddpos = self.label

        if oddpos:
            oddpos, odd_parities = oddpos_parse(oddpos, self.parity)
        else:
            # None or empty
            oddpos = ()
            odd_parities = ()

        self._oddpos = oddpos
        self._odd_parities = odd_parities

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

    @property
    def odd_parities(self):
        """The parities of each oddpos label carried by this array."""
        return self._odd_parities

    @property
    def parity(self):
        """The total parity of the array, 0 for even, 1 for odd."""
        return ar.do("sum", self.sectors[0] % 2, like=self.backend) % 2

    def check(self):
        """Check the internal consistency of the array."""
        self._check_abelian()
        if self._phases is not None:
            assert ar.do("shape", self._phases) == (self.num_blocks,)
            assert ar.do("all", ar.do("isin", self._phases, [-1, 1]))
        assert len(self._oddpos) == len(self._odd_parities)

    def new_with(self, sectors, blocks, indices) -> "FermionicArrayFlat":
        """Create a new flat fermionic array of the same class as this one.
        Unlike `copy`, this does not copy over any existing data and drops
        for example `label`, `phases`, and `oddpos`.
        """
        new = self._new_with_abelian(sectors, blocks, indices)
        new._phases = None
        new._oddpos = ()
        new._odd_parities = ()
        return new

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
        new._odd_parities = self._odd_parities
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
        new._odd_parities = self._odd_parities
        if DEBUG:
            new.check()
        return new

    def modify(
        self,
        sectors=None,
        blocks=None,
        indices=None,
        phases=None,
        oddpos=None,
        odd_parities=None,
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

        if oddpos is not None:
            self._oddpos = oddpos
        if odd_parities is not None:
            self._odd_parities = odd_parities
        if DEBUG:
            self.check()
        return self

    def set_params(self, params):
        """Set the underlying array blocks."""
        self._set_params_abelian(params)

        if self._phases is not None:
            try:
                self._phases = ar.do("array", self._phases, like=self.backend)
            except ImportError:
                # params is possibly a placeholder of some kind
                pass

        if DEBUG:
            self.check()

    def _map_blocks(self, fn_sector=None, fn_block=None):
        self._map_blocks_abelian(fn_sector=fn_sector, fn_block=fn_block)
        if fn_sector is not None:
            if self._phases is not None:
                # NOTE: leave missing phases, assumed to stay all ones
                self._phases = fn_sector(self.phases)

    @classmethod
    def from_blocks(
        cls,
        blocks,
        indices,
        phases=None,
        oddpos=None,
        symmetry=None,
        label=None,
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
        symmetry : str or Symmetry, optional
            The symmetry of the array, if not using a specific symmetry class.
        label : hashable, optional
            An optional label for the array, potentially needed for ordering
            dummy odd fermionic modes.

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
            label=label,
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
        symmetry : str or Symmetry, optional
            The symmetry to use. If not supplied, the symmetry of `x` is used.
        """
        return cls.from_blocks(
            blocks=x.blocks,
            indices=[ix.to_flat() for ix in x.indices],
            phases=x.phases,
            oddpos=x.oddpos,
            symmetry=symmetry or x.symmetry,
            label=x.label,
        )

    def to_blocksparse(self) -> "FermionicArray":
        """Convert this fermionic flat array to a fermionic blocksparse array.

        Returns
        -------
        FermionicArray
            The equivalent fermionic blocksparse array.
        """
        if self._phases is None:
            phases = {}
        else:
            phases = {
                tuple(map(int, sector)): int(phase)
                for sector, phase in zip(self._sectors, self._phases)
                if phase != 1
            }

        oddpos = tuple(
            f for f, p in zip(self._oddpos, self._odd_parities) if p == 1
        )

        return self._to_blocksparse_abelian(phases=phases, oddpos=oddpos)

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
        if self.ndim <= 1:
            # nothing to do
            return self if inplace else self.copy()

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
            nswap = (
                ar.do("sum", new.sectors % 2, axis=1, like=self.backend) // 2
            )
        elif all(ax == i for i, ax in enumerate(axes)):
            # identity, nothing to do
            return new
        else:
            # convert permutation to sequence of pairwise neighboring swaps
            swaps = perm_to_swaps(tuple(axes))

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

    def _resolve_oddpos_conj(self, phase_permutation=True):
        """Assuming we have effectively taken the conjugate of a fermionic
        array with dummy oddpos modes, get their new order and compute any
        phase changes coming from moving back to the beginning of the index
        order.
        """
        if not self.oddpos:
            return

        # 1. we get a reversal and conjugation of the oddpos modes
        #       dummy modes          real indices
        # | o0 o1 ... on-2 on-1 | P0 P1 ... Pn-2 Pn-1 |
        #                     <-->
        # | Pn-1 Pn-2 ... P1 P0 | on-1 on-2 ... o1 o0 |
        new_oddpos = tuple(r.dag for r in reversed(self.oddpos))
        new_odd_parities = tuple(reversed(self.odd_parities))
        self.modify(oddpos=new_oddpos, odd_parities=new_odd_parities)

        if phase_permutation:
            # 2. moving oddpos charges back to left
            # after flipping might generate global sign
            # | Pn-1 Pn-2 ... P1 P0 | on-1 on-2 ... o1 o0 |
            #                     <--
            # | on-1 on-2 ... o1 o0 | Pn-1 Pn-2 ... P1 P0 |
            sign = (-1) ** (self.parity * sum(self.odd_parities) % 2)
            self.modify(phases=self.phases * sign)

    def _resolve_oddpos_combine(self, a, b):
        """Calculate the new combined dummy odd modes and any associated global
        phases combing from contracting two fermionic arrays `a` and `b`. This
        modifies this array in place.
        """
        l_oddpos = a.oddpos
        r_oddpos = b.oddpos
        if not l_oddpos and not r_oddpos:
            # no odd modes to resolve
            self._oddpos = self._odd_parities = ()
            return

        l_odd_parities = a.odd_parities
        r_odd_parities = b.odd_parities

        oddpos = [*l_oddpos, *r_oddpos]
        odd_parities = [*l_odd_parities, *r_odd_parities]

        # 1. initially we have:
        # left dummy modes | left real modes | right dummy modes | right real modes
        # so we must calc phase from moving right dummy modes past left real modes
        phase = (-1) ** (a.parity * sum(r_odd_parities) % 2)

        # then we want to sort the joint set of left and right dummy modes,
        perm = tuple(sorted(range(len(oddpos)), key=lambda i: oddpos[i]))
        swaps = perm_to_swaps(perm)
        for i, j in swaps:
            a, b = oddpos[i], oddpos[j]
            pa, pb = odd_parities[i], odd_parities[j]
            # compute sign from swap
            phase = phase * (-1) ** (pa * pb)
            # perform swap
            oddpos[i], oddpos[j] = b, a
            odd_parities[i], odd_parities[j] = pb, pa

        # do the global phase, and set the new sorted oddpos labels and parities
        self.modify(
            oddpos=tuple(oddpos),
            odd_parities=tuple(odd_parities),
            phases=self.phases * phase,
        )

    def _resolve_oddpos_squeeze(self, axes_squeeze):
        """Assuming we are about to squeeze away `axes_squeeze`, compute the
        phases associated with moving them to the beginning of the array, and
        then turn them into dummy oddpos modes.
        """
        axes_leave = []
        squeezed_oddpos = []
        squeezed_odd_parities = []
        for ax, ix in enumerate(self.indices):
            if ax in axes_squeeze:
                # all sectors should have same charge for squeezed axes
                odd_parity = self._sectors[0, ax] % 2

                label = self.label
                if label is None:
                    raise ValueError(
                        "Cannot squeeze flat fermionic index with possible odd"
                        " parity if array has no ordering `.label` attribute."
                    )
                op = FermionicOperator(("squeeze", label, ax), ix.dual)
                squeezed_oddpos.append(op)
                squeezed_odd_parities.append(odd_parity)
            else:
                axes_leave.append(ax)

        self.phase_transpose((*axes_squeeze, *axes_leave), inplace=True)
        self.modify(
            oddpos=(*self.oddpos, *squeezed_oddpos),
            odd_parities=(*self.odd_parities, *squeezed_odd_parities),
        )

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

        # conjugate the actual arrays
        new._conj_abelian(inplace=True)

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new.indices) if not ix.dual
            )
            new.phase_flip(*axs_conj, inplace=True)

        if new.oddpos:
            # handle potential dummy odd modes
            new._resolve_oddpos_conj(phase_permutation)

        return new

    def dagger(self, phase_dual=False, inplace=False):
        """Fermionic conjugate transpose."""
        new = self._conj_abelian(inplace=inplace)
        new._transpose_abelian(inplace=True)

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new.indices) if not ix.dual
            )
            new.phase_flip(*axs_conj, inplace=True)

        # handle potential dummy odd modes
        #     dagger defined by phaseless reversal of all axes
        new._resolve_oddpos_conj(phase_permutation=True)

        return new

    # ------------------------------- linalg -------------------------------- #

    def eigh(
        self,
        phase_eigenvalues=True,
    ) -> tuple[FlatVector, "FermionicArrayFlat"]:
        """Hermitian eigen-decomposition of this flat fermionic array.

        Parameters
        ----------
        phase_eigenvalues : bool, optional
            If True, any local phase will be absorbed into the eigenvalues,
            such that `U @ diag(w) @ U.H == a` always holds. By default
            True. If `False` then one of `U` or `U.H` should be phased
            individually to account for local phasesin the above expression.

        Returns
        -------
        eigenvalues : FlatVector
            The eigenvalues.
        eigenvectors : FermionicArrayFlat
            The abelian array of right eigenvectors.
        """
        x = self.phase_sync()

        w, U = x._eigh_abelian()

        if phase_eigenvalues and not x.indices[1].dual:
            # inner index is like |x><x| so introduce a phase flip,
            # we don't explicitly have Wdag so put phase in eigenvalues
            # XXX: is this the most compatible thing to do?
            # it means U @ diag(w) @ U.H == x always
            parities = w._sectors % 2
            w.modify(blocks=w._blocks * ((-1) ** parities)[:, None])

        return w, U

    def eigh_truncated(
        self,
        cutoff=-1.0,
        cutoff_mode=4,
        max_bond=-1,
        absorb=0,
        renorm=0,
        positive=0,
        drop_oddpos=True,
        **kwargs,
    ) -> tuple["FermionicArrayFlat", FlatVector, "FermionicArrayFlat"]:
        """Truncated hermitian eigen-decomposition of this assumed hermitian
        flat fermionic array.

        Parameters
        ----------
        cutoff : float, optional
            Absolute eigenvalue cutoff threshold.
        cutoff_mode : int or str, optional
            How to perform the truncation:

            - 1 or 'abs': trim values below ``cutoff``
            - 2 or 'rel': trim values below ``s[0] * cutoff``
            - 3 or 'sum2': trim s.t. ``sum(s_trim**2) < cutoff``.
            - 4 or 'rsum2': trim s.t. ``sum(s_trim**2) < sum(s**2) * cutoff``.
            - 5 or 'sum1': trim s.t. ``sum(s_trim**1) < cutoff``.
            - 6 or 'rsum1': trim s.t. ``sum(s_trim**1) < sum(s**1) * cutoff``.

        max_bond : int
            An explicit maximum bond dimension, use -1 for none.
        absorb : {-1, 0, 1, None}
            How to absorb the eigenvalues.

            - -1 or 'left': absorb into the left factor (U).
            - 0 or 'both': absorb the square root into both factors.
            - 1 or 'right': absorb into the right factor (VH).
            - None: do not absorb, return eigenvalues as a BlockVector.

        renorm : {0, 1}
            Whether to renormalize the eigenvalues (depends on `cutoff_mode`).
        positive: bool, optional
            If True, assume all eigenvalues are positive for a faster sort.
            By default False.
        drop_oddpos : bool, optional
            Whether to drop any dummy oddpos modes after the decomposition.
            By default True.

        Returns
        -------
        u : FermionicArrayFlat
            The fermionic array of left eigenvectors.
        w : FlatVector or None
            The vector of eigenvalues, or None if absorbed.
        uh : FermionicArrayFlat
            The fermionic array of right eigenvectors.
        """
        if kwargs:
            import warnings

            warnings.warn(
                f"Got unexpected kwargs {kwargs} in eigh_truncated "
                f"for {self.__class__}. Ignoring them.",
                UserWarning,
            )

        # since we are handling UH, we can add phase there
        w, U = self.eigh(phase_eigenvalues=False)

        # make sure to sort by descending absolute value
        if not positive:
            idx = ar.do(
                "argsort", -ar.do("abs", w._blocks, like=self.backend), axis=1
            )
            w.modify(
                blocks=ar.do("take_along_axis", w._blocks, idx, axis=1),
            )
            U.modify(
                blocks=ar.do(
                    "take_along_axis", U._blocks, idx[:, None, :], axis=2
                )
            )
        else:
            # assume all positive, just need to flip
            w.modify(blocks=w._blocks[:, ::-1])
            U.modify(blocks=U._blocks[:, :, ::-1])

        if DEBUG:
            w.check()
            U.check()

        if drop_oddpos:
            U._oddpos = ()
            U._odd_parities = ()
            U._label = None

        VH = U._dagger_abelian()

        if VH.indices[0].dual:
            # inner index is like |x><x| so introduce a phase flip
            VH.phase_flip(0, inplace=True)

        return truncate_svd_result_flat(
            U, w, VH, cutoff, cutoff_mode, max_bond, absorb, renorm
        )


class Z2FermionicArrayFlat(FermionicArrayFlat):
    static_symmetry = get_symmetry("Z2")
