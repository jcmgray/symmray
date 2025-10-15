"""Fermionic symmetric arrays with block sparse backend."""

import autoray as ar

from ..abelian_common import AbelianCommon
from ..common import SymmrayCommon
from ..fermionic_common import FermionicCommon
from ..fermionic_local_operators import FermionicOperator
from ..symmetries import calc_phase_permutation, get_symmetry
from ..utils import DEBUG, get_rng
from .sparse_array_common import (
    SparseArrayCommon,
    permuted,
)
from .sparse_data_common import BlockCommon


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def oddpos_parse(oddpos, parity):
    """Parse the label ``oddpos``, given the parity of the array."""
    if parity and oddpos is None:
        raise ValueError(
            "`oddpos` required for calculating global phase of "
            "contractions with odd parity fermionic arrays."
        )

    if isinstance(oddpos, (list, tuple)):
        if len(oddpos) == 0:
            return ()
        elif isinstance(oddpos[0], FermionicOperator):
            # an explicit sequence of subsumed odd ranks
            return tuple(oddpos)

    if not parity:
        # we can just drop even parity labels
        return ()

    if not isinstance(oddpos, FermionicOperator):
        oddpos = FermionicOperator(oddpos)

    return (oddpos,)


class FermionicArray(
    FermionicCommon,
    SparseArrayCommon,
    BlockCommon,
    AbelianCommon,
    SymmrayCommon,
):
    """A fermionic block symmetry array.

    Parameters
    ----------
    indices : tuple of Index
        The indices of the array.
    charge : hashable, optionals
        The total charge of the array, if not given it will be inferred from
        either the first sector or set to the identity charge, if no sectors
        are given.
    blocks : dict, optional
        The blocks of the array, by default empty.
    phases : dict, optional
        The lazy phases of each block, by default empty.
    oddpos : object or FermionicOperator, optional
        If the array has odd parity, the 'position' of it, or the ordering of
        subsumed positions. This can be an arbitrary hashable, sortable type,
        in which case it will be wrapped in a FermionicOperator.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = (
        "_indices",
        "_blocks",
        "_charge",
        "_symmetry",
        "_phases",
        "_oddpos",
    )
    fermionic = True
    static_symmetry = None

    def __init__(
        self,
        indices,
        charge=None,
        blocks=(),
        phases=(),
        oddpos=None,
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
        self._phases = dict(phases)

        if oddpos is None and self.label is not None:
            # default to the array label
            oddpos = self.label

        self._oddpos = oddpos_parse(oddpos, self.parity)
        if DEBUG:
            self.check()

    @property
    def parity(self):
        """The parity of the total charge."""
        return self.symmetry.parity(self.charge)

    @property
    def phases(self):
        """The lazy phases of each sector. Trivial phases are not necessarily
        stored.
        """
        try:
            return self._phases
        except AttributeError:
            self._phases = {}
            return self._phases

    @property
    def oddpos(self):
        """The odd rank of the array, i.e. the ordering of subsumed odd
        positions.
        """
        try:
            return self._oddpos
        except AttributeError:
            self._oddpos = ()
            return self._oddpos

    def copy(self):
        """Create a copy of this fermionic array.

        Returns
        -------
        FermionicArray
            The copied array.
        """
        new = self._copy_abelian()
        new._phases = self.phases.copy()
        new._oddpos = self.oddpos
        return new

    def copy_with(self, indices=None, blocks=None, charge=None, phases=None):
        """Create a copy of this fermionic array with some attributes replaced.

        Parameters
        ----------
        indices : tuple of Index, optional
            The new indices, if None, the original indices are used.
        blocks : dict, optional
            The new blocks, if None, the original blocks are used.
        charge : int, optional
            The new total charge, if None, the original charge is used.
        phases : dict, optional
            The new phases, if None, the original phases are used.
        """
        new = self._copy_with_abelian(
            indices=indices, blocks=blocks, charge=charge
        )
        new._phases = self.phases.copy() if phases is None else phases
        new._oddpos = self.oddpos

        if DEBUG:
            new.check()

        return new

    def modify(
        self,
        indices=None,
        blocks=None,
        charge=None,
        phases=None,
        oddpos=None,
    ):
        """Modify this fermionic array inplace. This is for internal use, and
        does not perform any checks on the updated attributes.

        Parameters
        ----------
        indices : tuple of Index, optional
            The new indices, if None, the original indices are used.
        blocks : dict, optional
            The new blocks, if None, the original blocks are used.
        charge : int, optional
            The new total charge, if None, the original charge is used.
        phases : dict, optional
            The new phases, if None, the original phases are used.
        oddpos : object or FermionicOperator, optional
            The new oddpos, if None, the original oddpos is used.
        """
        if phases is not None:
            self._phases = phases
        if oddpos is not None:
            self._oddpos = oddpos
        return self._modify_abelian(
            indices=indices, blocks=blocks, charge=charge
        )

    def randomize_phases(self, seed=None, inplace=False) -> "FermionicArray":
        """Randomize the phases of each sector to either +1 or -1. This is
        useful for testing.

        Parameters
        ----------
        seed : int or numpy.random.Generator, optional
            The random seed or generator, by default None.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArray
            The phase randomized array.
        """
        rng = get_rng(seed)
        new_phases = {}
        for sector in self.sectors:
            if rng.uniform() > 0.5:
                new_phases[sector] = -1
        return self._modify_or_copy(inplace=inplace, phases=new_phases)

    def phase_sync(self, inplace=False) -> "FermionicArray":
        """Multiply all lazy phases into the block arrays.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArray
            The resolved array, which now has no lazy phases.
        """
        new = self if inplace else self.copy()
        phases = new.phases
        while phases:
            sector, phase = phases.popitem()
            if phase == -1:
                try:
                    new.set_block(sector, -new.get_block(sector))
                except KeyError:
                    # if the block is not present, it is zero
                    # this can happen e.g. if two arrays have been aligned
                    # for contraction
                    # TODO: use a drop_sectors method instead?
                    pass

        return new

    def phase_flip(self, *axs, inplace=False) -> "FermionicArray":
        """Flip the phase of all sectors with odd parity at the given axis.

        Parameters
        ----------
        ax : int
            The axis along which to flip the phase.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArray
            The phase-flipped array.
        """
        new = self if inplace else self.copy()

        if not axs:
            # nothing to do
            return new

        new_phases = new.phases.copy()

        for sector in new.sectors:
            parity = sum(new.symmetry.parity(sector[ax]) for ax in axs) % 2

            if parity:
                new_phase = -new_phases.get(sector, 1)
                if new_phase == 1:
                    new_phases.pop(sector, None)
                else:
                    # only keep non-trivial phases
                    new_phases[sector] = new_phase

        return new.modify(phases=new_phases)

    def phase_transpose(self, axes=None, inplace=False):
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
        FermionicArray
        """
        new = self if inplace else self.copy()

        for sector in new.sectors:
            parities = tuple(new.symmetry.parity(q) for q in sector)

            phase_new = (
                # start with old phase
                new._phases.get(sector, 1)
                *
                # get the phase from permutation
                calc_phase_permutation(parities, axes)
            )

            if phase_new == 1:
                new._phases.pop(sector, None)
            else:
                new._phases[sector] = phase_new

        return new

    def phase_sector(self, sector, inplace=False):
        """Flip the phase of a specific sector.

        Parameters
        ----------
        sector : tuple of hashable
            The sector to flip the phase of.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArray
        """
        new = self if inplace else self.copy()
        new_phase = -new._phases.pop(sector, 1)
        if new_phase == -1:
            new._phases[sector] = -1
        return new

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
        for sector in new.sectors:
            phase = -new.phases.pop(sector, 1)
            if phase == -1:
                new._phases[sector] = phase
        return new

    def _map_blocks(self, fn_block=None, fn_sector=None):
        """Map the blocks and their keys (sectors) of the array inplace."""
        self._map_blocks_blockcommon(fn_block, fn_sector)
        if fn_sector is not None:
            # need to update phase keys as well
            self._phases = {fn_sector(s): p for s, p in self._phases.items()}

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
        self.modify(oddpos=new_oddpos)

        if phase_permutation and self.parity and len(new_oddpos) % 2:
            # 2. moving oddpos charges back to left
            # after flipping might generate global sign
            # | Pn-1 Pn-2 ... P1 P0 | on-1 on-2 ... o1 o0 |
            #                     <--
            # | on-1 on-2 ... o1 o0 | Pn-1 Pn-2 ... P1 P0 |
            self.phase_global(inplace=True)

    def _resolve_oddpos_combine(self, left, right):
        """Calculate the new combined dummy odd modes and any associated global
        phases combing from contracting two fermionic arrays `a` and `b`. This
        modifies this array in place.
        """
        l_oddpos = left.oddpos
        r_oddpos = right.oddpos

        if not l_oddpos and not r_oddpos:
            self._oddpos = ()
            return

        oddpos = [*l_oddpos, *r_oddpos]

        # e.g. (1, 2, 4, 5) + (3, 6, 7) -> [1, 2, 4, 5, 3, 6, 7]
        if left.parity and len(r_oddpos) % 2 == 1:
            # moving right oddpos charges over left sectors will generate sign
            phase = -1
        else:
            phase = 1

        # do a phased sort and annihilation of conjugate pairs
        i = 0
        while i < len(oddpos) - 1:
            a = oddpos[i]
            b = oddpos[i + 1]
            if a.label == b.label:
                # 'trace' out the pair
                if a.dual != b.dual:
                    if b.dual:
                        # |x><x|, ket-bra contraction
                        phase = -phase
                    oddpos.pop(i)
                    oddpos.pop(i)
                    # check previous
                    i = max(0, i - 1)
                else:
                    # detect non conjugate duplicates here as well
                    raise ValueError(
                        "`oddpos` must be unique conjugate pairs."
                    )
            elif b < a:
                # sort with phased swap
                oddpos[i] = b
                oddpos[i + 1] = a
                # check previous
                i = max(0, i - 1)
                phase = -phase
            else:
                # already sorted and not conjugate pair, move to next
                i += 1

        if phase == -1:
            self.phase_global(inplace=True)

        self._oddpos = tuple(oddpos)

    def transpose(self, axes=None, phase=True, inplace=False):
        """Transpose the fermionic array, by default accounting for the phases
        accumulated from swapping odd charges.

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
        FermionicArray
            The transposed array.
        """
        new = self if inplace else self.copy()

        old_phases = new.phases

        if axes is None:
            axes = tuple(range(new.ndim - 1, -1, -1))
        else:
            axes = tuple(axes)

        if phase:
            # compute new sector phases
            new_phases = {}
            for sector in new.sectors:
                parities = tuple(new.symmetry.parity(q) for q in sector)
                perm_phase = calc_phase_permutation(parities, axes)
                new_phase = old_phases.get(sector, 1) * perm_phase
                if new_phase == -1:
                    # only populate non-trivial phases
                    new_phases[permuted(sector, axes)] = -1
        else:
            # just permute the phase keys
            new_phases = {
                permuted(sector, axes): phase
                for sector, phase in old_phases.items()
            }

        # update phase dict
        new.modify(phases=new_phases)

        # then transpose block arrays
        new._transpose_abelian(axes=axes, inplace=True)

        return new

    def conj(self, phase_permutation=True, phase_dual=False, inplace=False):
        """Conjugate this fermionic array. By default this include phases from
        both the virtual flipping of all axes, but *not* the conjugation of
        dual indices, such that::

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
            FermionicArray has a mix of dual and non-dual indices, and you are
            explicitly forming the norm, you may want to set this to True. But
            if it is part of a large tensor network you only need to flip the
            phase of true 'outer' dual indices.

        Returns
        -------
        FermionicArray
        """
        new = self if inplace else self.copy()

        _conj = ar.get_lib_fn(new.backend, "conj")

        new_indices = tuple(ix.conj() for ix in new.indices)

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new_indices) if not ix.dual
            )
        else:
            axs_conj = None

        for sector, array in new.get_sector_block_pairs():
            # conjugate the actual array
            new.set_block(sector, _conj(array))

            if phase_permutation or phase_dual:
                parities = tuple(map(new.symmetry.parity, sector))

                # start with old phase
                phase_new = new._phases.get(sector, 1)

                if phase_permutation:
                    # get the phase from 'virtually' reversing all axes:
                    #     (perm=[ndim-1, ..., 0])
                    phase_new *= calc_phase_permutation(parities, None)

                if phase_dual:
                    # get the phase from conjugating 'bra' indices
                    phase_new *= (
                        -1 if (sum(parities[ax] for ax in axs_conj) % 2) else 1
                    )

                if phase_new == 1:
                    new._phases.pop(sector, None)
                else:
                    new._phases[sector] = phase_new

        new.modify(
            indices=new_indices,
            charge=new.symmetry.sign(new._charge),
        )

        new._resolve_oddpos_conj(phase_permutation=phase_permutation)

        return new

    def dagger(self, phase_dual=False, inplace=False):
        """Fermionic adjoint, implements `.H` attribute.

        Parameters
        ----------
        phase_dual : bool, optional
            Whether to flip the phase of dual indices, by default False. If a
            FermionicArray has a mix of dual and non-dual indices, and you are
            explicitly forming the norm, you may want to set this to True. But
            if it is part of a large tensor network you only need to flip the
            phase of 'outer' dual indices.
        inplace : bool, optional
            Whether to perform the operation in place.

        Returns
        -------
        FermionicArray
            The conjugate transposed array.
        """
        new = self if inplace else self.copy()

        _conj = ar.get_lib_fn(new.backend, "conj")
        _transpose = ar.get_lib_fn(new.backend, "transpose")

        new_indices = tuple(ix.conj() for ix in reversed(new.indices))

        # conjugate transpose all arrays
        new_blocks = {}
        new_phases = {}
        for sector, array in new.get_sector_block_pairs():
            new_sector = sector[::-1]

            # keep -1 phases and update dict with reversed sector
            if new._phases.pop(sector, 1) == -1:
                new_phases[new_sector] = -1

            new_blocks[new_sector] = _transpose(_conj(array))

        new.modify(
            indices=new_indices,
            charge=new.symmetry.sign(new._charge),
            blocks=new_blocks,
            phases=new_phases,
        )

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new_indices) if not ix.dual
            )
            new.phase_flip(*axs_conj, inplace=True)

        # handle potential dummy odd modes
        #     dagger defined by phaseless reversal of all axes
        new._resolve_oddpos_conj(phase_permutation=True)

        return new


# --------------- specific fermionic symmetric array classes ---------------- #


class Z2FermionicArray(FermionicArray):
    """A fermionic block array with Z2 symmetry."""

    static_symmetry = get_symmetry("Z2")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.fermion import SparseFermionTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import Z2

        blocks = [
            SubTensor(
                -array if self.phases.get(sector, 1) == -1 else array,
                q_labels=tuple(map(Z2, sector)),
            )
            for sector, array in self.get_sector_block_pairs()
        ]

        data = SparseFermionTensor(
            blocks,
            pattern=["-" if dual else "+" for dual in self.duals],
        )

        if flat:
            data = data.to_flat()

        data.shape = self.shape

        return data


class U1FermionicArray(FermionicArray):
    """A fermionic block array with U1 symmetry."""

    static_symmetry = get_symmetry("U1")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.fermion import SparseFermionTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import U1

        blocks = [
            SubTensor(
                -array if self.phases.get(sector, 1) == -1 else array,
                q_labels=tuple(map(U1, sector)),
            )
            for sector, array in self.get_sector_block_pairs()
        ]

        data = SparseFermionTensor(
            blocks,
            pattern=["-" if dual else "+" for dual in self.duals],
        )

        if flat:
            data = data.to_flat()

        data.shape = self.shape

        return data


class Z2Z2FermionicArray(FermionicArray):
    """A fermionic block array with Z2 x Z2 symmetry."""

    static_symmetry = get_symmetry("Z2Z2")


class U1U1FermionicArray(FermionicArray):
    """A fermionic block array with U1 x U1 symmetry."""

    static_symmetry = get_symmetry("U1U1")
