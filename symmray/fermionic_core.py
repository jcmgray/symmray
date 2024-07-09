import autoray as ar

from .abelian_core import (
    AbelianArray,
    permuted,
    tensordot_abelian,
    without,
)
from .interface import tensordot
from .symmetries import calc_phase_permutation, get_symmetry


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)


def resolve_combined_oddpos(left, right, new):
    """Given we have contracted two fermionic arrays, resolve the oddposs
    of the new array, possibly flipping the global phase.

    Parameters
    ----------
    left, right, new : FermionicArray
        The left and right arrays that were contracted, and the new array.
    """
    l_oddpos = left.oddpos
    r_oddpos = right.oddpos
    oddpos = [*l_oddpos, *r_oddpos]

    # e.g. (1, 2, 4, 5) + (3, 6, 7) -> [1, 2, 4, 5, 3, 6, 7]
    if left.parity and len(r_oddpos) % 2 == 1:
        # moving right oddpos charges over left will generate sign
        phase = -1
    else:
        phase = 1

    if l_oddpos and r_oddpos and (l_oddpos[-1] > r_oddpos[0]):
        # overlapping -> compute the phase of sorting the oddposs
        perm = tuple(argsort(oddpos))
        phase *= calc_phase_permutation((1,) * len(oddpos), perm)
        # -> [1, 2, 3, 4, 5, 6, 7]
        oddpos = [oddpos[i] for i in perm]

    if phase == -1:
        new.phase_global(inplace=True)

    # trim adjacent pairs of oddposs, which act like even parity
    i = 0
    while i < len(oddpos) - 1:
        # e.g. -> [3, 4, 5, 6, 7] -> [5, 6, 7] -> [7]
        rl = oddpos[i]
        rr = oddpos[i + 1]
        if (rl + 1 == rr) or (rl, rr) == (-1, 1):
            oddpos.pop(i)
            oddpos.pop(i)
        elif rl == rr:
            # also detect duplicates here
            raise ValueError("`oddpos` values must be unique.")
        else:
            i += 1

    new._oddpos = tuple(oddpos)


_fermionic_array_slots = AbelianArray.__slots__ + ("_phases", "_oddpos")


class FermionicArray(AbelianArray):
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
    oddpos : int or tuple of int, optional
        If the array has odd parity, the 'position' of it, or the ordering of
        subsumed positions. If a single integer is given, it is assumed to be
        the position of the odd rank. The integer must be non-zero so that it
        is still sortable after negation. If a tuple is given, it is assumed to
        be the sorted ordering of subsumed odd ranks. By default empty.
    """

    __slots__ = _fermionic_array_slots

    def __init__(
        self,
        indices,
        charge=None,
        blocks=(),
        phases=(),
        oddpos=(),
    ):
        super().__init__(indices=indices, charge=charge, blocks=blocks)
        self._phases = dict(phases)
        if self.parity and oddpos == ():
            raise ValueError(
                "`oddpos` required for odd parity fermionic arrays."
            )
        if isinstance(oddpos, int):
            if oddpos == 0:
                raise ValueError(
                    "`oddpos` must be non-zero for sorting purposes."
                )
            self._oddpos = (oddpos,)
        else:
            self._oddpos = tuple(oddpos)

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
        new = super().copy()
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
        new = super().copy_with(indices=indices, blocks=blocks, charge=charge)
        new._phases = self.phases.copy() if phases is None else phases
        new._oddpos = self.oddpos

        if new.parity:
            assert new._oddpos

        return new

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

        new._phases = new_phases

        # transpose block arrays
        AbelianArray.transpose(new, axes, inplace=True)

        return new

    def phase_flip(self, *axs, inplace=False):
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

        new._phases = new_phases

        return new

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

    def phase_sync(self, inplace=False):
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
        while new._phases:
            sector, phase = new._phases.popitem()
            if phase == -1:
                try:
                    new._blocks[sector] = -new._blocks[sector]
                except KeyError:
                    # if the block is not present, it is zero
                    # this can happen e.g. if two arrays have been aligned
                    # for contraction
                    # TODO: use a drop_sectors method instead?
                    pass

        return new

    def conj(self, phase_permutation=True, phase_dual=False, inplace=False):
        """Conjugate this fermionic array. By default this include phases from
        both the virtual flipping of all axes, and the conjugation of dual
        indices, such that::

            (
                tensordot_fermionic(x.conj(), x, ndim) ==
                tensordot_fermionic(x, x.conj(), ndim)
            )

        If all indices have matching dualness (i.e. all bra or all ket), then
        the above contractions will also be equal to ``x.norm() ** 2``.

        Parameters
        ----------
        phase_dual : bool, optional
            Whether to flip the phase of dual indices, by default False. If a
            FermionicArray has a mix of dual and non-dual indices, and you are
            explicitly forming the norm, you may want to set this to True. But
            if it is part of a large tensor network you only need to flip the
            phase of 'outer' dual indices.
        phase_permutation : bool, optional
            Whether to flip the phase of sectors whose odd charges undergo a
            odd permutation due to *virtually* flipping the order of axes, by
            default True.
        """
        new = self if inplace else self.copy()

        _conj = ar.get_lib_fn(new.backend, "conj")

        new._indices = tuple(ix.conj() for ix in new.indices)
        axs_conj = tuple(ax for ax, ix in enumerate(new._indices) if ix.dual)

        for sector, array in new.blocks.items():
            # conjugate the actual array
            new.blocks[sector] = _conj(array)

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

        new._charge = new.symmetry.sign(new._charge)
        new._oddpos = tuple(-r for r in reversed(new._oddpos))

        if phase_permutation and new.parity and (len(new._oddpos) % 2 == 1):
            # moving oddpos charges back to left
            # # after flipping will generate sign
            new.phase_global(inplace=True)

        return new

    def dagger(self, phase_dual=False, inplace=False):
        """Fermionic adjoint.

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
        for sector, array in new.blocks.items():
            new_sector = sector[::-1]

            if new._phases.pop(sector, 1) == -1:
                new_phases[new_sector] = -1

            new_blocks[new_sector] = _transpose(_conj(array))

        new._indices = new_indices
        new._blocks = new_blocks
        new._phases = new_phases
        new._charge = new.symmetry.sign(new._charge)
        new._oddpos = tuple(-r for r in reversed(new._oddpos))

        if new.parity and (len(new._oddpos) % 2 == 1):
            # moving oddpos charges back to left
            # # after flipping will generate sign
            new.phase_global(inplace=True)

        if phase_dual:
            axs_conj = tuple(
                ax for ax, ix in enumerate(new_indices) if ix.dual
            )
            new.phase_flip(*axs_conj, inplace=True)

        return new

    @property
    def H(self):
        """Fermionic conjugate transpose."""
        return self.dagger()

    def fuse(self, *axes_groups):
        """Fermionic fusion of axes groups. This includes three sources of
        phase changes:

        1. Initial fermionic transpose to make each group contiguous.
        2. Flipping of non dual indices, if merged group is overall dual.
        3. Virtual transpose within a group, if merged group is overall dual.

        A grouped axis is overall dual if the first axis in the group is dual.

        Parameters
        ----------
        axes_groups : tuple of tuple of int
            The axes groups to fuse.

        Returns
        -------
        FermionicArray
        """
        from symmray.abelian_core import calc_fuse_group_info

        new = self.copy()

        # handle empty groups and ensure hashable
        axes_groups = tuple(tuple(group) for group in axes_groups if group)
        if not axes_groups:
            # ... and no groups -> nothing to do
            return new

        # first make groups into contiguous blocks using fermionic transpose
        perm = calc_fuse_group_info(axes_groups, new.duals)[2]
        # this is the first step which introduces phases
        new.transpose(perm, inplace=True)
        # update groups to reflect new axes
        axes_groups = tuple(tuple(map(perm.index, g)) for g in axes_groups)

        # process each group with another two sources of phase changes:
        axes_flip = []
        virtual_perm = None
        for group in axes_groups:
            if new.indices[group[0]].dual:
                # overall dual index:
                # 1. flip non dual sub indices
                for ax in group:
                    if not new.indices[ax].dual:
                        axes_flip.append(ax)

                # 2. virtual transpose within group
                if virtual_perm is None:
                    virtual_perm = list(range(new.ndim))
                for axi, axj in zip(group, reversed(group)):
                    virtual_perm[axi] = axj

        if axes_flip:
            new.phase_flip(*axes_flip, inplace=True)

        # if the fused axes is overall bra, need phases from effective flip
        #   <a|<b|<c|  |a>|b>|c>    ->    P * <c|<b|<a|  |a>|b>|c>
        #   but actual array layout should not be flipped, so do virtually
        if virtual_perm is not None:
            new.phase_transpose(tuple(virtual_perm), inplace=True)

        # insert phases
        new.phase_sync(inplace=True)

        # so we can do the actual block concatenations
        return AbelianArray.fuse(new, *axes_groups)

    def unfuse(self, axis, inplace=False):
        """Fermionic unfuse, which includes two sources of phase changes:

        1. Flipping of non dual sub indices, if overall index is dual.
        2. Virtual transpose within group, if overall index is dual.

        Parameters
        ----------
        axis : int
            The axis to unfuse.
        """
        index = self.indices[axis]

        if index.dual:
            sub_indices = self.indices[axis].subinfo.indices
            # if overall index is dual, need to (see fermionic fuse):
            #     1. flip not dual sub indices back
            #     2. perform virtual transpose within group

            nnew = len(sub_indices)
            axes_flip = []
            virtual_perm = list(range(self.ndim + nnew - 1))

            for i, ix in enumerate(sub_indices):
                if not ix.dual:
                    axes_flip.append(axis + i)
                # reverse the order of the groups subindices
                virtual_perm[axis + i] = axis + nnew - i - 1

        # need to insert actual phases prior to block operations
        new = self.phase_sync(inplace=inplace)
        # do the non-fermionic actual block unfusing
        new = AbelianArray.unfuse(new, axis, inplace=True)

        if index.dual:
            # apply the phase changes
            if axes_flip:
                new.phase_flip(*axes_flip, inplace=True)
            new.phase_transpose(tuple(virtual_perm), inplace=True)

        return new

    def __matmul__(self, other):
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D arrays.")

        if other.indices[0].dual:
            # have |x><x| -> want <x|x>
            other = other.phase_flip(0)

        new = super().__matmul__(other)

        resolve_combined_oddpos(self, other, new)

        return new

    def to_dense(self):
        """Return dense representation of the fermionic array, with lazy phases
        multiplied in.
        """
        return AbelianArray.to_dense(self.phase_sync())

    def allclose(self, other, **kwargs):
        """Check if two fermionic arrays are element-wise equal within a
        tolerance, accounting for phases.

        Parameters
        ----------
        other : FermionicArray
            The other fermionic array to compare.
        """
        return AbelianArray.allclose(
            self.phase_sync(), other.phase_sync(), **kwargs
        )

    def trace(self):
        """Fermionic matrix trace."""
        ixl, ixr = self.indices

        if ixl.dual and not ixr.dual:
            return AbelianArray.trace(self.phase_sync())
        elif not ixl.dual and ixr.dual:
            return AbelianArray.trace(self.phase_flip(0).phase_sync())
        else:
            raise ValueError("Cannot trace a non-bra or non-ket.")


@tensordot.register(FermionicArray)
def tensordot_fermionic(a, b, axes=2, preserve_array=False, **kwargs):
    """Contract two fermionic arrays along the specified axes, accounting for
    phases from both transpositions and contractions.

    Parameters
    ----------
    a : FermionicArray
        The first fermionic array.
    b : FermionicArray
        The second fermionic array.
    axes : int or (tuple[int], tuple[int]), optional
        The axes to contract over, by default 2.
    """
    ndim_a = a.ndim
    ndim_b = b.ndim

    # parse the axes argument for single integer and also negative indices
    if isinstance(axes, int):
        axes_a = tuple(range(ndim_a - axes, ndim_a))
        axes_b = tuple(range(0, axes))
    else:
        axes_a, axes_b = axes
        axes_a = tuple(x % ndim_a for x in axes_a)
        axes_b = tuple(x % ndim_b for x in axes_b)
        if not len(axes_a) == len(axes_b):
            raise ValueError("Axes must have same length.")

    left_axes = without(range(ndim_a), axes_a)
    right_axes = without(range(ndim_b), axes_b)
    ncon = len(axes_a)

    # permute a & b so we have axes like
    #     in terms of data layout => [..., x, y, z], [x, y, z, ...]
    a = a.transpose((*left_axes, *axes_a))
    b = b.transpose((*axes_b, *right_axes))
    #     but in terms of 'phase layout' =>  [..., x, y, z], [z, y, x, ...]
    b.phase_transpose(
        (*range(ncon - 1, -1, -1), *range(ncon, b.ndim)), inplace=True
    )

    # new axes for tensordot_abelian having permuted inputs
    new_axes_a = tuple(range(ndim_a - ncon, ndim_a))
    new_axes_b = tuple(range(ncon))

    # if contracted index is like |x><x| phase flip to get <x|x>
    if a.size <= b.size:
        axs_flip = tuple(ax for ax in new_axes_a if not a.indices[ax].dual)
        a.phase_flip(*axs_flip, inplace=True)
    else:
        axs_flip = tuple(ax for ax in new_axes_b if b.indices[ax].dual)
        b.phase_flip(*axs_flip, inplace=True)

    # actually multiply block arrays with phases
    a.phase_sync(inplace=True)
    b.phase_sync(inplace=True)

    # perform blocked contraction!
    c = tensordot_abelian(
        a,
        b,
        axes=(new_axes_a, new_axes_b),
        # preserve array for resolving oddposs
        preserve_array=True,
        **kwargs,
    )

    # potential global phase flip from oddpos sorting
    resolve_combined_oddpos(a, b, c)

    if (c.ndim == 0) and (not preserve_array):
        try:
            return c.blocks[()]
        except KeyError:
            # no aligned blocks, return zero
            return 0.0

    return c


# --------------- specific fermionic symmetric array classes ---------------- #


class Z2FermionicArray(FermionicArray):
    __slots__ = _fermionic_array_slots
    symmetry = get_symmetry("Z2")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.fermion_symmetry import Z2
        from pyblock3.algebra.fermion import SparseFermionTensor, SubTensor

        blocks = [
            SubTensor(
                -array if self.phases.get(sector, 1) == -1 else array,
                q_labels=tuple(map(Z2, sector)),
            )
            for sector, array in self.blocks.items()
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
    __slots__ = _fermionic_array_slots
    symmetry = get_symmetry("U1")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.fermion_symmetry import U1
        from pyblock3.algebra.fermion import SparseFermionTensor, SubTensor

        blocks = [
            SubTensor(
                -array if self.phases.get(sector, 1) == -1 else array,
                q_labels=tuple(map(U1, sector)),
            )
            for sector, array in self.blocks.items()
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
    __slots__ = _fermionic_array_slots
    symmetry = get_symmetry("Z2Z2")


class U1U1FermionicArray(FermionicArray):
    __slots__ = _fermionic_array_slots
    symmetry = get_symmetry("U1U1")
