import autoray as ar

from .symmetric_core import (
    SymmetricArray,
    permuted,
    tensordot_symmetric,
    without,
)
from .interface import tensordot
from .symmetries import calc_phase_permutation, get_symmetry

_fermionic_array_slots = SymmetricArray.__slots__ + ("_phases",)


class FermionicArray(SymmetricArray):
    """A fermionic block symmetry array.

    Parameters
    ----------
    indices : tuple of Index
        The indices of the array.
    charge_total : int
        The total charge of the array, should have even parity.
    blocks : dict, optional
        The blocks of the array, by default empty.
    phases : dict, optional
        The lazy phases of each block, by default empty.
    """

    __slots__ = _fermionic_array_slots

    def __init__(
        self,
        indices,
        charge_total,
        blocks=(),
        phases=(),
    ):
        super().__init__(
            indices=indices, charge_total=charge_total, blocks=blocks
        )
        self._phases = dict(phases)
        if self.symmetry.parity(self.charge_total):
            raise ValueError("Total charge must be even parity.")

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

    def copy(self):
        """Create a copy of this fermionic array.

        Returns
        -------
        FermionicArray
            The copied array.
        """
        new = super().copy()
        new._phases = self.phases.copy()
        return new

    def copy_with(
        self, indices=None, blocks=None, charge_total=None, phases=None
    ):
        """Create a copy of this fermionic array with some attributes replaced.

        Parameters
        ----------
        indices : tuple of Index, optional
            The new indices, if None, the original indices are used.
        blocks : dict, optional
            The new blocks, if None, the original blocks are used.
        charge_total : int, optional
            The new total charge, if None, the original charge is used.
        phases : dict, optional
            The new phases, if None, the original phases are used.
        """
        new = super().copy_with(
            indices=indices, blocks=blocks, charge_total=charge_total
        )
        new._phases = self.phases.copy() if phases is None else phases

        if new.symmetry.parity(new.charge_total):
            raise ValueError("Total charge must be even parity.")

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
        SymmetricArray.transpose(new, axes, inplace=True)

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
                new._blocks[sector] = -new._blocks[sector]
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

    def conj(self, phase=True, inplace=False):
        """Conjugate this fermionic array. By default this include phases from
        both the virtual flipping of all axes, and the conjugation of dual
        indices, such that::

            (
                tensordot_fermionic(x.conj(), x, ndim) ==
                tensordot_fermionic(x, x.conj(), ndim)
            )

        Parameters
        ----------
        phase : bool, optional
            Whether to compute fermionic phases.
        """
        new = self if inplace else self.copy()

        _conj = ar.get_lib_fn(new.backend, "conj")

        new._indices = tuple(ix.conj() for ix in new.indices)
        axs_conj = tuple(ax for ax, ix in enumerate(new._indices) if ix.dual)

        for sector, array in new.blocks.items():
            # conjugate the actual array
            new.blocks[sector] = _conj(array)

            if phase:
                parities = tuple(map(new.symmetry.parity, sector))

                phase_new = (
                    # start with old phase
                    new._phases.get(sector, 1)
                    # get the phase from 'virtually' reversing all axes:
                    #     (perm=[ndim-1, ..., 0])
                    * calc_phase_permutation(parities, None)
                    # get the phase from conjugating 'bra' indices
                    * (-1 if (sum(parities[ax] for ax in axs_conj) % 2) else 1)
                )

                if phase_new == 1:
                    new._phases.pop(sector, None)
                else:
                    new._phases[sector] = phase_new

        return new

    def dagger(self, phase=True, inplace=False):
        """Fermionic adjoint.

        Parameters
        ----------
        phase : bool, optional
            Whether to flip the phase conjugate indices, by default True.
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

        if phase:
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
        from symmray.symmetric_core import calc_fuse_info

        new = self.copy()

        # handle empty groups
        axes_groups = tuple(filter(None, axes_groups))
        if not axes_groups:
            # ... and no groups -> nothing to do
            return new

        # first make groups into contiguous blocks using fermionic transpose
        perm = calc_fuse_info(axes_groups, new.duals)[2]
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
            # print("fuse flips:", axes_flip)
            new.phase_flip(*axes_flip, inplace=True)

        # if the fused axes is overall bra, need phases from effective flip
        #   <a|<b|<c|  |a>|b>|c>    ->    P * <c|<b|<a|  |a>|b>|c>
        #   but actual array layout should not be flipped, so do virtually
        if virtual_perm is not None:
            new = new.phase_transpose(
                tuple(virtual_perm), inplace=True
            )

        # insert phases
        new.phase_sync(inplace=True)

        # so we can do the actual block concatenations
        return SymmetricArray.fuse(new, *axes_groups)

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
        new = SymmetricArray.unfuse(new, axis, inplace=True)

        if index.dual:
            # apply the phase changes
            if axes_flip:
                new.phase_flip(*axes_flip, inplace=True)
            new.phase_transpose(tuple(virtual_perm), inplace=True)

        return new

    def __matmul__(self, other):
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D arrays.")

        if not other.indices[0].dual:
            other = other.phase_flip(0)

        return super().__matmul__(other)

    def to_dense(self):
        """Return dense representation of the fermionic array, with lazy phases
        multiplied in.
        """
        return SymmetricArray.to_dense(self.phase_sync())

    def allclose(self, other, **kwargs):
        """Check if two fermionic arrays are element-wise equal within a
        tolerance, accounting for phases.

        Parameters
        ----------
        other : FermionicArray
            The other fermionic array to compare.
        """
        return SymmetricArray.allclose(self.phase_sync(), other.phase_sync())


@tensordot.register(FermionicArray)
def tensordot_fermionic(a, b, axes=2, **kwargs):
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
    perm_a = (*left_axes, *axes_a)
    perm_b = (*axes_b, *right_axes)
    a = a.transpose(perm_a)
    b = b.transpose(perm_b)
    #     but in terms of 'phase layout' =>  [..., x, y, z], [z, y, x, ...]
    b.phase_transpose(
        (*range(ncon - 1, -1, -1), *range(ncon, b.ndim)), inplace=True
    )

    # new axes for tensordot_symmetric having permuted inputs
    new_axes_a = tuple(range(ndim_a - ncon, ndim_a))
    new_axes_b = tuple(range(ncon))

    # if contracted index is like |x><x| phase flip to get <x|x>
    if a.size <= b.size:
        axs_flip = tuple(ax for ax in new_axes_a if a.indices[ax].dual)
        a.phase_flip(*axs_flip, inplace=True)
    else:
        axs_flip = tuple(ax for ax in new_axes_b if not b.indices[ax].dual)
        b.phase_flip(*axs_flip, inplace=True)

    # actually multiply block arrays with phases
    a.phase_sync(inplace=True)
    b.phase_sync(inplace=True)

    # perform blocked contraction!
    c = tensordot_symmetric(a, b, axes=(new_axes_a, new_axes_b), **kwargs)

    return c


# --------------- specific fermionic symmetric array classes ---------------- #


class Z2FermionicArray(FermionicArray):
    __slots__ = _fermionic_array_slots
    symmetry = get_symmetry("Z2")

    def to_pyblock3(self):
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
            pattern=["-" if f else "+" for f in self.duals],
        )

        data.shape = self.shape

        return data


class U1FermionicArray(FermionicArray):
    __slots__ = _fermionic_array_slots
    symmetry = get_symmetry("U1")
