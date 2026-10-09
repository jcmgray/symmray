"""Fermionic symmetric arrays with packed backend."""

from ..array_common import ArrayCommon, parse_tensordot_axes
from ..common import SymmrayCommon
from ..fermionic_common import (
    FermionicCommon,
    _annihilate_sorted_phase,
    _combine_phases,
    _koszul_sort_phase,
    parse_dummy_modes,
)
from ..fermionic_local_operators import FermionicOperator
from ..symmetries import get_symmetry
from ..utils import DEBUG, get_rng
from .kernels import get_kernels
from .packed_array_common import (
    PackedArrayCommon,
    cached_plan,
)
from .packed_data_common import PackedCommon
from .packed_vector import PackedVector


def calc_permutation_phases(indices, sectors, symmetry, axes, axes_flip):
    """Get the phase, +1 or -1, of each block from permuting its axes by
    ``axes`` and flipping ``axes_flip``. None if all +1. Each pair of odd
    charges that swap order gives -1, as does each flipped odd charge.
    """
    import numpy as np

    if not indices:
        return None
    parities = [ix.charge_parities(symmetry) for ix in indices]
    parity_offsets = np.zeros(len(indices), dtype=np.int64)
    np.cumsum([len(x) for x in parities[:-1]], out=parity_offsets[1:])
    return get_kernels().permutation_phases(
        np.concatenate(parities), parity_offsets, sectors, axes, axes_flip
    )


# marks that the lazy phases should be left as they are
_KEEP = object()


def _as_phases(phases):
    """Normalize lazy phases to an int8 array, or None."""
    import numpy as np

    return None if phases is None else np.asarray(phases, dtype=np.int8)


def _none_if_trivial(phases):
    """Return None if every phase is +1, else ``phases``."""
    if phases is None or (phases == 1).all():
        return None
    return phases


def _mul_phases(phases, other):
    """Combine two block phase arrays, either of which may be None."""
    if phases is None:
        return other
    if other is None:
        return phases
    return _none_if_trivial(phases * other)


class FermionicArrayPacked(
    FermionicCommon,
    PackedArrayCommon,
    PackedCommon,
    ArrayCommon,
    SymmrayCommon,
):
    """A fermionic block symmetric array with packed storage.

    Parameters
    ----------
    sectors : array_like
        Integer array, ``(num_blocks, ndim)``, of the charge id of each block
        on each axis. Rows must be unique and lexicographically sorted.
    data : array_like
        The 1D buffer of all blocks, in the order of ``sectors``.
    indices : tuple[PackedIndex]
        The indices of the array.
    charge : hashable, optional
        The total charge of the array.
    phases : array_like, optional
        The lazy phase, +1 or -1, of each block. None means all +1.
    label : hashable, optional
        An optional label for the array, potentially needed for ordering dummy
        odd fermionic modes.
    dummy_modes : sequence[FermionicOperator], optional
        Dummy fermionic modes effectively prepended to the array.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = (
        "_charge",
        "_data",
        "_dummy_modes",
        "_indices",
        "_label",
        "_offsets",
        "_phases",
        "_sectors",
        "_shapes",
        "_skey",
        "_symmetry",
    )
    fermionic = True
    static_symmetry = None

    def __init__(
        self,
        sectors,
        data,
        indices,
        charge=None,
        phases=None,
        label=None,
        dummy_modes=None,
        symmetry=None,
    ):

        self._init_abelian(
            sectors=sectors,
            data=data,
            indices=indices,
            charge=charge,
            symmetry=symmetry,
            label=label,
        )
        self._phases = _as_phases(phases)
        self._dummy_modes = parse_dummy_modes(
            parity=self.parity,
            label=self.label,
            dummy_modes=dummy_modes,
            warn_odd=True,
            prune_even=True,
        )
        if DEBUG:
            self.check()

    def check(self):
        """Check the internal consistency of the array."""
        self._check_abelian()
        if self._phases is not None:
            assert self._phases.shape == (self.num_blocks,)
            assert set(self._phases.tolist()) <= {-1, 1}

    @property
    def parity(self):
        """The parity of the total charge."""
        return self._symmetry.parity(self._charge)

    @property
    def phases(self):
        """The lazy phase of each block, as an array."""
        import numpy as np

        if self._phases is None:
            return np.ones(self.num_blocks, dtype=np.int8)
        return self._phases

    def copy(self) -> "FermionicArrayPacked":
        """Copy this array. The data is shared and treated as immutable."""
        new = self._copy_abelian()
        new._phases = self._phases
        new._dummy_modes = self._dummy_modes
        return new

    def new_with(
        self, indices, charge, data, sectors, shapes, offsets, label=None
    ) -> "FermionicArrayPacked":
        """Create a new array of the same class, with no lazy phases or dummy
        modes.
        """
        new = self._new_with_abelian(
            indices=indices,
            charge=charge,
            data=data,
            sectors=sectors,
            shapes=shapes,
            offsets=offsets,
            label=label,
        )
        new._phases = None
        new._dummy_modes = ()
        return new

    def copy_with(self, phases=_KEEP, **kwargs) -> "FermionicArrayPacked":
        """Copy this array with some attributes replaced, for internal use."""
        new = self.copy()
        new._modify_abelian(**kwargs)
        if phases is not _KEEP:
            new._phases = _as_phases(phases)
        if DEBUG:
            new.check()
        return new

    def modify(
        self, phases=_KEEP, dummy_modes=None, **kwargs
    ) -> "FermionicArrayPacked":
        """Modify this array inplace, for internal use."""
        self._modify_abelian(**kwargs)
        if phases is not _KEEP:
            self._phases = _as_phases(phases)
        if dummy_modes is not None:
            self._dummy_modes = dummy_modes
        if DEBUG:
            self.check()
        return self

    @classmethod
    def from_blocksparse(cls, x, symmetry=None) -> "FermionicArrayPacked":
        """Create a packed array from a block sparse `FermionicArray`."""
        indices, sectors, data, phases = cls._parts_from_blocksparse(x)
        return cls(
            sectors=sectors,
            data=data,
            indices=indices,
            charge=x.charge,
            phases=phases,
            label=x.label,
            dummy_modes=x.dummy_modes,
            symmetry=symmetry if symmetry is not None else x.symmetry,
        )

    def to_blocksparse(self):
        """Convert to a block sparse `FermionicArray`."""
        from ..utils import get_array_cls

        indices, blocks, phases = self._to_blocksparse_parts()
        cls = get_array_cls(self.symmetry, fermionic=True, layout="sparse")
        return cls(
            indices=indices,
            charge=self.charge,
            blocks=blocks,
            phases=phases,
            label=self.label,
            dummy_modes=self.dummy_modes,
            symmetry=self.symmetry,
        )

    def to_pytree(self):
        """Convert to a pytree purely of non-symmray objects."""
        return self.to_blocksparse().to_pytree()

    # -------------------------------- phases ------------------------------- #

    def _permutation_phases(self, axes, axes_flip=()):
        """The phase of each block from permuting by ``axes`` and flipping
        ``axes_flip``, cached by structure.
        """
        return cached_plan(
            ("permphase", self.structure_key(), tuple(axes), tuple(axes_flip)),
            calc_permutation_phases,
            self._indices,
            self._sectors,
            self._symmetry,
            tuple(axes),
            tuple(axes_flip),
        )

    def randomize_phases(self, seed=None, inplace=False):
        """Randomize the lazy phase of each block, for testing."""
        import numpy as np

        rng = get_rng(seed)
        phases = np.where(rng.uniform(size=self.num_blocks) > 0.5, -1, 1)
        new = self if inplace else self.copy()
        new._phases = phases.astype(np.int8)
        return new

    def phase_sync(self, inplace=False):
        """Multiply all lazy phases into the data."""
        new = self if inplace else self.copy()
        if new._phases is not None:
            if (new._phases == -1).any():
                new._data = get_kernels().run_scale(
                    new._data, new._offsets, new._phases
                )
            new._phases = None
        return new

    def phase_flip(self, *axs, inplace=False):
        """Flip the phase of all blocks with odd parity summed over ``axs``."""
        new = self if inplace else self.copy()
        if not axs:
            return new
        axs = tuple(ax % self.ndim for ax in axs)
        phases = self._permutation_phases(tuple(range(self.ndim)), axs)
        new._phases = _mul_phases(new._phases, phases)
        return new

    def phase_transpose(self, axes=None, inplace=False):
        """Phase as if transposed by ``axes``, without moving any data."""
        new = self if inplace else self.copy()
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        phases = self._permutation_phases(tuple(axes))
        new._phases = _mul_phases(new._phases, phases)
        return new

    def phase_sector(self, sector, inplace=False):
        """Flip the phase of a specific sector."""
        new = self if inplace else self.copy()
        i = new._find_block(sector)
        if i >= 0:
            phases = new.phases.copy()
            phases[i] = -phases[i]
            new._phases = phases
        return new

    def phase_global(self, inplace=False, *, parity=1):
        """Flip the phase of every block, if ``parity`` is odd."""
        new = self if inplace else self.copy()
        if parity % 2 == 0:
            return new
        new._phases = -new.phases
        return new

    def _phase_permuted_lazy(self, axes, axes_flip=()):
        """Copy with the phases from transposing by ``axes`` and flipping
        ``axes_flip`` added to the lazy phases. No data is moved.
        """
        new = self.copy()
        if axes_flip or axes != tuple(range(self.ndim)):
            phases = self._permutation_phases(axes, axes_flip)
            new._phases = _mul_phases(new._phases, phases)
        return new

    def _resolve_dummy_modes_combine(self, left, right):
        """Calculate the new combined dummy odd modes and any associated global
        phases from contracting ``left`` and ``right``, inplace.
        """
        l_dummy_modes = left.dummy_modes
        r_dummy_modes = right.dummy_modes

        if not l_dummy_modes and not r_dummy_modes:
            self._dummy_modes = ()
            return

        dummy_modes = [*l_dummy_modes, *r_dummy_modes]

        # merged as l-dummies, l-sectors, r-dummies, r-sectors. Moving
        # r-dummies left over l-sectors gives a sign
        if left.parity and right.dummy_parity:
            phase = -1
        else:
            phase = 1

        # sort merged dummy modes into canonical label order, the sign of this
        # sort is the Koszul sign
        perm = sorted(range(len(dummy_modes)), key=dummy_modes.__getitem__)
        swap_phase = _koszul_sort_phase(dummy_modes, self.backend)
        dummy_modes = [dummy_modes[k] for k in perm]

        # trace out conjugate pairs, picking up the sign of each crossing
        dummy_modes, trace_phase = _annihilate_sorted_phase(
            dummy_modes, self.backend
        )

        if _combine_phases(phase, swap_phase, trace_phase) == -1:
            self.phase_global(inplace=True)

        self._dummy_modes = tuple(dummy_modes)

    def _resolve_dummy_modes_squeeze(self, axes_squeeze):
        """Turn any odd charges on ``axes_squeeze`` into dummy modes, with the
        phases from moving them to the front. Call just before squeezing those
        axes.
        """
        axes_leave = []
        squeezed_dummy_modes = []
        for ax, ix in enumerate(self.indices):
            if ax in axes_squeeze:
                (c,) = ix.charges
                if self.symmetry.parity(c):
                    label = self.label
                    if label is None:
                        raise ValueError(
                            "Cannot squeeze fermionic index with odd parity "
                            "if array has no ordering `.label` attribute."
                        )
                    op = FermionicOperator(("squeeze", label, ax), ix.dual)
                    squeezed_dummy_modes.append(op)
            else:
                axes_leave.append(ax)

        if squeezed_dummy_modes:
            self.phase_transpose((*axes_squeeze, *axes_leave), inplace=True)
            self.modify(dummy_modes=(*self.dummy_modes, *squeezed_dummy_modes))

    def _prepare_for_tensordot_fermionic(self, other, axes):
        """Add the phases of transposing to ``[..., x, y, z], [z, y, x, ...]``
        as lazy phases. The packed tensordot applies them in the copies that
        fuse each side.
        """
        left_axes, axes_a, axes_b, right_axes = parse_tensordot_axes(
            axes, self.ndim, other.ndim
        )

        # if contracted index is like |x><x| phase flip to get <x|x>. Either
        # side gives the same result, so flip the smaller one
        if self.size <= other.size:
            flip_a = tuple(ax for ax in axes_a if not self.indices[ax].dual)
            flip_b = ()
        else:
            flip_a = ()
            flip_b = tuple(ax for ax in axes_b if other.indices[ax].dual)

        a = self._phase_permuted_lazy((*left_axes, *axes_a), flip_a)
        b = other._phase_permuted_lazy(
            (*reversed(axes_b), *right_axes), flip_b
        )
        return a, b, axes_a, axes_b

    def _fuse_core(self, *axes_groups, inplace=False):
        """Fermionic fusion of axes groups, see the sparse `_fuse_core`. Phases
        are added lazily and applied in the fusing copy.
        """
        from ..sparse.sparse_array_common import calc_fuse_group_info

        # the transpose making each group contiguous
        perm = list(calc_fuse_group_info(axes_groups, self.duals)[3])
        axes_flip = []
        for group in axes_groups:
            if self.indices[group[0]].dual:
                # overall dual group: flip its non dual axes and reverse it
                axes_flip.extend(
                    ax for ax in group if not self.indices[ax].dual
                )
                start = perm.index(group[0])
                perm[start : start + len(group)] = reversed(group)

        x = self._phase_permuted_lazy(tuple(perm), tuple(axes_flip))
        if inplace:
            self._become(x)
            x = self
        return x._fuse_core_abelian(*axes_groups, inplace=True)

    def transpose(self, axes=None, phase=True, inplace=False):
        """Transpose the fermionic array, by default including the phases from
        swapping odd charges.
        """
        if axes is None:
            axes = tuple(range(self.ndim - 1, -1, -1))
        else:
            axes = tuple(axes)
        new = self if inplace else self.copy()
        if phase:
            new._phases = _mul_phases(
                new._phases, self._permutation_phases(axes)
            )
        return new._transpose_abelian(axes=axes, inplace=True)

    def conj(
        self,
        phase_permutation=True,
        phase_dual=False,
        inner_dummy_labels=(),
        inplace=False,
    ):
        """Conjugate this fermionic array, see the sparse `conj`."""
        new = self if inplace else self.copy()
        axes, phase_dummy = new._parse_phase_dual(phase_dual)
        new_indices = tuple(ix.conj() for ix in new.indices)
        phased_axes = tuple(ax for ax in axes if not new_indices[ax].dual)

        if phase_permutation:
            # conj reverses the axis order
            new._phases = _mul_phases(
                new._phases,
                self._permutation_phases(
                    tuple(range(self.ndim - 1, -1, -1)), phased_axes
                ),
            )
        elif phased_axes:
            new._phases = _mul_phases(
                new._phases,
                self._permutation_phases(tuple(range(self.ndim)), phased_axes),
            )

        new._conj_abelian(inplace=True)
        new._resolve_dummy_modes_conj(
            phase_permutation=phase_permutation,
            phase_dummy=phase_dummy,
            inner_dummy_labels=inner_dummy_labels,
        )
        return new

    def _phase_project_fermionic(self, axes, inplace=False):
        """Apply the phase-only correction for a fermionic projector, see the
        sparse `_phase_project_fermionic`.
        """
        import numpy as np

        new = self if inplace else self.copy()
        reference_dual = new.indices[axes[0]].dual
        kept_axes = set(axes)
        axes_flip = tuple(
            ax
            for ax, ix in enumerate(new.indices)
            if (ax not in kept_axes) and (ix.dual is reference_dual)
        )
        const = reference_dual * new.dummy_parity
        phases = self._permutation_phases(tuple(range(self.ndim)), axes_flip)
        if const % 2:
            if phases is None:
                phases = np.full(self.num_blocks, -1, dtype=np.int8)
            else:
                phases = -phases
        new._phases = _mul_phases(new._phases, phases)
        return new

    def dagger(self, phase_dual=False, inner_dummy_labels=(), inplace=False):
        """Return the fermionic conjugate transpose, see the sparse `dagger`."""
        new = self if inplace else self.copy()
        axes, phase_dummy = new._parse_phase_dual(phase_dual)

        # reverse axes, applying any lazy phases, then conj
        new._transpose_abelian(inplace=True)
        new._conj_abelian(inplace=True)
        new_indices = new.indices

        phased_axes = tuple(
            new.ndim - 1 - ax
            for ax in axes
            if not new_indices[new.ndim - 1 - ax].dual
        )
        if phased_axes:
            new.phase_flip(*phased_axes, inplace=True)

        new._resolve_dummy_modes_conj(
            phase_permutation=True,
            phase_dummy=phase_dummy,
            inner_dummy_labels=inner_dummy_labels,
        )
        return new

    def eigh(self, phase_eigenvalues=True, **kwargs):
        """Eigen decompose this array, which should be Hermitian, see the
        sparse `eigh`.
        """
        import numpy as np

        w, u = self._eigh_abelian(**kwargs)

        if phase_eigenvalues and (not self._indices[1].dual):
            # put phase of odd charges into eigenvalues, so that
            # U @ diag(w) @ U.H == a always

            symm = self._symmetry
            odd = symm.parity_array(symm.charges_to_array(w.charges))
            signs = np.repeat(1 - 2 * odd, np.diff(w.offsets))
            w = PackedVector.from_parts(w.charges, w.data * signs, w.offsets)

        return w, u


# --------------------------------------------------------------------------- #


class Z2FermionicArrayPacked(FermionicArrayPacked):
    """A packed fermionic array with Z2 symmetry."""

    static_symmetry = get_symmetry("Z2")


class U1FermionicArrayPacked(FermionicArrayPacked):
    """A packed fermionic array with U1 symmetry."""

    static_symmetry = get_symmetry("U1")


class Z2Z2FermionicArrayPacked(FermionicArrayPacked):
    """A packed fermionic array with Z2 x Z2 symmetry."""

    static_symmetry = get_symmetry("Z2Z2")


class U1U1FermionicArrayPacked(FermionicArrayPacked):
    """A packed fermionic array with U1 x U1 symmetry."""

    static_symmetry = get_symmetry("U1U1")
