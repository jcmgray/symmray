"""Common methods for any fermionic arrays."""

import autoray as ar

from .array_common import _normalize_axes, parse_tensordot_axes
from .fermionic_local_operators import FermionicOperator
from .linalg_common import Absorb


def parse_dummy_modes(
    parity,
    label,
    dummy_modes,
    warn_odd=False,
    prune_even=False,
):
    """Parse the dummy modes for a fermionic array, possibly creating a single
    mode from the array label to mimic an overall even parity array.
    """
    if isinstance(dummy_modes, (list, tuple)):
        if len(dummy_modes) == 0:
            # can't check first element
            return ()
        elif isinstance(dummy_modes[0], FermionicOperator):
            # explicit sequence of dummy modes already given, assume correct
            return tuple(dummy_modes)

    if dummy_modes is None:
        if label is None:
            if warn_odd and parity:
                import warnings

                warnings.warn(
                    "This fermionic array has odd parity but no `label` "
                    "to create a matching dummy mode, and no explicitly "
                    "specified `dummy_modes`. Global phase may be incorrect.",
                )

            # no dummy modes, assume even parity
            return ()
        else:
            if prune_even and (not parity):
                # even parity array, no dummy modes needed
                return ()

            # else create a default single dummy mode matching the array
            # parity to form an overall even parity array
            return (FermionicOperator(label, parity=parity),)

    # XXX: might be other valid specifications in future
    raise ValueError(
        "Invalid dummy_modes specification, must be "
        "None or a sequence of FermionicOperator."
    )


def _koszul_sort_phase(modes, backend):
    """Fermionic (Koszul) sign from sorting ``modes`` into ascending label
    order: ``(-1) ** K`` with ``K = sum_{i<j, modes[j] < modes[i]} p_i p_j``.

    Mode *labels* are static, so the inverted pairs are enumerated at trace
    time. Mode *parities* ``p`` may be tracer arrays, so the parity vectors of
    the inverted pairs are stacked and reduced as ``K = sum(p_i * p_j)`` in a
    few vectorized ops, rather than unrolling O(n^2) scalar multiplies into the
    graph resulting in slow compile time. Parities that are plain ints
    (possibly mixed with traced ones) are folded in as python data, both since
    they can simplify away and since e.g. ``torch.stack`` rejects non-tensors.

    Returns the plain int ``1`` when no inverted pair can contribute.
    """
    n = len(modes)
    if n < 2:
        return 1

    static = tuple(isinstance(m.parity, int) for m in modes)
    # a statically-even parity contributes no sign -> we can just ignore
    static_zero = tuple(
        static[i] and (modes[i].parity % 2 == 0) for i in range(n)
    )

    # enumerate inverted pairs (static), partitioning their parity products
    #     K = sum_{(i, j) in pairs} p_i * p_j
    # by which factors are static, so each stack is free of plain ints:
    #     both static -> sum in python
    #     one static parity: must be odd, so p_i * p_j == other (mod 2)
    #     both traced -> stack and multiply as vectors
    K = 0
    singles = []
    parities_i = []
    parities_j = []
    for i in range(n):
        if static_zero[i]:
            # can ignore i
            continue
        mi = modes[i]
        for j in range(i + 1, n):
            mj = modes[j]
            if static_zero[j] or (mi < mj):
                # can ignore j, or not inverted (labels unique -> no ties)
                continue
            if static[i] and static[j]:
                # can eagerly accrue parity product
                K += mi.parity * mj.parity
            elif static[i]:
                # parity product is just traced j
                singles.append(mj.parity)
            elif static[j]:
                # parity product is just traced i
                singles.append(mi.parity)
            else:
                # both traced, need to stack and multiply
                parities_i.append(mi.parity)
                parities_j.append(mj.parity)

    return _reduce_parity_terms(K, singles, parities_i, parities_j, backend)


def _reduce_parity_terms(K, singles, parities_i, parities_j, backend):
    """Evaluate the sign ``(-1) ** K``, where ``K`` is the static count already
    accrued, plus the traced parities in ``singles``, plus the products of the
    traced parities paired up in ``parities_i`` and ``parities_j``. The traced
    terms are stacked and reduced in a few vectorized ops, rather than unrolled
    into the graph.

    Returns the plain int ``1`` if there is nothing traced to add.
    """
    if not (singles or parities_i):
        # no traced contributions -> evaluate the sign statically
        return -1 if (K % 2) else 1

    if singles:
        pij = ar.do("stack", singles, like=backend)
        K = K + ar.do("sum", pij, like=backend)

    if parities_i:
        pi = ar.do("stack", parities_i, like=backend)
        pj = ar.do("stack", parities_j, like=backend)
        K = K + ar.do("sum", pi * pj, like=backend)

    return (K % 2) * -2 + 1


def _annihilate_sorted_phase(modes, backend):
    """Trace out conjugate pairs from an already sorted sequence of ``modes``,
    returning the surviving modes and the fermionic sign this produces.

    Sorted order places the dual modes first, by descending label, then the
    non-dual modes by ascending label, so the two halves of a pair sit at
    mirrored positions, and any other pair between them is fully nested.
    Tracing out a pair moves its right half left onto its left half, giving
    ``(-1) ** (p * p_k)`` for each mode ``k`` it crosses. A nested pair
    contributes both of its halves, so ``2 * p * p_k`` is even and cancels,
    leaving only the unpaired modes in between.

    Which modes pair up follows from their labels and dualnesses, which are
    static, so the surviving modes are known at trace time even when the
    parities are tracer arrays. Only the sign depends on the parities, and it
    is computed vectorized, see `_reduce_parity_terms`.
    """
    positions = {}
    for k, m in enumerate(modes):
        positions.setdefault(m.label, []).append(k)

    # which modes pair up is static, so is the set of survivors
    pairs = []
    traced_out = set()
    for ks in positions.values():
        if len(ks) == 1:
            continue
        if len(ks) != 2 or modes[ks[0]].dual == modes[ks[1]].dual:
            raise ValueError("`dummy_modes` must be unique conjugate pairs.")
        pairs.append(ks)
        traced_out.update(ks)

    if not pairs:
        # nothing to trace out, e.g. an amplitude contraction
        return modes, 1

    # accrue the crossing terms, partitioned as in `_koszul_sort_phase`
    K = 0
    singles = []
    parities_i = []
    parities_j = []

    for i, j in pairs:
        pp = modes[j].parity
        static_pp = isinstance(pp, int)
        if static_pp and (pp % 2 == 0):
            # a statically-even pair contributes no sign
            continue
        for k in range(i + 1, j):
            if k in traced_out:
                # nested pair, both halves cross, so the sign cancels
                continue
            pk = modes[k].parity
            static_pk = isinstance(pk, int)
            if static_pk and (pk % 2 == 0):
                continue
            if static_pp and static_pk:
                K += pp * pk
            elif static_pp:
                singles.append(pk)
            elif static_pk:
                singles.append(pp)
            else:
                parities_i.append(pp)
                parities_j.append(pk)

    modes = [m for k, m in enumerate(modes) if k not in traced_out]
    phase = _reduce_parity_terms(K, singles, parities_i, parities_j, backend)
    return modes, phase


class FermionicCommon:
    @property
    def dummy_modes(self) -> tuple[FermionicOperator, ...]:
        """A sequence of dummy fermionic modes effectively prepended to the
        array, used to describe odd parity arrays.
        """
        return self._dummy_modes

    @property
    def dummy_parity(self):
        """The combined parity of the dummy modes."""
        return sum(mode.parity for mode in self.dummy_modes) % 2

    def _do_unary_op(self, fn, inplace=False) -> "FermionicCommon":
        """Need to sync phases before applying an elementwise function, which
        does not generally commute with them.

        This is used by `abs`, `sqrt`, `clip` and others.
        """
        new = self.phase_sync(inplace=inplace)
        return new._do_unary_op_blockwise(fn, inplace=True)

    def _do_reduction(self, fn):
        """Need to sync phases before reducing over all elements.

        This is used by `sum`, `max`, `min` and others.
        """
        return self.phase_sync()._do_reduction_blockwise(fn)

    def get_scalar_element(self):
        """Assuming the array is a scalar, get that scalar element, with any
        lazy phase resolved first.
        """
        return self.phase_sync()._get_scalar_element_blockwise()

    def item(self):
        """Convert this array to a scalar, if it is a scalar array, with any
        lazy phase resolved first.
        """
        return self.phase_sync()._item_blockwise()

    def _binary_blockwise_op(self, other, fn, inplace=False, **kwargs):
        """Need to sync phases before performing blockwise operations.

        This is used across many basic methods defined in `AbelianArray` such
        as `__add__`, `__imul__` etc.
        """
        xy = self.phase_sync(inplace=inplace)

        if isinstance(other, FermionicCommon):
            other = other.phase_sync()

        return xy._binary_blockwise_op_abelian(
            other, fn, inplace=True, **kwargs
        )

    def _fuse_core(
        self,
        *axes_groups,
        inplace=False,
    ) -> "FermionicCommon":
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
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FermionicArray
        """
        from .sparse.sparse_array_common import calc_fuse_group_info

        x = self if inplace else self.copy()

        # first make groups into contiguous
        # blocks using fermionic transpose
        perm = calc_fuse_group_info(axes_groups, x.duals)[3]
        # this is the first step which introduces phases
        x.transpose(perm, inplace=True)
        # update groups to reflect new axes
        axes_groups = tuple(tuple(map(perm.index, g)) for g in axes_groups)

        # process each group with another two sources of phase changes:
        axes_flip = []
        virtual_perm = None
        for group in axes_groups:
            if x.indices[group[0]].dual:
                # overall dual index:
                # 1. flip non dual sub indices
                for ax in group:
                    if not x.indices[ax].dual:
                        axes_flip.append(ax)

                # 2. virtual transpose within group
                if virtual_perm is None:
                    virtual_perm = list(range(x.ndim))
                for axi, axj in zip(group, reversed(group)):
                    virtual_perm[axi] = axj

        if axes_flip:
            x.phase_flip(*axes_flip, inplace=True)

        # if the fused axes is overall bra, need phases from effective flip
        #   <a|<b|<c|  |a>|b>|c>    ->    P * <c|<b|<a|  |a>|b>|c>
        #   but actual array layout should not be flipped, so do virtually
        if virtual_perm is not None:
            x.phase_transpose(tuple(virtual_perm), inplace=True)

        # insert phases
        x.phase_sync(inplace=True)

        # so we can do the actual block concatenations
        x._fuse_core_abelian(*axes_groups, inplace=True)

        return x

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
        new._unfuse_abelian(axis, inplace=True)

        if index.dual:
            # apply the phase changes
            if axes_flip:
                new.phase_flip(*axes_flip, inplace=True)
            new.phase_transpose(tuple(virtual_perm), inplace=True)

        return new

    def einsum(self, eq, preserve_array=False):
        """Einsum for fermionic arrays, currently only single term.

        Parameters
        ----------
        eq : str
            The einsum equation, e.g. "abcb->ca". The output indices must be
            specified and only trace and permutations are allowed.
        preserve_array : bool, optional
            If tracing to a scalar, whether to return an AbelianArray object
            with no indices, or simply scalar itself (the default).

        Returns
        -------
        FermionicCommon or scalar
        """
        lhs, rhs = eq.split("->")

        def key(i):
            c = lhs[i]
            return (
                # group traced then kept indices
                rhs.find(c),
                # pair up traced indices
                c,
                # make sure traced pairs grouped like (-+)
                not self.indices[i].dual,
            )

        # transposition introduces all necessary phases
        perm = tuple(sorted(range(self.ndim), key=key))
        x = self.transpose(perm)
        x.phase_sync(inplace=True)

        # then can use AbelianArray einsum
        new_lhs = "".join(lhs[i] for i in perm)
        new_eq = f"{new_lhs}->{rhs}"

        return x._einsum_abelian(new_eq, preserve_array=preserve_array)

    def trace(self):
        """Fermionic matrix trace."""
        ixl, ixr = self.indices

        if ixl.dual and not ixr.dual:
            return self.phase_sync()._trace_abelian()
        elif not ixl.dual and ixr.dual:
            return self.phase_flip(0).phase_sync(inplace=True)._trace_abelian()
        else:
            raise ValueError("Cannot trace a non-bra or non-ket.")

    def _prepare_for_tensordot_fermionic(self, other, axes):
        """Perform necessary fermionic phase operations to prepare two arrays
        for an abelian tensordot.
        """
        ndim_a, ndim_b = self.ndim, other.ndim
        left_axes, axes_a, axes_b, right_axes = parse_tensordot_axes(
            axes, ndim_a, ndim_b
        )

        ncon = len(axes_a)

        # XXX: do all three as virtual phases?

        # permute a & b so we have axes like
        #     in terms of data layout => [..., x, y, z], [x, y, z, ...]
        a = self.transpose((*left_axes, *axes_a))
        b = other.transpose((*axes_b, *right_axes))
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

        return a, b, new_axes_a, new_axes_b

    def tensordot(
        self, other, axes=2, preserve_array=False, **kwargs
    ) -> "FermionicCommon":
        """Contract two fermionic arrays along the specified axes, accounting
        for phases from both transpositions and contractions.

        Parameters
        ----------
        a : FermionicArray
            The first fermionic array.
        b : FermionicArray
            The second fermionic array.
        axes : int or (tuple[int], tuple[int]), optional
            The axes to contract over, by default 2.
        preserve_array : bool, optional
            Whether to preserve the array structure even if the result is a
            scalar, by default False.
        kwargs
            Passed to the underlying (non-fermionic) tensordot call.

        Returns
        -------
        FermionicCommon or scalar
        """

        if not isinstance(other, self.__class__):
            if getattr(other, "ndim", 0) == 0:
                # assume scalar
                return self * other
            else:
                raise TypeError(
                    f"Expected {self.__class__}, got {other.__class__}."
                )

        # make modes contiguous and sync phase etc.
        a, b, new_axes_a, new_axes_b = self._prepare_for_tensordot_fermionic(
            other, axes
        )

        # perform blocked contraction!
        c = a._tensordot_abelian(
            b,
            axes=(new_axes_a, new_axes_b),
            # preserve array for resolving dummy_modes
            preserve_array=True,
            **kwargs,
        )

        c._resolve_dummy_modes_combine(a, b)

        if (c.ndim == 0) and (not preserve_array):
            return c.get_scalar_element()

        return c

    def __matmul__(self, other: "FermionicCommon", preserve_array=False):
        """Matrix or vector multiplication, accounting for fermionic
        phases.

        Parameters
        ----------
        other : FermionicCommon
            The other fermionic array to multiply.
        preserve_array : bool, optional
            Whether to preserve the array structure even if the result is a
            scalar, by default False.
        """
        if self.ndim > 2 or other.ndim > 2:
            raise ValueError("Matrix multiplication requires <=2D arrays.")

        if other.indices[0].dual:
            # have |x><x| -> want <x|x>
            other = other.phase_flip(0)

        a = self.phase_sync()
        b = other.phase_sync()
        c = a._matmul_abelian(b, preserve_array=True)
        c._resolve_dummy_modes_combine(a, b)

        if c.ndim == 0 and (not preserve_array):
            return c.get_scalar_element()

        return c

    def to_dense(self, index_maps=None):
        """Return dense representation of the fermionic array, with lazy phases
        multiplied in.

        Parameters
        ----------
        index_maps : Sequence[Sequence[hashable]], optional
            For each dimension, a sequence mapping each output linear index
            to a charge sector. Repeated charges select degeneracy offsets in
            occurrence order. If not supplied, charge sectors are packed in
            sorted contiguous order.
        """
        return self.phase_sync()._to_dense_abelian(index_maps=index_maps)

    def squeeze(self, axis=None, inplace=False) -> "FermionicCommon":
        """Squeeze the fermionic array, removing axes of size 1. If those axes
        correspond to odd parity charges, then they are converged into dummy
        `dummy_modes` modes effectively to the left of the array. The sorting
        `label` of the array is then required to have been set.

        Parameters
        ----------
        axis : int or tuple[int], optional
            The axis or axes to squeeze. If None, all axes of size 1 are
            removed, by default None.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FermionicCommon
        """
        new = self if inplace else self.copy()

        if ar.is_scalar(axis):
            axis = (axis,)

        axes_squeeze = []
        for ax, ix in enumerate(new.indices):
            if axis is None:
                remove = ix.size_total == 1
            else:
                remove = ax in axis
                if remove and ix.size_total > 1:
                    raise ValueError("Cannot squeeze d > 1 index")
            if remove:
                axes_squeeze.append(ax)

        if not axes_squeeze:
            # nothing to do
            return new

        # this takes care of phases from moving the squeezed axes to the
        # beginning of the array, and also turns them into dummy modes
        new._resolve_dummy_modes_squeeze(axes_squeeze)

        # actually do the data squeeze
        return new._squeeze_abelian(axes_squeeze, inplace=True)

    def dagger_compose_left(self) -> "FermionicCommon":
        """Take the dagger (conjugate transpose) of this fermionic array,
        assuming we have the right factor of a hermitian decomposition and want
        the left factor.
        """
        new = self._dagger_abelian()
        if not new.indices[-1].dual:
            # 'inner' index is like |x><x| so introduce a phase flip
            new.phase_flip(-1, inplace=True)
        return new

    def dagger_project_left(self) -> "FermionicCommon":
        """Take the dagger (conjugate transpose) of this fermionic array,
        assuming we are going to use to project from the left on another
        operator.
        """
        new = self.dagger()
        return new._phase_project_fermionic((0,), inplace=True)

    def dagger_compose_right(self) -> "FermionicCommon":
        """Take the dagger (conjugate transpose) of this fermionic array,
        assuming we have the left factor of a hermitian decomposition and want
        the right factor.
        """
        new = self._dagger_abelian()
        if new.indices[0].dual:
            # 'inner' index is like |x><x| so introduce a phase flip
            new.phase_flip(0, inplace=True)
        return new

    def dagger_project_right(self) -> "FermionicCommon":
        """Take the dagger (conjugate transpose) of this fermionic array,
        assuming we are going to use to project from the right on another
        operator.
        """
        new = self.dagger()
        return new._phase_project_fermionic((new.ndim - 1,), inplace=True)

    def conj_project(self, axes=-1, inplace=False) -> "FermionicCommon":
        """Conjugate this array for use with itself as a projector.

        The selected axes carry the uncontracted bonds. All other axes are
        contracted back into the tensor network. The first selected axis
        sets the reference duality for the fermionic phase correction.

        Parameters
        ----------
        axes : int or sequence of int, optional
            The axes carrying the uncontracted bonds. A single integer can be
            supplied when keeping one axis.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FermionicCommon
        """
        axes = _normalize_axes(axes, self.ndim)
        new = self.conj(inplace=inplace)
        return new._phase_project_fermionic(axes, inplace=True)

    def allclose(self, other, **kwargs):
        """Check if two fermionic arrays are element-wise equal within a
        tolerance, accounting for phases.

        Parameters
        ----------
        other : FermionicArray
            The other fermionic array to compare.
        """
        return self.phase_sync()._allclose_abelian(
            other.phase_sync(), **kwargs
        )

    def test_allclose(self, other, **allclose_opts):
        """Assert that this ``SparseArrayCommon`` is close to another,
        that is, has all the same sectors, and the corresponding arrays are
        close. Unlike `allclose`, this raises an AssertionError with details
        if not.

        Parameters
        ----------
        other : SparseArrayCommon
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Raises
        ------
        AssertionError
            If the arrays are not close.
        """
        return self.phase_sync()._test_allclose_abelian(
            other.phase_sync(), **allclose_opts
        )

    # --------------------------- linalg methods ---------------------------- #

    def _split(
        self,
        *args,
        charge_side="auto",
        drop_dummy_modes="auto",
        **kwargs,
    ):
        """Fermionic array splitting, involving a phase sync, the abelian split
        which is handled by the backend, and then a possible phase flip
        depending of the dualness of the inner bond.

        Parameters
        ----------
        drop_dummy_modes : {"auto", bool}, optional
            Whether to drop ``dummy_modes`` (and the array ``label``) from the
            returned factors. Useful is using the factors as projectors rather
            than replacements for the original array. If "auto", will drop if
            the method is "eigh".
        """
        x = self.phase_sync()

        left, s, right = x._split_abelian(*args, **kwargs)

        # check if inner index is like |x><x| and needs a phase flip
        charge_side = Absorb.choose_charge_side(kwargs["absorb"], charge_side)
        if charge_side == "left":
            if right is not None and right.indices[0].dual:
                right.phase_flip(0, inplace=True)
        else:  # charge_side == "right"
            if left is not None and not left.indices[-1].dual:
                left.phase_flip(-1, inplace=True)

        if drop_dummy_modes == "auto":
            # XXX: push this logic up into e.g. quimb
            # compression algorithms explicitly?
            drop_dummy_modes = kwargs.get("method") == "eigh"
        if drop_dummy_modes:
            for factor in (left, right):
                if factor is not None:
                    factor._dummy_modes = ()
                    factor._label = None

        return left, s, right

    def cholesky(
        self,
        *,
        upper=False,
        drop_dummy_modes="auto",
    ) -> "FermionicCommon":
        """Cholesky decomposition of a fermionic array.

        Parameters
        ----------
        upper : bool, optional
            Whether to return the upper triangular Cholesky factor.
            Default is False, returning the lower triangular factor.
        drop_dummy_modes : {"auto", bool}, optional
            Whether to drop ``dummy_modes`` (and the array ``label``) from the
            returned factors. Useful is using the factors as projectors rather
            than replacements for the original array. If "auto", will drop.

        Returns
        -------
        l_or_r : FermionicCommon
            The Cholesky factor. Lower triangular if ``upper=False``,
            upper triangular if ``upper=True``.
        """
        x = self.phase_sync()
        l_or_r = x._cholesky_abelian(upper=upper)
        if upper and l_or_r.indices[0].dual:
            # inner index is like |x><x| so introduce a phase flip
            l_or_r.phase_flip(0, inplace=True)
        if drop_dummy_modes == "auto" or drop_dummy_modes:
            # cholesky output is always one half of a hermitian sandwich
            l_or_r._dummy_modes = ()
            l_or_r._label = None
        return l_or_r

    def cholesky_regularized(
        self,
        absorb=0,
        shift=True,
        drop_dummy_modes="auto",
    ) -> "FermionicCommon":
        """Cholesky decomposition with optional diagonal regularization,
        returning results in an SVD-like ``(left, None, right)`` format
        for compatibility with tensor network split drivers. Handles
        fermionic phase synchronization automatically.

        Parameters
        ----------
        absorb : {-12, 0, 12}, optional
            How to return the factors:

            - ``0`` (``'both'``): return ``(L, None, L^H)``.
            - ``-12`` (``'lsqrt'``): return ``(L, None, None)``.
            - ``12`` (``'rsqrt'``): return ``(None, None, L^H)``.

        shift : float, optional
            Diagonal regularization shift. If True or negative, auto-compute
            from dtype machine epsilon. The shift is always applied as a
            relative shift scaled by the trace of each block. Default is True.
        drop_dummy_modes : {"auto", bool}, optional
            Whether to drop ``dummy_modes`` (and the array ``label``) from the
            returned factors. Useful is using the factors as projectors rather
            than replacements for the original array. If "auto", will drop.

        Returns
        -------
        left : FermionicCommon or None
            The lower Cholesky factor, or None.
        s : None
            Always None (no singular values).
        right : FermionicCommon or None
            The conjugate transpose of the Cholesky factor, or None.
        """
        absorb = Absorb.parse(absorb)
        x = self.phase_sync()
        if absorb == Absorb.sqVH:
            r = x._cholesky_abelian(shift=shift, upper=True)
            if not r.indices[0].dual:
                r.phase_flip(0, inplace=True)
            left, right = None, r
        elif absorb == Absorb.Usq:
            left = x._cholesky_abelian(shift=shift, upper=False)
            right = None
        elif absorb == Absorb.Usq_sqVH:
            left = x._cholesky_abelian(shift=shift, upper=False)
            right = left.dagger_compose_right()
        else:
            raise ValueError(
                "Invalid absorb value, must be one of -12 ('lsqrt' / 'Usq') 0 "
                "('both' / 'Usq_sqVH'), or 12 ( 'rsqrt' / 'sqVH')."
            )

        if drop_dummy_modes == "auto" or drop_dummy_modes:
            for factor in (left, right):
                if factor is not None:
                    factor._dummy_modes = ()
                    factor._label = None

        return left, None, right

    def solve(self, b: "FermionicCommon", **kwargs) -> "FermionicCommon":
        """Solve linear system Ax = b for x, where A is this fermionic array.

        Parameters
        ----------
        b : FermionicCommon
            The right hand side array.

        Returns
        -------
        x : FermionicCommon
            The solution array.
        """
        A = self.phase_sync()
        x = A._solve_abelian(b.phase_sync(), **kwargs)

        if x.indices[0].dual:
            # inner index is like |x><x| so introduce a phase flip
            x.phase_flip(0, inplace=True)

        return x
