"""Common methods for any fermionic arrays."""

import numbers

from .abelian_common import parse_tensordot_axes


class FermionicCommon:
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

        # tranposition introduces all necessary phases
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
            # preserve array for resolving oddposs
            preserve_array=True,
            **kwargs,
        )

        c._resolve_oddpos_combine(a, b)

        if (c.ndim == 0) and (not preserve_array):
            c.phase_sync(inplace=True)
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
        c._resolve_oddpos_combine(a, b)

        if c.ndim == 0 and (not preserve_array):
            c.phase_sync(inplace=True)
            return c.get_scalar_element()

        return c

    def to_dense(self):
        """Return dense representation of the fermionic array, with lazy phases
        multiplied in.
        """
        return self.phase_sync()._to_dense_abelian()

    def squeeze(self, axis=None, inplace=False) -> "FermionicCommon":
        """Squeeze the fermionic array, removing axes of size 1. If those axes
        correspond to odd parity charges, then they are converged into dummy
        `oddpos` modes effectively to the left of the array. The sorting label
        of the array is then required to have been set.

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

        if isinstance(axis, numbers.Integral):
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
        # beginning of the array, and also turns them into dummy oddpos modes
        new._resolve_oddpos_squeeze(axes_squeeze)

        # actually do the data squeeze
        return new._squeeze_abelian(axes_squeeze, inplace=True)

    def dagger_project_left(self) -> "FermionicCommon":
        """Take the dagger (conjugate transpose) of this fermionic array,
        assuming we are going to use to projector from the left on another
        operator.
        """
        new = self._dagger_abelian()
        if new.indices[-1].dual:
            new.phase_flip(-1, inplace=True)
        return new

    def dagger_project_right(self) -> "FermionicCommon":
        """Take the dagger (conjugate transpose) of this fermionic array,
        assuming we are going to use to projector from the right on another
        operator.
        """
        new = self._dagger_abelian()
        if not new.indices[0].dual:
            new.phase_flip(0, inplace=True)
        return new

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

    def qr(
        self, stabilized=False
    ) -> tuple["FermionicCommon", "FermionicCommon"]:
        """QR decomposition of a fermionic array.

        Parameters
        ----------
        stabilized : bool, optional
            Whether to use a stabilized QR decomposition, that is, with
            positive diagonal elements in the R factor. Default is False.

        Returns
        -------
        q : FermionicCommon
            The orthogonal matrix.
        r : FermionicCommon
            The upper triangular matrix.
        """
        x = self.phase_sync()
        q, r = x._qr_abelian(stabilized=stabilized)

        if r.indices[0].dual:
            # inner index is like |x><x| so introduce a phase flip
            r.phase_flip(0, inplace=True)

        return q, r

    def svd(self):
        """Singular Value Decomposition of a fermionic array.

        Returns
        -------
        u : FermionicCommon
            Array of left singular vectors.
        s : VectorCommon
            Singular values.
        vh : FermionicCommon
            Array of right singular vectors.
        """
        x = self.phase_sync()
        u, s, vh = x._svd_abelian()

        if vh.indices[0].dual:
            # inner index is like |x><x| so introduce a phase flip
            vh.phase_flip(0, inplace=True)

        return u, s, vh

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
