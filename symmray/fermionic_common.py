"""Common methods for any fermionic arrays."""


class FermionicCommon:
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
        from .sparse.sparse_array import calc_fuse_group_info

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
        AbelianArray or scalar
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

    def to_dense(self):
        """Return dense representation of the fermionic array, with lazy phases
        multiplied in.
        """
        return self.phase_sync()._to_dense_abelian()

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
