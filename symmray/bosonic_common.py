"""Common methods for any 'bosonic' (non-fermionic) arrays."""

from .linalg_common import Absorb


class BosonicCommon:
    """Mixin for bosonic (non-fermionic) arrays."""

    def to_pytree(self):
        """Convert this abelian array to a pytree purely of non-symmray
        containers and objects.
        """
        return self._to_pytree_abelian()

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        return self._binary_blockwise_op_abelian(
            other, fn, missing=missing, inplace=inplace
        )

    def _split(self, *args, **kwargs):
        return self._split_abelian(*args, **kwargs)

    def transpose(
        self,
        axes=None,
        inplace=False,
    ) -> "BosonicCommon":
        """Transpose this abelian array.

        Parameters
        ----------
        axes : tuple[int, ...] | None, optional
            A permutation of the axes to transpose the array by. If None,
            the axes will be reversed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        BosonicCommon
        """
        return self._transpose_abelian(axes=axes, inplace=inplace)

    def conj(self, inplace=False) -> "BosonicCommon":
        """Return the complex conjugate of this abelian array, including
        the indices and any subindex fusing information.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        BosonicCommon
        """
        return self._conj_abelian(inplace=inplace)

    def dagger(self, inplace=False) -> "BosonicCommon":
        """Return the adjoint of this abelian array, including the
        indices and any subindex fusing information.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        return self._dagger_abelian(inplace=inplace)

    dagger_project_left = dagger
    dagger_project_right = dagger
    dagger_compose_left = dagger
    dagger_compose_right = dagger

    def _fuse_core(
        self,
        *axes_groups,
        mode="auto",
        inplace=False,
    ) -> "BosonicCommon":
        """The core implementation of the fuse operation, which fuses
        multiple axes into a single group, and returns a new array with
        the new sectors and blocks. The new axes are inserted at the minimum
        axis of any of the groups.
        """
        return self._fuse_core_abelian(
            *axes_groups, mode=mode, inplace=inplace
        )

    def unfuse(self, axis, inplace=False) -> "BosonicCommon":
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.

        Parameters
        ----------
        axis : int
            The axis to unfuse. It must have subindex information
            (``.indices[axis].subinfo``), typically from a previous fusing
            operation.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        BosonicCommon
        """
        return self._unfuse_abelian(axis, inplace=inplace)

    def squeeze(self, axis=None, inplace=False) -> "BosonicCommon":
        """Squeeze the abelian array, removing axes of size 1.

        Parameters
        ----------
        axis : int or sequence of int, optional
            The axis or axes to squeeze. If None, all axes of size 1 will be
            removed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        BosonicCommon
        """
        return self._squeeze_abelian(axis=axis, inplace=inplace)

    def einsum(self, eq, preserve_array=False):
        """Einsum for abelian arrays, currently only single term.

        Parameters
        ----------
        eq : str
            The einsum equation, e.g. "abcb->ca". The output indices must be
            specified and only trace and permutations are allowed.
        preserve_array : bool, optional
            If tracing to a scalar, whether to return an abelian array object
            with no indices, or simply the scalar itself (the default).

        Returns
        -------
        BosonicCommon or scalar
        """
        return self._einsum_abelian(eq, preserve_array=preserve_array)

    def tensordot(self, other, axes=2, mode="auto", preserve_array=False):
        """Tensordot between two abelian symmetric arrays.

        Parameters
        ----------
        other : BosonicCommon
            The other array to contract with.
        axes : int or tuple[int]
            The axes to contract. If an integer, the last ``axes`` axes of
            ``self`` will be contracted with the first ``axes`` axes of
            ``other``. If a tuple, the axes to contract in ``self`` and
            ``other`` respectively.
        mode : {"auto", "fused", "blockwise"}
            The mode to use for the contraction. If "auto", it will choose
            between "fused" and "blockwise" based on the number of axes to
            contract.
        preserve_array : bool, optional
            Whether to return a scalar if the result is a scalar.
        """
        return self._tensordot_abelian(
            other,
            axes=axes,
            mode=mode,
            preserve_array=preserve_array,
        )

    def __matmul__(self, other, preserve_array=False):
        return self._matmul_abelian(other, preserve_array=preserve_array)

    def trace(self):
        """Compute the trace of the abelian array, assuming it is a square
        matrix.
        """
        return self._trace_abelian()

    def to_dense(self):
        """Convert this abelian array to a dense array, by combining all the
        blocks into a single large array, filling in zeros where necessary.

        Returns
        -------
        array_like
            A dense array with the same shape as this abelian array.
        """
        return self._to_dense_abelian()

    def allclose(self, other, **allclose_opts):
        """Test whether this abelian array is close to another, that is,
        has all the same sectors, and the corresponding arrays are close.

        Parameters
        ----------
        other : BosonicCommon
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to ``allclose``.

        Returns
        -------
        bool
        """
        return self._allclose_abelian(other, **allclose_opts)

    def test_allclose(self, other, **allclose_opts):
        """Assert that this abelian array is close to another, that is, has
        all the same sectors, and the corresponding arrays are close. Unlike
        ``allclose``, this raises an AssertionError with details if not.

        Parameters
        ----------
        other : BosonicCommon
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to ``allclose``.

        Raises
        ------
        AssertionError
            If the arrays are not close.
        """
        return self._test_allclose_abelian(other, **allclose_opts)

    def eigh(self, **kwargs):
        """Eigenvalue decomposition of this assumed Hermitian abelian array.

        Returns
        -------
        w : vector
            The eigenvalues as a vector.
        u : BosonicCommon
            The array of eigenvectors.
        """
        return self._eigh_abelian(**kwargs)

    def cholesky(self, *, upper=False) -> "BosonicCommon":
        """Cholesky decomposition of this assumed positive-definite array.

        Parameters
        ----------
        upper : bool, optional
            Whether to return the upper triangular Cholesky factor.
            Default is False, returning the lower triangular factor.

        Returns
        -------
        l_or_r : BosonicCommon
            The Cholesky factor. Lower triangular if ``upper=False``,
            upper triangular if ``upper=True``.
        """
        return self._cholesky_abelian(upper=upper, shift=0)

    def cholesky_regularized(self, absorb=0, shift=True):
        """Cholesky decomposition with optional diagonal regularization,
        returning results in an SVD-like ``(left, None, right)`` format
        for compatibility with tensor network split drivers.

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

        Returns
        -------
        left : BosonicCommon or None
            The lower Cholesky factor, or None.
        s : None
            Always None (no singular values).
        right : BosonicCommon or None
            The conjugate transpose of the Cholesky factor, or None.
        """
        absorb = Absorb.parse(absorb)
        if absorb == Absorb.sqVH:
            r = self._cholesky_abelian(shift=shift, upper=True)
            return None, None, r

        l = self._cholesky_abelian(shift=shift, upper=False)

        if absorb == Absorb.Usq:
            return l, None, None

        if absorb == Absorb.Usq_sqVH:
            return l, None, l.H

        raise ValueError(f"Invalid absorb option: {absorb}")
