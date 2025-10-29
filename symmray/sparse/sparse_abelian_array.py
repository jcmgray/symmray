"""Abelian symmetric arrays with block sparse backend."""

from ..abelian_common import AbelianCommon
from ..common import SymmrayCommon
from ..symmetries import get_symmetry
from ..utils import DEBUG
from .sparse_array_common import SparseArrayCommon
from .sparse_data_common import BlockCommon
from .sparse_vector import BlockVector


class AbelianArray(
    SparseArrayCommon,
    BlockCommon,
    AbelianCommon,
    SymmrayCommon,
):
    """A block sparse array with symmetry constraints.

    Parameters
    ----------
    indices : tuple[BlockIndex]
        The indices of the array.
    charge : hashable, optionals
        The total charge of the array, if not given it will be inferred from
        either the first sector or set to the identity charge, if no sectors
        are given.
    blocks : dict[tuple[hashable], array_like]
        A mapping of each 'sector' (tuple of charges) to the data array.
    symmetry : str or Symmetry, optional
        The symmetry of the array, if not using a specific symmetry class.
    """

    __slots__ = (
        "_blocks",
        "_charge",
        "_indices",
        "_label",
        "_symmetry",
    )
    fermionic = False
    static_symmetry = None

    def __init__(
        self,
        indices,
        charge=None,
        blocks=(),
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

        if DEBUG:
            self.check()

    def new_with(self, indices, charge, blocks):
        """Create a new block sparse abelian array of the same class as this
        one. Unlike `copy`, this does not copy over any existing data and drops
        for example `label`.
        """
        return self._new_with_abelian(
            indices=indices,
            charge=charge,
            blocks=blocks,
        )

    def copy(self):
        """Copy this abelian block sparse array."""
        return self._copy_abelian()

    def copy_with(self, indices=None, charge=None, blocks=None):
        """A copy of this block array with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self._copy_with_abelian(
            indices=indices,
            charge=charge,
            blocks=blocks,
        )
        if DEBUG:
            new.check()
        return new

    def modify(self, indices=None, charge=None, blocks=None):
        """Modify this block array in place with some attributes replaced. Note
        that checks are not performed on the new properties, this is intended
        for internal use.
        """
        self._modify_abelian(indices=indices, charge=charge, blocks=blocks)
        if DEBUG:
            self.check()
        return self

    def _binary_blockwise_op(self, other, fn, missing=None, inplace=False):
        return self._binary_blockwise_op_abelian(
            other, fn, missing=missing, inplace=inplace
        )

    def _map_blocks(self, fn_block=None, fn_sector=None, fn_filter=None):
        """Map the blocks and their keys (sectors) of the array inplace."""
        self._map_blocks_blockcommon(fn_block, fn_sector, fn_filter)

    def transpose(self, axes=None, inplace=False):
        """Transpose this block sparse abelian array.

        Parameters
        ----------
        axes : tuple[int, ...] | None, optional
            A permutation of the axes to transpose the array by. If None,
            the axes will be reversed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        return self._transpose_abelian(axes=axes, inplace=inplace)

    def conj(self, inplace=False) -> "AbelianArray":
        """Return the complex conjugate of this block array, including the
        indices."""
        return self._conj_abelian(inplace=inplace)

    def dagger(self, inplace=False) -> "AbelianArray":
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

    def _fuse_core(self, *axes_groups, mode="auto", inplace=False):
        return self._fuse_core_abelian(
            *axes_groups, mode=mode, inplace=inplace
        )

    def unfuse(self, axis, inplace=False) -> "AbelianArray":
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.

        Parameters
        ----------
        axis : int
            The axis to unfuse. It must have subindex information (`.subinfo`).
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
        """
        return self._unfuse_abelian(axis, inplace=inplace)

    def squeeze(self, axis=None, inplace=False):
        """Squeeze the block array, removing axes of size 1.

        Parameters
        ----------
        axis : int or tuple[int], optional
            The axis or axes to squeeze. If None, all axes of size 1 will be
            removed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArray
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
            If tracing to a scalar, whether to return an AbelainArray object
            with no indices, or simply scalar itself (the default).

        Returns
        -------
        AbelianArray or scalar
        """
        return self._einsum_abelian(eq, preserve_array=preserve_array)

    def tensordot(self, other, axes=2, mode="auto", preserve_array=False):
        """Tensordot between two block sparse abelian symmetric arrays.

        Parameters
        ----------
        a, b : SparseArrayCommon
            The arrays to be contracted.
        axes : int or tuple[int]
            The axes to contract. If an integer, the last ``axes`` axes of
            ``a`` will be contracted with the first ``axes`` axes of ``b``. If
            a tuple, the axes to contract in ``a`` and ``b`` respectively.
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
        """Compute the trace of the block array, assuming it is a square
        matrix.
        """
        return self._trace_abelian()

    def to_dense(self):
        return self._to_dense_abelian()

    def allclose(self, other, **allclose_opts):
        """Test whether this `AbelianArray` is close to another, that is,
        has all the same sectors, and the corresponding arrays are close.

        Parameters
        ----------
        other : AbelianArray
            The other array to compare to.
        allclose_opts
            Keyword arguments to pass to `allclose`.

        Returns
        -------
        bool
        """
        return self._allclose_abelian(other, **allclose_opts)

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
        return self._test_allclose_abelian(other, **allclose_opts)

    # --------------------------- linalg methods ---------------------------- #

    def qr(
        self, stabilized=False, **kwargs
    ) -> tuple["AbelianArray", "AbelianArray"]:
        """QR decomposition of an AbelianArray.

        Parameters
        ----------
        x : AbelianArray
            The block symmetric array to decompose.
        stabilized : bool, optional
            Whether to use a stabilized QR decomposition, that is, with
            positive diagonal elements in the R factor. Default is False.

        Returns
        -------
        q : AbelianArray
            The orthogonal matrix.
        r : AbelianArray
            The upper triangular matrix.
        """
        return self._qr_abelian(stabilized=stabilized, **kwargs)

    def svd(
        self, **kwargs
    ) -> tuple["AbelianArray", "BlockVector", "AbelianArray"]:
        """SVD decomposition of an AbelianArray.

        Parameters
        ----------
        x : AbelianArray
            The block symmetric array to decompose.

        Returns
        -------
        u : AbelianArray
            The left singular vectors.
        s : BlockVector
            The singular values as a vector.
        vh : AbelianArray
            The right singular vectors (hermitian conjugated).
        """
        return self._svd_abelian(**kwargs)

    def eigh(self, **kwargs) -> tuple["BlockVector", "AbelianArray"]:
        """Eigenvalue decomposition of this assumed Hermitian AbelianArray.

        Parameters
        ----------
        x : AbelianArray
            The block symmetric array to decompose.

        Returns
        -------
        w : BlockVector
            The eigenvalues as a vector.
        u : AbelianArray
            The array of eigenvectors.
        """
        return self._eigh_abelian(**kwargs)

    def eigh_truncated(
        self,
        cutoff=-1.0,
        cutoff_mode=4,
        max_bond=-1,
        absorb=0,
        renorm=0,
        positive=0,
        **kwargs,
    ) -> tuple["AbelianArray", "BlockVector", "AbelianArray"]:
        """Truncated hermitian eigen-decomposition of this assumed hermitian
        block sparse abelian array.

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

        Returns
        -------
        u : AbelianArray
            The abelian array of left eigenvectors.
        w : VectorCommon or None
            The vector of eigenvalues, or None if absorbed.
        uh : AbelianArray
            The abelian array of right eigenvectors.
        """
        return self._eigh_truncated_abelian(
            cutoff=cutoff,
            cutoff_mode=cutoff_mode,
            max_bond=max_bond,
            absorb=absorb,
            renorm=renorm,
            positive=positive,
            **kwargs,
        )

    def solve(self, b: "AbelianArray", **kwargs) -> "AbelianArray":
        """Solve the linear system `A @ x == b` for x, where A is this array.

        Parameters
        ----------
        b : AbelianArray
            The right-hand side array.

        Returns
        -------
        x : AbelianArray
            The solution array.
        """
        return self._solve_abelian(b, **kwargs)


# --------------------------------------------------------------------------- #


class Z2Array(AbelianArray):
    """A block array with Z2 symmetry."""

    static_symmetry = get_symmetry("Z2")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.core import SparseTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import Z2

        blocks = [
            SubTensor(array, q_labels=tuple(map(Z2, sector)))
            for sector, array in self.get_sector_block_pairs()
        ]

        data = SparseTensor(blocks)

        if flat:
            from pyblock3.algebra.flat import FlatSparseTensor

            data = FlatSparseTensor.from_sparse(data)

        data.shape = self.shape

        return data


class U1Array(AbelianArray):
    """A block array with U1 symmetry."""

    static_symmetry = get_symmetry("U1")

    def to_pyblock3(self, flat=False):
        from pyblock3.algebra.core import SparseTensor, SubTensor
        from pyblock3.algebra.fermion_symmetry import U1

        blocks = [
            SubTensor(array, q_labels=tuple(map(U1, sector)))
            for sector, array in self.get_sector_block_pairs()
        ]

        data = SparseTensor(blocks)

        if flat:
            from pyblock3.algebra.flat import FlatSparseTensor

            data = FlatSparseTensor.from_sparse(data)

        data.shape = self.shape

        return data

    def to_yastn(self, **config_opts):
        import yastn

        t = yastn.Tensor(
            config=yastn.make_config(sym="U1", **config_opts),
            s=tuple(-1 if ix.dual else 1 for ix in self.indices),
            n=self.charge,
        )
        for sector, array in self.get_sector_block_pairs():
            t.set_block(ts=sector, Ds=array.shape, val=array)

        return t


class Z2Z2Array(AbelianArray):
    """A block array with Z2 x Z2 symmetry."""

    static_symmetry = get_symmetry("Z2Z2")


class U1U1Array(AbelianArray):
    """A block array with U1 x U1 symmetry."""

    static_symmetry = get_symmetry("U1U1")
