"""Methods that apply to abelian arrays with flat backend, both
fermionic and bosonic.
"""

import functools
import math
import operator
from itertools import repeat

import autoray as ar
import cotengra as ctg

from ..sparse.sparse_abelian_array import AbelianArray
from ..sparse.sparse_array import calc_fuse_group_info, parse_tensordot_axes
from ..utils import DEBUG, get_array_cls
from .flat_base import FlatVector
from .flat_index import FlatIndex, FlatSubIndexInfo

try:
    from einops import rearrange as _einops_rearrange
    from einops import repeat as _einops_repeat
    from einops.array_api import rearrange as _einops_rearrange_api
    from einops.array_api import repeat as _einops_repeat_api

    # want to support both standard and array_api versions

    def einops_rearrange(tensor, *args, **kwargs):
        if hasattr(tensor, "__array_namespace__"):
            return _einops_rearrange_api(tensor, *args, **kwargs)
        return _einops_rearrange(tensor, *args, **kwargs)

    def einops_repeat(tensor, *args, **kwargs):
        if hasattr(tensor, "__array_namespace__"):
            return _einops_repeat_api(tensor, *args, **kwargs)
        return _einops_repeat(tensor, *args, **kwargs)

except ImportError:

    def missinglib(*args, name, **kwargs):
        raise ImportError(f"'{name}' required for this function.")

    einops_rearrange = functools.partial(missinglib, name="einops.rearrange")
    einops_repeat = functools.partial(missinglib, name="einops.repeat")


def lexsort_sectors(sectors, stable=True):
    """Given a sequence of columns of positive integers, or equivalently a
    matrix of shape (num_sectors, num_charges), find the indices that
    sort them lexicographically, such that the first column is the most
    significant, then the second, and so forth.

    Parameters
    ----------
    sectors : array_like or sequence[array_like]
        The sectors to sort, each column should be a 1D array of positive
        integer charges. Either supplied as a 2D array, or a sequence of 1D
        arrays (columns of charges), in which case they will be stacked along a
        second axis.
    stable : bool, optional
        Whether to use a stable sort. Default is True, which uses the
        `argsort` function with the `stable` parameter set to True. If False,
        it will use the default sorting behavior which may not be stable.

    Returns
    -------
    array_like
        The indices that would sort the stack of sectors lexicographically.

    Examples
    --------

        >>> sectors = np.array([[4, 1, 0], [3, 2, 1], [3, 1, 0], [2, 0, 1]])
        >>> k = lexsort_sectors(sectors)
        >>> sectors[k]
        array([[2, 0, 1],
               [3, 1, 0],
               [3, 2, 1],
               [4, 1, 0]])
    """
    # XXX: this is a bottleneck

    if ar.is_array(sectors):
        ncol = sectors.shape[1]
        cols = tuple(sectors[:, c] for c in range(ncol))
    else:
        cols = sectors

    xp = ar.get_namespace(cols[0])

    if len(cols) == 1:
        return xp.argsort(cols[0], stable=stable)

    strides = [1]

    for col in cols[:0:-1]:
        strides.insert(0, (xp.max(col) + 1) * strides[0])

    ranks = functools.reduce(
        operator.add, (w * col for w, col in zip(strides, cols))
    )

    return xp.argsort(ranks, stable=stable)

    # # alternative implementation:

    # if not ar.is_array(sectors):
    #     sectors = ar.do("stack", sectors, axis=1)

    # backend = ar.infer_backend(sectors)

    # n = ar.do("shape", sectors, like=backend)[1]
    # limits = ar.do("max", sectors, axis=0, like=backend) + 1

    # # ws = ar.do("ones", (1, n), dtype=sectors.dtype, like=backend)

    # ws = [ar.do("array", 1, like=sectors)] * n
    # for ax in range(n - 2, -1, -1):
    #     # reverse cumulative product to get 'strides'
    #     ws[ax] = ws[ax + 1] * limits[ax]
    # ws = ar.do("stack", tuple(ws), like=sectors)
    # ws = ar.do("reshape", ws, (1, n))

    # ranks = ar.do("sum", ws * sectors, axis=1, like=backend)
    # kord2 = ar.do("argsort", ranks, stable=stable, like=backend)

    # assert ar.do("all", kord1 == kord2)
    # return kord1


@ar.compose
def select_slice(x, i):
    """Select the i'th slice of the input array."""
    return x[i]


@select_slice.register("torch")
def select_slice_torch(x, i):
    """`torch` doesn't support vmapping the above operation."""
    import torch

    i = torch.unsqueeze(i, 0)
    xi = torch.index_select(x, 0, i)
    return torch.squeeze(xi, 0)


def zn_combine(order, sectors, duals=None, like=None):
    """Implement vectorized addition modulo group order, with signature.

    Parameters
    ----------
    order : int
        The order of the symmetry group, i.e., the number of distinct charges
        in each axis. E.g. 2 corresponds to Z2 symmetry.
    sectors : array_like
        The stack of sectors, with shape (num_blocks, num_charges). Each
        row represents a subsector.
    duals : sequence[bool] | None, optional
        The dualness of each index, i.e., whether the charge contributes
        positively or negatively. If not given, it will be assumed that all
        charges are positive.
    like : str or array_like, optional
        The array-like object to use as a reference for the output type and
        backend. If not given, will be inferred..
    """
    if order == 2:
        # self inverse, no need to check duals
        return ar.do("sum", sectors, axis=-1, like=like) % 2

    if (duals is not None) and any(duals):
        # turn duals into phases
        signs = [(-1) ** dual for dual in duals]
        # broadcasted multiply
        signs = ar.do("array", signs, like=like)
        signs = ar.do("reshape", signs, (1, -1))
        signed_sectors = sectors * signs
    else:
        signed_sectors = sectors

    return ar.do("sum", signed_sectors, axis=-1, like=like) % order


def _calc_fused_sectors_subkeys_slice(
    num_groups,
    axes_groups,
    new_sectors,
    old_sectors,
    pos,
    ndim,
    new_ndim,
    order,
    group_singlets,
    backend,
):
    """Calculate new sectors and group subkeys, by slicing the existing
    sectors.
    """
    # now we compute subcharge information for each group
    # first we reshape the old sectors given the sort above:
    #     (*fused_charges, *unfused_charges, --locked dim, *subcharges)
    # whichever is the last charge of the first two groups is locked,
    # and so taken away from the reshaping.
    keys_reshaper = []
    axes_seen = 0
    g_locked_map = {}

    for g in range(num_groups):
        axes_seen += len(axes_groups[g])
        if axes_seen == ndim:
            # if all axes are being fused, the last one is locked
            keys_reshaper.append(1)
            if num_groups > 1:
                # all axes are fused, multiple new charges, last of which
                # is locked to prev, so we can't take a slice across it
                # directly, nor will it be sorted
                g_locked_map[g] = g - 1
        else:
            # one dimension for each fused charge
            keys_reshaper.append(order)
    # one dimension for all the unfused axes
    keys_reshaper.append(-1)
    # then one dimension for each *extra* subcharge in each group
    nsubaxes = 0
    for group in axes_groups:
        # each group has one locked charge within it
        nsub = len(group) - 1
        keys_reshaper.extend([order] * nsub)
        nsubaxes += nsub

    # reshape! including finally one dimension for the row of charges
    new_sectors = ar.do("reshape", new_sectors, (*keys_reshaper, new_ndim))
    old_sectors = ar.do("reshape", old_sectors, (*keys_reshaper, ndim))

    # drop sub charge axes from new_sectors
    new_sectors = new_sectors[
        (
            *repeat(slice(None), num_groups),  # fused charges
            slice(None),  # unfused axes
            *repeat(0, nsubaxes),  # sub charges, take first (arbitrary)
            slice(None),  # row of charges
        )
    ]

    # then we take slices across these to strore subcharge information
    # for unfusing
    subkeys = []
    for g in range(num_groups):
        if g in group_singlets:
            # no need to record subcharges
            subkeys.append(None)
            continue

        subkey_selector = []

        # is this group is locked, need to slice over previous group
        g_lock = g_locked_map.get(g, g)

        # fix other grouped charges to any value, take slice across g
        for go in range(num_groups):
            if go == g_lock:
                subkey_selector.append(slice(None))
            else:
                subkey_selector.append(0)

        # fix stack of unfused axes to any value
        subkey_selector.append(0)

        # then we fix the subcharges for other groups to any value
        # and slice across fused subcharges
        for go in range(num_groups):
            m = len(axes_groups[go]) - 1
            if go == g:
                subkey_selector.extend([slice(None)] * m)
            else:
                subkey_selector.extend([0] * m)

        # finally we take the slice across the legs
        # i.e. the row of charges making up the sector
        subkey_selector.append(slice(None))

        # take the slice!
        subkey_selector = tuple(subkey_selector)
        subkey = old_sectors[subkey_selector]
        # flatten into (overall_charge, subcharges, row)
        subkey = ar.do("reshape", subkey, (keys_reshaper[g_lock], -1, ndim))
        # just get axes relevant to this group
        subkey = subkey[:, :, axes_groups[g]]

        if g_lock != g:
            # this group axis is locked to the previous group,
            # so we need to do an additional sort the overall charge axis
            gcharges = new_sectors[
                (
                    *repeat(0, g - 1),  # other groups
                    slice(None),  # group we are locked to
                    0,  # locked g axis
                    0,  # unfused axes
                    pos + g,
                )
            ]
            kord = ar.do("argsort", gcharges, like=backend)
            subkey = subkey[kord]

        subkeys.append(subkey)

    # reflatten new keys into stack
    new_sectors = ar.do("reshape", new_sectors, (-1, new_ndim), like=backend)

    return new_sectors, subkeys


def _calc_fused_sectors_subkeys_create(
    unmerged_batch_sizes,
    num_groups,
    axes_groups,
    new_sectors,
    old_sectors,
    group_duals,
    ndim,
    order,
    duals,
    like,
):
    """Calculate new sectors and group subkeys, by explicit generation."""
    # then we update the sectors, these have been sorted already by
    # grouped charge, each group being of equal size / stride:
    stride = math.prod(unmerged_batch_sizes.values())
    new_sectors = new_sectors[::stride]

    if num_groups == 1 and len(axes_groups[0]) == ndim:
        # full fuse, only one overall charge, subkeys are all current keys
        # but we do need to take into account possible permutation
        subkeys0 = ar.do(
            "reshape", old_sectors[..., axes_groups[0]], (1, -1, ndim)
        )
        subkeys = [subkeys0]
    else:
        subkeys = [
            build_cyclic_keys_by_charge(
                ndim=len(gaxes),
                order=order,
                duals=[duals[ax] != group_duals[g] for ax in gaxes],
                like=like,
            )
            for g, gaxes in enumerate(axes_groups)
        ]

    return new_sectors, subkeys


def _calc_fuse_rearrange_pattern(
    num_groups,
    axes_groups,
    axes_before,
    axes_after,
    order,
    ndim,
):
    # now we create the unmerge/merge pattern for einops:
    # heres a full example for 2 groups and 6 axes:
    # axes_groups = ((5, 2), (4, 1))
    # '(B B0 B1) p0 p1 p2 p3 p4 p5 -> B p0 (B0 p5 p2) (B1 p4 p1) p3'

    # LHS, first we 'unfuse' the block index
    # with one new axis for each group, `ax0 -> (B B0 B1 B2 ...)`
    pattern = ["(B"]
    for g in range(num_groups):
        pattern.append(f" B{g}")
    pattern.append(")")

    # then we label each of the input axes `... p0 p1 p2 ...`
    for ax in range(ndim):
        pattern.append(f" p{ax}")

    # RHS, start with the new block index
    pattern.append(" -> B")

    # then add the unfused output axes before the groups
    for ax in axes_before:
        pattern.append(f" p{ax}")

    # then the groups, each looks like `(B0 p5 p2 ...)`, one dimension
    # coming from the batch index, and the rest real axis fusions
    unmerged_batch_sizes = {}
    for g, gaxes in enumerate(axes_groups):
        pattern.append(f" (B{g}")
        for ax in gaxes:
            pattern.append(f" p{ax}")
            bax = f"B{g}"
            # keep track of the unmerged
            if bax in unmerged_batch_sizes:
                unmerged_batch_sizes[bax] *= order
            else:
                unmerged_batch_sizes[bax] = 1
        pattern.append(")")

    # then add the unfused output axes after the groups
    for ax in axes_after:
        pattern.append(f" p{ax}")

    pattern = "".join(pattern)
    return pattern, unmerged_batch_sizes


class FlatArrayCommon:
    def _init_flatarraycommon(
        self,
        sectors,
        blocks,
        indices,
        symmetry=None,
    ):
        self._symmetry = self.get_class_symmetry(symmetry)

        self._blocks = (
            blocks if hasattr(blocks, "shape") else ar.do("array", blocks)
        )
        # infer the backend to reuse for efficiency
        self.backend = ar.infer_backend(self._blocks)

        self._sectors = (
            sectors
            if hasattr(sectors, "shape")
            else ar.do("array", sectors, like=self._blocks)
        )
        self._indices = tuple(
            # allow sequence of duals to be supplied directly
            x
            if isinstance(x, FlatIndex)
            else FlatIndex(
                num_charges=1 if self.ndim == 1 else self.order,
                charge_size=d,
                dual=x,
            )
            for x, d in zip(indices, self.shape_block)
        )

    def _check_abelian(self):
        assert len(self._sectors) == len(self._blocks)
        assert self.ndim == len(self._indices)
        assert self.ndim == len(self._sectors[0])
        # check blocks all have the same overall charge
        sector_charges = ar.do(
            "unique", zn_combine(self.order, self._sectors, self.duals)
        )
        assert ar.do("size", sector_charges) == 1
        for ix, ds in zip(self._indices, self.shape_block):
            ix.check()
            assert ds == ix.charge_size

    def _copy_flatarraycommon(self, deep=False) -> "FlatArrayCommon":
        """Create a copy of the array."""
        if deep:
            sectors = ar.do("copy", self._sectors, like=self.backend)
            blocks = ar.do("copy", self._blocks, like=self.backend)
        else:
            sectors = self._sectors
            blocks = self._blocks
        return self.__class__(
            sectors, blocks, self._indices, symmetry=self._symmetry
        )

    def _copy_with_flatarraycommon(
        self,
        sectors=None,
        blocks=None,
        indices=None,
    ) -> "FlatArrayCommon":
        """A copy of this flat array with some attributes replaced. Note that
        checks are not performed on the new properties, this is intended for
        internal use.
        """
        new = self.__new__(self.__class__)
        new._sectors = self._sectors if sectors is None else sectors
        new._indices = self._indices if indices is None else indices
        new._blocks = self._blocks if blocks is None else blocks
        new._symmetry = self._symmetry
        new.backend = self.backend
        return new

    def _modify_flatarraycommon(
        self,
        sectors=None,
        blocks=None,
        indices=None,
    ) -> "FlatArrayCommon":
        """Modify this flat array in place with some attributes replaced. Note
        that checks are not performed on the new properties, this is intended
        for internal use.
        """
        if sectors is not None:
            self._sectors = sectors
        if blocks is not None:
            self._blocks = blocks
        if indices is not None:
            self._indices = indices

        return self

    @property
    def order(self) -> int:
        """Get the order of the symmetry group."""
        return self._symmetry.N

    @property
    def charge(self):
        """Compute the overall charge of the array."""
        return zn_combine(
            self.order,
            self._sectors[[0], :],
            self.duals,
            like=self.backend,
        )[0]

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the effective shape of the array."""
        return tuple(ix.size_total for ix in self._indices)

    @property
    def size(self) -> int:
        """Get the total size of the array, i.e., the product of all dimensions
        in the effective shape."""
        return functools.reduce(operator.mul, self.shape, 1)

    @classmethod
    def from_scalar(cls, x, symmetry=None) -> "FlatArrayCommon":
        """Create a flat abelian array from a scalar."""
        sectors = [[]]
        indices = ()
        blocks = ar.do("reshape", x, (1,))
        return cls(sectors, blocks, indices, symmetry=symmetry)

    def _to_blocksparse_flatarraycommon(self) -> AbelianArray:
        """Create a blocksparse abelian array from this flat abelian array."""
        cls = get_array_cls(
            self.symmetry,
            self.fermionic,
            False,
        )

        blocks = {}
        for i in range(self.num_blocks):
            sector = tuple(map(int, self._sectors[i]))
            block = self._blocks[i]
            blocks[sector] = block

        return cls.from_blocks(
            blocks,
            duals=self.duals,
            symmetry=self.symmetry,
        )

    def get_sorting_indices(
        self,
        axes=None,
        all_axes=None,
    ):
        """Get the indices that would lexicgraphically sort the stack of
        sectors according to the values of charges in the specified axes,
        optionally filling in the rest of the axes with the remaining axes in
        the order they appear.

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
        """
        if axes is None:
            axes = ()
            if all_axes is None:
                all_axes = True
        elif all_axes is None:
            all_axes = False

        # ensure tuple and handle negative axes
        if not isinstance(axes, (tuple, list)):
            if axes < 0:
                axes = axes + self.ndim
            axes = (axes,)
        else:
            axes = tuple(ax if ax >= 0 else ax + self.ndim for ax in axes)

        if all_axes:
            # include all non-specified axes as tie-breakers
            axes = (*axes, *(ax for ax in range(self.ndim) if ax not in axes))

        cols = self._sectors[:, axes]
        return lexsort_sectors(cols)

    def _transpose_flatarraycommon(
        self,
        axes=None,
        inplace=False,
    ) -> "FlatArrayCommon":
        """Transpose this flat abelian array.

        Parameters
        ----------
        axes : tuple[int, ...] | None, optional
            A permutation of the axes to transpose the array by. If None,
            the axes will be reversed.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FlatArrayCommon
        """
        if axes is None:
            # reverse the axes
            axes = tuple(range(self.ndim - 1, -1, -1))

        axes = tuple(map(int, axes))

        # transpose block as usual, but with broadcasted block axis
        new_blocks = ar.do(
            "transpose",
            self._blocks,
            (0, *(ax + 1 for ax in axes)),
            like=self.backend,
        )

        new_sectors = self._sectors[:, axes]
        new_indices = tuple(self._indices[ax] for ax in axes)

        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def _conj_flatarraycommon(self, inplace=False) -> "FlatArrayCommon":
        """Return the complex conjugate of this block array, including the
        indices and any subindex fusing information.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FlatArrayCommon
        """
        new_sectors = self._sectors
        new_blocks = ar.do("conj", self._blocks, like=self.backend)
        new_indices = tuple(ix.conj() for ix in self._indices)
        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def expand_dims(
        self, axis, c=None, dual=None, inplace=False
    ) -> "FlatArrayCommon":
        """Expand the shape of an abelian array.

        Parameters
        ----------
        axis : int
            The position along which to expand.
        c : hashable, optional
            The charge to insert at the new axis. If not given, a zero charge
            will be inserted.
        dual : bool, optional
            The dual-ness of the new index. If not given, it will be inherited
            from the axis before or after, if any. If there is no axis before
            or after, it will default to `False`.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        FlatArrayCommon
        """
        if axis < 0:
            axis += self.ndim + 1

        if dual is None:
            # we inherit the dual-ness from the axis before or after
            # to make fusing and unfusing singleton axes commutative
            if axis > 0:
                # inherit from left
                dual = self._indices[axis - 1].dual
            elif axis < self.ndim:
                # inherit from right
                dual = self._indices[axis].dual
            else:
                # no axes to inherit from
                dual = False

        if c is None:
            # insert a zero charge
            c = 0

        # insert the self charge as a column
        keys_new_col = ar.do(
            "tile", c, (self.num_blocks, 1), like=self.backend
        )
        new_sectors = ar.do(
            "concatenate",
            (self._sectors[:, :axis], keys_new_col, self._sectors[:, axis:]),
            axis=1,
            like=self.backend,
        )

        # expand the actual blocks (accounting for extra flat ax at start)
        selector = (
            (slice(None),) * (axis + 1)
            + (None,)
            + (slice(None),) * (self.ndim - axis - 1)
        )
        new_blocks = self._blocks[selector]

        # expand the index information
        new_indices = (
            *self._indices[:axis],
            FlatIndex(
                num_charges=1,
                charge_size=1,
                dual=dual,
            ),
            *self._indices[axis:],
        )

        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def _fuse_core_abelian(
        self,
        *axes_groups,
        inplace=False,
    ) -> "FlatArrayCommon":
        """The core implementation of the fuse operation, which fuses
        multiple axes into a single group, and returns a new array with
        the new sectors and blocks. The new axes are inserted at the minimum
        axis of any of the groups.
        """
        (
            num_groups,
            group_singlets,
            new_ndim,
            _,  # perm,
            pos,
            axes_before,
            axes_after,
            _,  # ax2group,
            group_duals,
            _,  # new_axes,
        ) = calc_fuse_group_info(axes_groups, self.duals)

        # create the new sectors, starting with the unfused axes before
        new_sectors = [self._sectors[:, ax] for ax in axes_before]
        for axs, dg in zip(axes_groups, group_duals):
            # charges with opposite sign to overall group need to be flipped
            eff_duals = [self.duals[ax] != dg for ax in axs]
            new_sectors.append(
                zn_combine(self.order, self._sectors[:, axs], eff_duals)
            )
        # then we add the unfused axes after
        new_sectors.extend(self._sectors[:, ax] for ax in axes_after)
        # combine into single array
        new_sectors = ar.do("stack", tuple(new_sectors), axis=1)

        # then we find the correct order to sort the new keys
        sortingcols = (
            # first we sort by fused charge
            *(new_sectors[:, pos + g] for g in range(num_groups)),
            # then we sort by the unfused axes
            *(self._sectors[:, ax] for ax in axes_before),
            *(self._sectors[:, ax] for ax in axes_after),
            # and finally by the fused charges within each group
            *(self._sectors[:, ax] for group in axes_groups for ax in group),
        )
        kord = lexsort_sectors(sortingcols)
        new_blocks = self._blocks[kord]
        new_sectors = new_sectors[kord]
        # XXX: only optionally store the fusing information
        old_sectors = self._sectors[kord]

        # get the einops rearrangement pattern for the new blocks
        pattern, unmerged_batch_sizes = _calc_fuse_rearrange_pattern(
            num_groups,
            axes_groups,
            axes_before,
            axes_after,
            self.order,
            self.ndim,
        )
        # perform the rearrangement!
        new_blocks = einops_rearrange(
            new_blocks, pattern, **unmerged_batch_sizes
        )

        # now we calculate the new sectors and subkeys, either by slicing
        # the existing sectors, or by creating them from scratch
        new_sectors, subkeys = _calc_fused_sectors_subkeys_slice(
            num_groups,
            axes_groups,
            new_sectors,
            old_sectors,
            pos,
            self.ndim,
            new_ndim,
            self.order,
            group_singlets,
            self.backend,
        )
        # new_sectors, subkeys = _calc_fused_sectors_subkeys_create(
        #     unmerged_batch_sizes,
        #     num_groups,
        #     axes_groups,
        #     new_sectors,
        #     old_sectors,
        #     group_duals,
        #     self.ndim,
        #     self.order,
        #     self.duals,
        #     like=self._sectors,
        # )

        # finally we construct info, including for unfusing
        new_indices = []

        for ax in axes_before:
            new_indices.append(self.indices[ax])

        old_indices = self.indices

        for g, gaxes in enumerate(axes_groups):
            if g in group_singlets:
                # no new fuse -> just propagate current index
                new_indices.append(old_indices[gaxes[0]])
            else:
                # create a new subinfo for this group
                subindices = [old_indices[ax] for ax in gaxes]
                subinfo = FlatSubIndexInfo(subindices, subkeys[g])

                if num_groups > 1 or axes_before or axes_after:
                    num_charges = self.order
                else:
                    # output is 1D
                    num_charges = 1

                charge_size = (
                    # this is contribution from fusing the actual blocks
                    math.prod(ix.charge_size for ix in subindices)
                    # & this is contribution from grouping sectors
                    # e.g. 01 and 10 both fuse to 1
                    * self.order ** (len(gaxes) - 1)
                )

                ix = FlatIndex(
                    num_charges=num_charges,
                    charge_size=charge_size,
                    dual=group_duals[g],
                    subinfo=subinfo,
                )

                new_indices.append(ix)

        for ax in axes_after:
            new_indices.append(self.indices[ax])

        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def _unfuse_abelian(self, axis, inplace=False):
        new = self if inplace else self.copy()

        fi = new.indices[axis].subinfo
        if fi is None:
            raise ValueError(f"Axis {axis} is not fused in this array.")

        axs_rem = tuple(range(axis)) + tuple(range(axis + 1, self.ndim))
        new.sort_stack((axis, *axs_rem), inplace=True)

        # keys coming from remaining axes
        ka = einops_repeat(
            new.sectors, "(Bf B) s -> (Bf B x) s", Bf=fi.ncharge, x=fi.nsectors
        )[:, axs_rem]

        # keys coming from unfused axis
        kb = fi.subkeys
        kb = einops_repeat(
            kb, "B Bu s -> (B x Bu) s", x=new.num_blocks // fi.ncharge
        )

        # concatenate into the full new keys!
        new_sectors = ar.do(
            "concatenate", (ka[:, :axis], kb, ka[:, axis:]), axis=-1
        )

        # now we need to unfuse the actual blocks an example pattern:
        #     B p0 ( Bu u0 u1 u2 ) p2 p3 -> (B Bu) p0 u0 u1 u2 p2 p3
        # i.e. we unfuse the current axis, and shift its internal sector index
        # (`Bu`) into the total sector index
        pattern = ["B "]
        rhs = ["(B Bu) "]
        sizes = {}
        for i in range(axis):
            pattern.append(f"p{i} ")
            rhs.append(f"p{i} ")
        pattern.append("( Bu ")
        for g, sz in enumerate(ix.charge_size for ix in fi.indices):
            pattern.append(f"u{g} ")
            rhs.append(f"u{g} ")
            sizes[f"u{g}"] = sz
        pattern.append(") ")
        for i in range(axis + 1, new.ndim):
            pattern.append(f"p{i} ")
            rhs.append(f"p{i} ")
        pattern.append("-> ")
        pattern.extend(rhs)
        pattern = "".join(pattern)

        # perform the unfuse!
        new_blocks = einops_rearrange(new.blocks, pattern, **sizes)

        # unpack sub indices
        new_indices = (
            *new.indices[:axis],
            *fi.indices,
            *new.indices[axis + 1 :],
        )

        return new.modify(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
        )

    def select_charge(self, axis, charge, inplace=False):
        """Drop all but the specified charge along the specified axis. Note the
        axis is not removed, it is simply restricted to a single charge.

        Parameters
        ----------
        axis : int
            The axis along which to select the charge.
        charge : int
            The charge to select along the specified axis.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        FlatArrayCommon
        """
        if axis < 0:
            axis += self.ndim

        new = self.sort_stack(axis, inplace=inplace)

        shape_sectors = ar.do("shape", new.sectors, like=self.backend)
        shape_blocks = ar.do("shape", new.blocks, like=self.backend)

        dc = self.order
        dB = shape_sectors[0]

        new_sectors = ar.do(
            "reshape",
            new.sectors,
            (dc, dB // dc, *shape_sectors[1:]),
            like=self.backend,
        )
        new_sectors = select_slice(new_sectors, charge)

        new_blocks = ar.do(
            "reshape",
            new.blocks,
            (dc, dB // dc, *shape_blocks[1:]),
            like=self.backend,
        )
        new_blocks = select_slice(new_blocks, charge)

        if self.ndim == 2:
            # axes are locked to each other -> select both
            if axis == 0:
                other_charge = new_sectors[0, 1]
                new_indices = (
                    self.indices[0].select_charge(charge),
                    self.indices[1].select_charge(other_charge),
                )
            else:  # axis == 1
                other_charge = new_sectors[0, 0]
                new_indices = (
                    self.indices[0].select_charge(other_charge),
                    self.indices[1].select_charge(charge),
                )
        else:
            new_indices = (
                *self.indices[:axis],
                self.indices[axis].select_charge(charge),
                *self.indices[axis + 1 :],
            )

        return new.modify(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
        )

    def squeeze(self, axis, inplace=False):
        """Assuming `axis` has total size 1, remove it from this array."""
        axs_rem = tuple(i for i in range(self.ndim) if i != axis)

        new_sectors = self.sectors[:, axs_rem]
        new_indices = tuple(self._indices[i] for i in axs_rem)
        block_selector = tuple(
            slice(None) if i != axis + 1 else 0 for i in range(self.ndim + 1)
        )
        new_blocks = self.blocks[block_selector]

        return self._modify_or_copy(
            sectors=new_sectors,
            indices=new_indices,
            blocks=new_blocks,
            inplace=inplace,
        )

    def isel(self, axis, idx, inplace=False):
        """Select a single index along the specified axis."""
        if axis < 0:
            axis += self.ndim
        new = self.select_charge(axis, idx, inplace=inplace)
        return new.squeeze(axis, inplace=True)

    def __getitem__(self, item):
        axis = None
        idx = None

        if not isinstance(item, tuple):
            raise TypeError(
                f"Expected a tuple for indexing, got {type(item)}: {item}"
            )

        for i, s in enumerate(item):
            if isinstance(s, slice):
                if not s.start is s.stop is s.step is None:
                    raise NotImplementedError("Can only slice whole axes.")
            else:
                if axis is not None:
                    raise ValueError(
                        "Can only index one axis at a time, "
                        f"got {item} with multiple indices."
                    )
                axis = i
                idx = s

        return self.isel(axis, idx)

    def align_axes(
        self: "FlatArrayCommon",
        other: "FlatArrayCommon",
        axes: tuple[tuple[int, ...], tuple[int, ...]],
        inplace=False,
    ) -> tuple["FlatArrayCommon", "FlatArrayCommon"]:
        """Align the axes of two arrays for contraction."""
        a, b = (self, other) if inplace else (self.copy(), other.copy())

        ndim_a = a.ndim
        ndim_b = b.ndim
        ncon = len(axes[0])

        if ndim_a > ncon and ndim_b > ncon:
            # ~ matmat: just need to sort sectors along common axes
            a.sort_stack(axes[0], inplace=True)
            b.sort_stack(axes[1], inplace=True)

        elif ndim_a > ncon and ndim_b == ncon:
            # ~matvec: b has locked axis and only one charge
            (axis,) = axes[0]
            a.select_charge(axis, b._sectors[0, 0], inplace=True)

        elif ndim_a == ncon and ndim_b > ncon:
            # ~vecmat: a has locked axis and only one charge
            (axis,) = axes[1]
            b.select_charge(axis, a._sectors[0, 0], inplace=True)

        else:
            # ~vecvec: must have equal charges
            matching = (a.charge + b.charge) % a.order == 0

            if a.ndim > 1:
                a.sort_stack(axes[0], inplace=True)
            if b.ndim > 1:
                b.sort_stack(axes[1], inplace=True)

            # branchless set to zero
            a._blocks = a._blocks * matching
            b._blocks = b._blocks * matching

        if DEBUG:
            a.check()
            b.check()

        return a, b

    def _tensordot_inner_abelian(
        self, other, axes_a, axes_b, preserve_array=False
    ):
        """Perform the tensor inner product of two flat abelian arrays along
        the specified axes.

        Parameters
        ----------
        other : FlatArrayCommon
            The other array to contract with.
        axes_a : tuple[int, ...]
            The axes of this array to contract along.
        axes_b : tuple[int, ...]
            The axes of the other array to contract along.
        preserve_array : bool, optional
            Whether to always return an array, even if the result is a scalar.

        Returns
        -------
        array_like | FlatArrayCommon
            The result of the contraction, either as a scalar if
            `preserve_array=False` or else a new flat abelian array.
        """
        a, b = self.align_axes(other, axes=(axes_a, axes_b))

        linput = [-1, *repeat(None, a.ndim)]
        rinput = [-1, *repeat(None, b.ndim)]
        for c, (axa, axb) in enumerate(zip(axes_a, axes_b)):
            linput[axa + 1] = c
            rinput[axb + 1] = c

        c = ctg.array_contract(
            arrays=(a.blocks, b.blocks),
            inputs=(linput, rinput),
            output=(),
        )

        if preserve_array:
            c = a.__class__.from_scalar(c, symmetry=self._symmetry)

        return c

    def _tensordot_outer_abelian(
        self, other: "FlatArrayCommon"
    ) -> "FlatArrayCommon":
        """Perform the tensor outer product of two flat abelian arrays.

        Parameters
        ----------
        other : FlatArrayCommon
            The other array to contract with.

        Returns
        -------
        FlatArrayCommon
        """
        shape_a = self._get_shape_blocks_full()
        num_blocks_a = shape_a[0]
        shape_b = other._get_shape_blocks_full()
        num_blocks_b = shape_b[0]

        # do outer via broadcasted multiplication
        new_shape_a = (num_blocks_a, 1, *shape_a[1:], *repeat(1, other.ndim))
        new_shape_b = (1, num_blocks_b, *repeat(1, self.ndim), *shape_b[1:])
        new_blocks = ar.do("reshape", self.blocks, new_shape_a) * ar.do(
            "reshape", other.blocks, new_shape_b
        )
        # remerge batch index
        new_blocks = ar.do(
            "reshape",
            new_blocks,
            (num_blocks_a * num_blocks_b, *shape_a[1:], *shape_b[1:]),
        )

        # get new keys from 'broadcasted' concatenation
        ka = einops_repeat(self.sectors, "b r -> (b x) r", x=num_blocks_b)
        kb = einops_repeat(other.sectors, "b r -> (x b) r", x=num_blocks_a)
        new_sectors = ar.do("concatenate", (ka, kb), axis=1)

        new_indices = self.indices + other.indices

        return self.copy_with(
            sectors=new_sectors,
            indices=new_indices,
            blocks=new_blocks,
        )

    def trace(self):
        raise NotImplementedError()

    def einsum(self, eq, preserve_array=False):
        raise NotImplementedError

    def _tensordot_abelian(
        self,
        other,
        axes=2,
        mode="auto",
        preserve_array=False,
    ):
        return tensordot_abelian_flat(
            self,
            other,
            axes=axes,
            mode=mode,
            preserve_array=preserve_array,
        )

    def _matmul_abelian(
        self: "FlatArrayCommon",
        other: "FlatArrayCommon",
        preserve_array=False,
    ):
        a, b = self.align_axes(other, axes=((-1,), (0,)))

        # new sectors given by concatenation of the sectors
        new_sectors = ar.do(
            "concatenate",
            (a.sectors[:, :-1], b.sectors[:, 1:]),
            axis=1,
            like=self.backend,
        )

        inputs = []
        output = ["B"]
        if a.ndim == 1:
            inputs.append(("B", "x"))
        else:
            inputs.append(("B", "l", "x"))
            output.append("l")
        if b.ndim == 1:
            inputs.append(("B", "x"))
        else:
            inputs.append(("B", "x", "r"))
            output.append("r")

        # new full block given by batch matrix multiplication
        new_blocks = ctg.array_contract(
            arrays=(a.blocks, b.blocks),
            inputs=inputs,
            output=output,
        )

        new_indices = (*a.indices[:-1], *b.indices[1:])

        if new_indices or preserve_array:
            # array output, wrap in a new class
            return a.copy_with(
                blocks=new_blocks,
                sectors=new_sectors,
                indices=new_indices,
            )

        # scalar output
        return new_blocks[0]

    def multiply_diagonal(
        self,
        v: FlatVector,
        axis,
        power=1,
        inplace=False,
    ):
        """Multiply this flat array by a vector as if contracting a diagonal
        matrix along the given axis.

        Parameters
        ----------
        v : FlatVector
            The vector to contract with.
        axis : int
            The axis along which to contract.
        inplace : bool, optional
            Whether to perform the operation inplace.

        Returns
        -------
        FlatArrayCommon
        """
        if axis < 0:
            axis += self.ndim

        # find the order that sorts the sectors of `v` to match
        k = ar.do("argsort", self._sectors[:, axis])[v.sectors]
        vblocks_aligned = v.blocks[k]

        # expand with new dimensions
        reshaper = (
            slice(None),
            *repeat(None, axis),
            slice(None),
            *repeat(None, self.ndim - axis - 1),
        )

        if power == 1:
            new_blocks = self._blocks * vblocks_aligned[reshaper]
        elif power == -1:
            new_blocks = self._blocks / vblocks_aligned[reshaper]
        else:
            raise ValueError("Invalid power value")

        return self._modify_or_copy(blocks=new_blocks, inplace=inplace)

    def ldmul(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-2, inplace=inplace)

    def rdmul(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-1, inplace=inplace)

    def lddiv(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-2, power=-1, inplace=inplace)

    def rddiv(self, v, inplace=False):
        return self.multiply_diagonal(v, axis=-1, power=-1, inplace=inplace)

    def allclose(self, other: "FlatArrayCommon", **allclose_opts):
        """Check if two flat abelian arrays are equal to within some tolerance,
        including their sectors and signature.
        """
        # blocks might not be stored in the same order
        a = self.sort_stack()
        b = other.sort_stack()

        if a.duals != b.duals:
            return False

        if not ar.do("allclose", a.sectors, b.sectors, **allclose_opts):
            return False

        return ar.do("allclose", a.blocks, b.blocks, **allclose_opts)


def tensordot_flat_fused(
    a: FlatArrayCommon,
    b: FlatArrayCommon,
    left_axes: tuple[int, ...],
    axes_a: tuple[int, ...],
    axes_b: tuple[int, ...],
    right_axes: tuple[int, ...],
    preserve_array=False,
):
    # NOTE: we use abelian fusion as, while fermionic tensordot uses this
    # routine, it handles all phases ahead of time then expects abelian

    if left_axes:
        af = a._fuse_core_abelian(left_axes, axes_a)
    elif len(axes_a) > 1:
        af = a._fuse_core_abelian(axes_a)
    else:
        af = a

    if right_axes:
        bf = b._fuse_core_abelian(axes_b, right_axes)
    elif len(axes_b) > 1:
        bf = b._fuse_core_abelian(axes_b)
    else:
        bf = b

    cf = af.__matmul__(bf, preserve_array=preserve_array)

    if isinstance(cf, FlatArrayCommon):
        # if we got a new flat array, unfuse all axes
        for ax in reversed(range(cf.ndim)):
            if cf.is_fused(ax):
                cf = cf._unfuse_abelian(ax, inplace=True)

    return cf


def tensordot_flat_direct(
    a: FlatArrayCommon,
    b: FlatArrayCommon,
    left_axes: tuple[int, ...],
    axes_a: tuple[int, ...],
    axes_b: tuple[int, ...],
    right_axes: tuple[int, ...],
    preserve_array=False,
):
    """Contract two flat abelian arrays without fusing."""
    if not left_axes or not right_axes:
        raise NotImplementedError(
            "Currently both `a` and `b` must have kept axes for direct "
            "tensordot. Consider using mode='fused' instead."
        )

    _reshape = ar.get_lib_fn(a.backend, "reshape")

    dc = a.order ** (len(axes_a) - 1)

    # sort and reshape left blocks
    d0 = a.duals[axes_a[0]]
    lcon_sectors = zn_combine(
        a.order,
        a.sectors[:, axes_a],
        duals=[a.duals[ax] != d0 for ax in axes_a],
    )
    lkord = lexsort_sectors(
        (
            lcon_sectors,
            *(a.sectors[:, ax] for ax in left_axes),
            *(a.sectors[:, ax] for ax in axes_a),
        )
    )
    larray = _reshape(a.blocks[lkord], (a.order, -1, dc, *a.shape_block))

    # sort and reshape right blocks
    d0 = b.duals[axes_b[0]]
    rcon_sectors = zn_combine(
        b.order,
        b.sectors[:, axes_b],
        duals=[b.duals[ax] != d0 for ax in axes_b],
    )
    rkord = lexsort_sectors(
        (
            rcon_sectors,
            *(b.sectors[:, ax] for ax in right_axes),
            *(b.sectors[:, ax] for ax in axes_b),
        )
    )
    rarray = _reshape(b.blocks[rkord], (b.order, -1, dc, *b.shape_block))

    linput = ["B0", "Bl", "Bc"]
    rinput = ["B0", "Br", "Bc"]
    output = ["B0", "Bl", "Br"]

    aconmap = {}
    bconmap = {}
    for c, (axa, axb) in enumerate(zip(axes_a, axes_b)):
        aconmap[int(axa)] = c
        bconmap[int(axb)] = c

    for i in range(a.ndim):
        c = aconmap.get(i, None)
        if c is None:
            linput.append(f"a{i}")
            output.append(f"a{i}")
        else:
            linput.append(f"C{c}")

    for i in range(b.ndim):
        c = bconmap.get(i, None)
        if c is None:
            rinput.append(f"b{i}")
            output.append(f"b{i}")
        else:
            rinput.append(f"C{c}")

    new_blocks = ctg.array_contract(
        (larray, rarray), inputs=(linput, rinput), output=output
    )

    # flatten all batch indices
    db, dl, dr, *shape_new_block = new_blocks.shape
    new_blocks = _reshape(new_blocks, (db * dl * dr, *shape_new_block))

    # XXX: do we need the B axis at all here?

    # now we handle sectors
    lsectors = _reshape(a.sectors[lkord], (a.order, -1, dc, a.ndim))[
        :, :, 0, left_axes
    ]
    rsectors = _reshape(b.sectors[rkord], (b.order, -1, dc, b.ndim))[
        :, :, 0, right_axes
    ]

    new_sectors = ar.do(
        "concatenate",
        (
            einops_repeat(lsectors, "B Bl c -> (B Bl repeat) c", repeat=dr),
            einops_repeat(rsectors, "B Br c -> (B repeat Br) c", repeat=dl),
        ),
        axis=1,
    )

    new_indices = (
        *(a.indices[ax] for ax in left_axes),
        *(b.indices[ax] for ax in right_axes),
    )

    if new_indices or preserve_array:
        # array output, wrap in a new class
        return a.copy_with(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
        )

    # scalar output
    return new_blocks[0]


def tensordot_abelian_flat(
    a: FlatArrayCommon,
    b: FlatArrayCommon,
    axes=2,
    mode="auto",
    preserve_array=False,
):
    """Contract two flat abelian arrays along the specified axes."""
    left_axes, axes_a, axes_b, right_axes = parse_tensordot_axes(
        axes, a.ndim, b.ndim
    )

    if not axes_a:
        # outer product
        return a._tensordot_outer_abelian(b)

    if not (left_axes or right_axes):
        # inner product
        return a._tensordot_inner_abelian(b, axes_a, axes_b, preserve_array)

    if mode == "auto":
        if left_axes and right_axes:
            mode = "direct"
        else:
            mode = "fused"

    if mode == "direct":
        return tensordot_flat_direct(
            a,
            b,
            left_axes=left_axes,
            axes_a=axes_a,
            axes_b=axes_b,
            right_axes=right_axes,
            preserve_array=preserve_array,
        )
    elif mode == "fused":
        return tensordot_flat_fused(
            a,
            b,
            left_axes=left_axes,
            axes_a=axes_a,
            axes_b=axes_b,
            right_axes=right_axes,
            preserve_array=preserve_array,
        )
    else:
        raise ValueError(f"Unknown flat tensordot mode: {mode}")


def print_charge_fusions(keys, duals, axes_groups):
    from colorama import Fore

    RESET = Fore.RESET
    PINK = Fore.MAGENTA
    RED = Fore.RED
    BLUE = Fore.BLUE
    GREEN = Fore.GREEN

    color_options = [BLUE, RED, GREEN, PINK]

    (
        num_groups,
        _,  # group_singlets,
        new_ndim,
        _,  # perm,
        position,
        axes_before,
        axes_after,
        ax2group,
        _,  # group_duals,
        new_axes,
    ) = calc_fuse_group_info(axes_groups, duals)

    colors = dict(zip(range(num_groups), color_options))
    colors[None] = RESET
    old_key = None

    keys = ar.do("reshape", keys, (-1, ar.do("shape", keys)[-1]))

    for key in keys:
        new_key = [0] * new_ndim
        for oldax, newax in new_axes.items():
            new_key[newax] ^= int(key[oldax])

        if old_key is not None and old_key != new_key:
            print("- " * len(duals))
        old_key = new_key

        for i, c in enumerate(key):
            g = ax2group[i]
            print(f"{colors[g]}{c}{RESET}", end=" ")
        print("->", end=" ")

        for ax in axes_before:
            print(f"{key[ax]}", end=" ")
        for g in range(num_groups):
            print(f"{colors[g]}{new_key[position + g]}{RESET}", end=" ")
        for ax in axes_after:
            print(f"{key[ax]}", end=" ")

        print()


def build_cyclic_keys_all(ndim, order=2, flat=False, like=None):
    """For cyclic group of order `order`, build all possible subkeys of
    length `ndim`, in lexicographic order.

    Parameters
    ----------
    ndim : int
        The number of sub charges to build keys for.
    order : int, optional
        The order of the cyclic group, i.e., the number of distinct charges.
        Default is 2, which corresponds to the Z2 group.
    flat : bool, optional
        Whether to flatten the keys into a 2D array. If True, the output will
        be a 2D array of shape (order ** ndim, ndim). If False, the output will
        be a multi-dimensional array of shape `(order,) * ndim + (ndim,)`.
        Default is False.
    like : array_like, optional
        If provided, the output will be created with the same backend as this
        array. If None, the output will be created with the default backend,
        which is usually `numpy`.

    Returns
    -------
    array_like
        An array of shape (order ** ndim, ndim) or
        (order, order, ..., order, ndim) depending on the `flat` parameter.
    """
    kws = {}
    if like is None:
        kws["dtype"] = "int64"
    keys = ar.do("zeros", (order,) * ndim + (ndim,), like=like, **kws)

    for i in range(1, order):
        for j in range(ndim):
            selector = (
                *(
                    slice(i, i + 1) if k == j else slice(None)
                    for k in range(ndim)
                ),
                j,
            )
            keys[selector] = i

    if flat:
        # flatten the keys to a 2D array
        keys = ar.do("reshape", keys, (-1, ndim), like=like)

    return keys


def build_cyclic_keys_conserve(
    ndim,
    order=2,
    charge=0,
    duals=None,
    flat=False,
    like=None,
):
    """For cyclic group of order `order`, build all possible subkeys of
    length `ndim` with overall charge `charge`, in lexicographic order.

    Parameters
    ----------
    ndim : int
        The number of sub charges to build keys for.
    order : int, optional
        The order of the cyclic group, i.e., the number of distinct charges.
        Default is 2, which corresponds to the Z2 group.
    flat : bool, optional
        Whether to flatten the keys into a 2D array. If True, the output will
        be a 2D array of shape (order ** (ndim - 1), ndim). If False, the
        output will be a multi-dimensional array of shape
        `(order,) * (ndim - 1) + (ndim,)`.. Default is False.
    like : array_like, optional
        If provided, the output will be created with the same backend as this
        array. If None, the output will be created with the default backend,
        which is usually `numpy`.

    Returns
    -------
    array_like
        An array of shape (order ** (ndim - 1), ndim) or
        (order,) * (ndim - 1) + (ndim,) depending on the `flat` parameter.
    """
    kws = {}
    if like is None:
        kws["dtype"] = "int64"
    keys = ar.do("zeros", (order,) * (ndim - 1) + (ndim,), like=like, **kws)

    for i in range(1, order):
        for j in range(ndim - 1):
            selector = (
                *(
                    slice(i, i + 1) if k == j else slice(None)
                    for k in range(ndim - 1)
                ),
                j,
            )
            keys[selector] = i

    if duals is not None:
        signs = ar.do("array", [-1 if d else 1 for d in duals], like=keys)
        last_sign = signs[-1]
        signs = ar.do("reshape", signs, (1,) * ndim + (-1,), like=keys)
        keys[..., -1] = (
            last_sign * (charge - ar.do("sum", signs * keys, axis=-1))
        ) % order
    else:
        keys[..., -1] = (charge - (ar.do("sum", keys, axis=-1))) % order

    if flat:
        # flatten the keys to a 2D array
        keys = ar.do("reshape", keys, (-1, ndim), like=like)

    return keys


def build_cyclic_keys_by_charge(ndim, order=2, duals=None, like=None):
    """For cyclic group of order `order`, build all possible subkeys of
    length `ndim`, grouped (via the first axis) by their overall charge,
    then lexicographically ordered within each charge.

    Parameters
    ----------
    ndim : int
        The number of sub charges to build keys for.
    order : int, optional
        The order of the cyclic group, i.e., the number of distinct charges.
        Default is 2, which corresponds to the Z2 group.
    like : array_like, optional
        If provided, the output will be created with the same backend as this
        array. If None, the output will be created with the default backend,
        which is usually `numpy`.

    Returns
    -------
    array_like
        An array of shape (order, order ** (ndim - 1), ndim).
    """
    kws = {}
    if like is None:
        kws["dtype"] = "int64"
    keys = ar.do("zeros", (order,) * ndim + (ndim,), like=like, **kws)

    for i in range(1, order):
        for j in range(ndim - 1):
            selector = (
                slice(None),
                *(
                    slice(i, i + 1) if k == j else slice(None)
                    for k in range(ndim - 1)
                ),
                j,
            )
            keys[selector] = i

    # flatten all but the overall charge and sector axis
    keys = ar.do("reshape", keys, (order, -1, ndim), like=like)

    # create broadcastable array of possible total charges
    c_total = ar.do("reshape", ar.do("arange", order), (order, 1), like=like)

    # compute last column of each sector, based on the total charge
    if duals is not None:
        signs = ar.do("array", [-1 if d else 1 for d in duals], like=keys)
        last_sign = signs[-1]
        signs = ar.do("reshape", signs, (1,) * (ndim + 1) + (-1,), like=keys)
        c_last = (
            last_sign
            * (c_total - ar.do("sum", signs * keys, axis=-1, like=like))
        ) % order
    else:
        c_last = (c_total - ar.do("sum", keys, axis=-1, like=like)) % order
    keys[:, :, -1] = c_last

    return keys
