"""Flat Abelian array implementation.

TODO:

- [ ] cache properties
- [ ] cache patterns and reshapers/slicers
- [ ] store size in FlatIndex and remove extents?

"""

import functools
from itertools import repeat

import autoray as ar

from .abelian_core import (
    AbelianCommon,
    AbelianArray,
    calc_fuse_group_info,
    get_zn_array_cls,
)
from .symmetries import get_symmetry


class FlatIndex:
    """Simple class to store dualness and any fuse information of an index.

    Parameters
    ----------
    dual : bool, optional
        Whether the index is dual (i.e., contributes a negative sign to the
        corresponding charge). Default is False.
    subinfo : FlatSubIndexInfo, optional
        Information about the subindex, if this index is a fused index.
        Default is None, which means the index is not fused.
    """

    __slots__ = ("_dual", "_subinfo")

    def __init__(self, dual=False, subinfo=None):
        self._dual = dual
        self._subinfo = subinfo

    @property
    def dual(self) -> bool:
        """Whether the index flows 'outwards' / (+ve) / ket-like = ``False`` or
        'inwards' / (-ve) / bra-like= ``True``. The charge sign is given by
        ``(-1) ** dual``.
        """
        return self._dual

    @property
    def subinfo(self) -> "FlatSubIndexInfo | None":
        """Information about the subindices of this index and their extents if
        this index was formed from fusing.
        """
        return self._subinfo

    def conj(self) -> "FlatIndex":
        """Return the conjugate of the index, i.e., flip the dualness and
        subinfo.
        """
        return FlatIndex(
            dual=not self._dual,
            subinfo=None if self._subinfo is None else self._subinfo.conj(),
        )


class FlatSubIndexInfo:
    """Information required to unfuse a fused index.

    Parameters
    ----------
    indices : tuple[FlatIndex]
        The indices that have been fused.
    subkeys : array_like
        The subkeys for the fused index, with shape
        (ncharge, nsectors, nsubcharges). I.e. the first axis selects the
        overall fused charge, the second axis selects the subsector within
        that charge, and the third axis selects the individual charge within
        that subsector.
    extents : tuple[int]
        The size of each subindex, i.e. 'sub shape'.
    """

    __slots__ = (
        "_indices",
        "_subkeys",
        "_extents",
        "_ncharge",
        "_nsectors",
        "_nsubcharges",
    )

    def __init__(self, indices, subkeys, extents):
        self._indices = tuple(
            x if isinstance(x, FlatIndex) else FlatIndex(x) for x in indices
        )
        self._subkeys = subkeys
        subkey_shape = ar.do("shape", subkeys)
        # number of overall charges, e.g. {0, 1} -> 2
        # number of subsectors e.g. [000, 011, 101, 110] -> 4
        # number of subcharges, e.g. 3 for above
        self._ncharge, self._nsectors, self._nsubcharges = subkey_shape
        self._extents = tuple(extents)

    @property
    def indices(self) -> tuple[FlatIndex]:
        """The subkeys for the fused index, with shape
        (ncharge, nsectors, nsubcharges). I.e. the first axis selects the
        overall fused charge, the second axis selects the subsector within
        that charge, and the third axis selects the individual charge within
        that subsector."""
        return self._indices

    @property
    def extents(self) -> tuple[int]:
        """The extents of the fused index."""
        return self._extents

    @property
    def subkeys(self):
        """The subkeys for the fused index."""
        return self._subkeys

    @property
    def ncharge(self) -> int:
        """Number of overall charges in this subindex."""
        return self._ncharge

    @property
    def nsectors(self) -> int:
        """Number of subsectors in this subindex."""
        return self._nsectors

    @property
    def nsubcharges(self) -> int:
        """Number of subcharges in this subindex."""
        return self._nsubcharges

    def check(self):
        assert len(self._indices) == len(self._subkeys[0])
        assert len(self._indices) == len(self._extents)

    def conj(self) -> "FlatSubIndexInfo":
        return FlatSubIndexInfo(
            indices=tuple(ix.conj() for ix in self._indices),
            subkeys=self._subkeys,
            extents=self._extents,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ncharge={self._ncharge}, "
            f"nsectors={self._nsectors}, "
            f"nsubcharges={self._nsubcharges}, "
            f"extents={self._extents})"
        )


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
    if not ar.is_array(sectors):
        sectors = ar.do("stack", sectors, axis=1)

    backend = ar.infer_backend(sectors)

    n = ar.do("shape", sectors, like=backend)[1]
    limits = ar.do("max", sectors, axis=0, like=backend) + 1
    ws = ar.do("ones", (1, n), dtype=sectors.dtype, like=backend)
    for ax in range(n - 2, -1, -1):
        # reverse cumulative product to get 'strides'
        ws[0, ax] = ws[0, ax + 1] * limits[ax]

    ranks = ar.do("sum", ws * sectors, axis=1, like=backend)
    return ar.do("argsort", ranks, stable=stable, like=backend)


def zn_combine(sectors, duals=None, order=2, like=None):
    """Implement vectorized addition modulo group order, with signature.

    Parameters
    ----------
    sectors : array_like
        The stack of sectors, with shape (num_blocks, num_charges). Each
        row represents a subsector.
    duals : sequence[bool] | None, optional
        The dualness of each index, i.e., whether the charge contributes
        positively or negatively. If not given, it will be assumed that all
        charges are positive.
    order : int, optional
        The order of the symmetry group, i.e., the number of distinct charges
        in each axis. Default is 2, which corresponds to Z2 symmetry.
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


class AbelianArrayFlat(AbelianCommon):
    """Base class for abelian arrays with flat storage and cyclic symmetry.

    Parameters
    ----------
    sectors : array_like
        The stack of sector keys, with shape (num_blocks, ndim). Each row
        represents a sector of a corresponding block, and each column
        represents a charge in a given axis.
    blocks : array_like
        The stack of array blocks, with shape (num_blocks, *shape_block), i.e.
        `ndim + 1` dimensions, where the first dimension is the block index,
        which should match the first dimension of `sectors`, and the rest are
        the dimensions of individual blocks.
    indices : sequence[FlatIndex]
        Indices describing the dualness and any subindex information for each
        dimension of the array. If bools are supplied, they will be converted
        to a FlatIndex with the corresponding dualness, and no subindex
        information.
    """

    fermionic = False
    static_symmetry = None

    def __init__(
        self,
        sectors,
        blocks,
        indices,
        symmetry=None,
    ):
        self._sectors = (
            sectors if hasattr(sectors, "shape") else ar.do("array", sectors)
        )
        self._blocks = (
            blocks if hasattr(blocks, "shape") else ar.do("array", blocks)
        )
        self._indices = tuple(
            # allow sequence of duals to be supplied directly
            x if isinstance(x, FlatIndex) else FlatIndex(x)
            for x in indices
        )
        self._symmetry = self.get_class_symmetry(symmetry)

        # infer the backend to reuse for efficiency
        self.backend = ar.infer_backend(self._blocks)

    @property
    def order(self) -> int:
        """Get the order of the symmetry group."""
        return self._symmetry.N

    @property
    def sectors(self):
        """The stack of sector keys, with shape (num_blocks, ndim). Each row
        represents a sector of a corresponding block, and each column
        represents a charge in a given axis."""
        return self._sectors

    @property
    def blocks(self):
        """The stack of array blocks, with shape (num_blocks, *shape_block),
        i.e. `ndim + 1` dimensions, where the first dimension is the block
        index, which should match the first dimension of `sectors`, and the
        rest are the dimensions of individual blocks."""
        return self._blocks

    @property
    def indices(self) -> tuple[FlatIndex]:
        """Indices describing the dualness and any subindex information for \
        each dimension of the array."""
        return self._indices

    @property
    def charge(self):
        """Compute the overall charge of the array."""
        return zn_combine(self._sectors[[0], :], self.duals, self.order)[0]

    @property
    def duals(self) -> tuple[bool]:
        """Get the dualness of each index."""
        return tuple(index.dual for index in self._indices)

    def check(self):
        assert len(self._sectors) == len(self._blocks)
        assert self.ndim == len(self._indices)
        assert self.ndim == len(self._sectors[0])
        # check blocks all have the same overall charge
        sector_charges = ar.do(
            "unique", zn_combine(self._sectors, self.duals, self.order)
        )
        assert ar.do("size", sector_charges) == 1

    def copy(self, deep=False) -> "AbelianArrayFlat":
        """Create a copy of the array."""
        if deep:
            sectors = ar.do("copy", self._sectors, like=self.backend)
            blocks = ar.do("copy", self._blocks, like=self.backend)
        else:
            sectors = self._sectors
            blocks = self._blocks
        return self.__class__(sectors, blocks, self._indices)

    def _modify_or_copy(
        self, sectors=None, blocks=None, indices=None, inplace=False
    ) -> "AbelianArrayFlat":
        sectors = self._sectors if sectors is None else sectors
        blocks = self._blocks if blocks is None else blocks
        indices = self._indices if indices is None else indices

        if inplace:
            self._sectors = sectors
            self._blocks = blocks
            self._indices = indices
            return self
        else:
            return self.__class__(
                sectors=sectors,
                blocks=blocks,
                indices=indices,
            )

    def _get_shape_blocks_full(self) -> tuple[int, ...]:
        """Get the full shape of the stacked blocks, including the number of
        blocks."""
        return ar.do("shape", self._blocks, like=self.backend)

    @property
    def shape_block(self) -> tuple[int, ...]:
        """Get the shape of an individual block."""
        return self._get_shape_blocks_full()[1:]

    @property
    def ndim(self) -> int:
        """Get the number of effective dimensions of the array."""
        return len(self.shape_block)

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the effective shape of the array."""
        return tuple(self.order * d for d in self.shape_block)

    @property
    def num_blocks(self) -> int:
        """Get the number of blocks in the array."""
        return self._get_shape_blocks_full()[0]

    @classmethod
    def from_blocks(cls, blocks, indices) -> "AbelianArrayFlat":
        """Create a flat array from an explicit dictionary of blocks, and
        sequence of indices or duals.

        Parameters
        ----------
        blocks : dict[tuple[int, ...], array_like]
            A dictionary mapping sector keys (tuples of charges) to blocks
            (arrays).
        indices : sequence[FlatIndex] | sequence[bool]
            A sequence of indices describing the dualness and any subindex
            information for each dimension of the array. If bools are supplied,
            they will be converted to a FlatIndex with the corresponding
            dualness, and no subindex information.

        Returns
        -------
        AbelianArrayFlat
        """
        sectors = []
        full_blocks = None
        fshape = None
        for i, key in enumerate(sorted(blocks)):
            block = blocks[key]
            bshape = ar.do("shape", block)
            if full_blocks is None:
                fshape = (len(blocks), *bshape)
                full_blocks = ar.do("empty", fshape, like=block)
            sectors.append(list(key))
            full_blocks[i] = block
        return cls(sectors, full_blocks, indices)

    @classmethod
    def from_blocksparse(cls, x: AbelianArray) -> "AbelianArrayFlat":
        """Create a flat abelian array from a blocksparse abelian array.

        Parameters
        ----------
        x : AbelianArray
            The blocksparse abelian array to convert.
        """
        return cls.from_blocks(blocks=x.blocks, indices=x.duals)

    def to_blocksparse(self) -> AbelianArray:
        """Create a blocksparse abelian array from this flat abelian array."""
        blocks = {}
        for i in range(self.num_blocks):
            sector = tuple(map(int, self._sectors[i]))
            block = self._blocks[i]
            blocks[sector] = block
        cls = get_zn_array_cls(self.order)
        return cls.from_blocks(blocks, duals=self.duals)

    def is_fused(self, ax: int) -> bool:
        """Does axis `ax` carry subindex information, i.e., is it a fused
        index?
        """
        return self._indices[ax].subinfo is not None

    def sort_stack(
        self,
        axes=None,
        all_axes=None,
        inplace=False,
    ) -> "AbelianArrayFlat":
        """Lexicgraphic sort the stack of sectors and blocks according to the
        values of charges in the specified axes, optionally filling in the rest
        of the axes with the remaining axes in the order they appear.

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
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.
            Default is False, which returns a new array.
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
        kord = lexsort_sectors(cols)
        new_sectors = self._sectors[kord]
        new_blocks = self._blocks[kord]

        return self._modify_or_copy(
            sectors=new_sectors, blocks=new_blocks, inplace=inplace
        )

    def expand_dims(
        self, axis, c=None, dual=None, inplace=False
    ) -> "AbelianArrayFlat":
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
        AbelianArrayFlat
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
            FlatIndex(dual),
            *self._indices[axis:],
        )

        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def _fuse_core(
        self,
        *axes_groups,
        inplace=False,
    ) -> "AbelianArrayFlat":
        """The core implementation of the fuse operation, which fuses
        multiple axes into a single group, and returns a new array with
        the new sectors and blocks. The new axes are inserted at the minimum
        axis of any of the groups.
        """
        from einops import rearrange

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
                zn_combine(self._sectors[:, axs], eff_duals, self.order)
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
            if axes_seen == self.ndim:
                # if all axes are being fused, the last one is locked
                keys_reshaper.append(1)
                if num_groups > 1:
                    # all axes are fused, multiple new charges, last of which
                    # is locked to prev, so we can't take a slice across it
                    # directly, nor will it be sorted
                    g_locked_map[g] = g - 1
            else:
                # one dimension for each fused charge
                keys_reshaper.append(self.order)
        # one dimension for all the unfused axes
        keys_reshaper.append(-1)
        # then one dimension for each *extra* subcharge in each group
        nsubaxes = 0
        for group in axes_groups:
            # each group has one locked charge within it
            nsub = len(group) - 1
            keys_reshaper.extend([self.order] * nsub)
            nsubaxes += nsub

        # reshape! including finally one dimension for the row of charges
        new_sectors = ar.do("reshape", new_sectors, (*keys_reshaper, new_ndim))
        old_sectors = ar.do(
            "reshape", old_sectors, (*keys_reshaper, self.ndim)
        )

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
        subkeys = {}
        for g in range(num_groups):
            if g in group_singlets:
                # no need to record subcharges
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
            subkey = ar.do(
                "reshape", subkey, (keys_reshaper[g_lock], -1, self.ndim)
            )
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
                kord = ar.do("argsort", gcharges, like=self.backend)
                subkey = subkey[kord]

            # store
            subkeys[g] = subkey

        # reflatten new keys into stack
        new_sectors = ar.do(
            "reshape", new_sectors, (-1, new_ndim), like=self.backend
        )

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
        for ax in range(self.ndim):
            pattern.append(f" p{ax}")

        # RHS, start with the new block index
        pattern.append(" -> B")

        # then add the unfused output axes before the groups
        for ax in axes_before:
            pattern.append(f" p{ax}")

        # then the groups, each looks like `(B0 p5 p2 ...)`, one dimension
        # coming from the batch index, and the rest real axis fusions
        unmerged_batch_sizes = {}
        for g in range(num_groups):
            pattern.append(f" (B{g}")
            for ax in axes_groups[g]:
                pattern.append(f" p{ax}")
                bax = f"B{g}"
                # keep track of the unmerged
                if bax in unmerged_batch_sizes:
                    unmerged_batch_sizes[bax] *= self.order
                else:
                    unmerged_batch_sizes[bax] = 1
            pattern.append(")")

        # then add the unfused output axes after the groups
        for ax in axes_after:
            pattern.append(f" p{ax}")

        pattern = "".join(pattern)

        # perform the rearrangement!
        new_blocks = rearrange(new_blocks, pattern, **unmerged_batch_sizes)

        # # then we update the sectors, these have been sorted already by
        # # grouped charge, each group being of equal size / stride:
        # stride = math.prod(unmerged_batch_sizes.values())
        # new_sectors = new_sectors[::stride]

        # if num_groups == 1 and len(axes_groups[0]) == self.ndim:
        #     # full fuse, only one overall charge, subkeys are all current keys
        #     subkeys = {0: ar.do("reshape", self._sectors, (1, -1, self.ndim))}
        # else:
        #     subkeys = {
        #         g: build_cyclic_keys_by_charge(
        #             ndim=len(axs),
        #             order=self.order,
        #             duals=[self.duals[ax] != group_duals[g] for ax in axs],
        #             # duals=[self.duals[ax] for ax in axs],
        #             like=self._sectors,
        #         )
        #         for g, axs in enumerate(axes_groups)
        #     }

        # finally we construct info, including for unfusing
        new_indices = []

        for ax in axes_before:
            new_indices.append(self.indices[ax])

        old_indices = self.indices
        old_shape_block = self.shape_block

        for g in range(num_groups):
            if g in group_singlets:
                # no new fuse -> just propagate any current info
                new_indices.append(old_indices[axes_groups[g][0]])
            else:
                # create a new subinfo for this group
                sub_indices = []
                sub_extents = []

                for ax in axes_groups[g]:
                    sub_indices.append(old_indices[ax])
                    sub_extents.append(old_shape_block[ax])

                subinfo = FlatSubIndexInfo(
                    indices=sub_indices,
                    subkeys=subkeys[g],
                    extents=sub_extents,
                )
                new_indices.append(FlatIndex(group_duals[g], subinfo=subinfo))

        for ax in axes_after:
            new_indices.append(self.indices[ax])

        return self._modify_or_copy(
            sectors=new_sectors,
            blocks=new_blocks,
            indices=new_indices,
            inplace=inplace,
        )

    def unfuse(self, axis, inplace=False) -> "AbelianArrayFlat":
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.

        Parameters
        ----------
        axis : int
            The axis to unfuse. It must have fuse information
            (`.indices[axis].subinfo`), typically from a previous fusing
            operation.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArrayFlat
        """
        from einops import rearrange, repeat

        new = self if inplace else self.copy()

        fi = new.indices[axis].subinfo
        if fi is None:
            raise ValueError(f"Axis {axis} is not fused in this array.")

        axs_rem = tuple(range(axis)) + tuple(range(axis + 1, self.ndim))
        new.sort_stack((axis, *axs_rem), inplace=True)

        # keys coming from remaining axes
        ka = repeat(
            new.sectors, "(Bf B) s -> (Bf B x) s", Bf=fi.ncharge, x=fi.nsectors
        )[:, axs_rem]

        # keys coming from unfused axis
        kb = fi.subkeys
        kb = repeat(kb, "B Bu s -> (B x Bu) s", x=new.num_blocks // fi.ncharge)

        # concatenate into the full new keys!
        new_keys = ar.do(
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
        for g, sz in enumerate(fi.extents):
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
        new_blocks = rearrange(new.blocks, pattern, **sizes)

        # unpack sub indices
        new_indices = (
            *new.indices[:axis],
            *fi.indices,
            *new.indices[axis + 1 :],
        )

        new._sectors = new_keys
        new._blocks = new_blocks
        new._indices = new_indices

        return new

    def conj(self, inplace=False) -> "AbelianArrayFlat":
        """Return the complex conjugate of this block array, including the
        indices and any subindex fusing information.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArrayFlat
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

    def transpose(self, axes=None, inplace=False) -> "AbelianArrayFlat":
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
        AbelianArrayFlat
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

    def select_charge(self, axis, charge, inplace=False):
        """Drop all but the specified charge along the specified axis."""
        new = self.sort_stack(axis, inplace=inplace)

        shp_sectors = ar.do("shape", new.sectors, like=self.backend)
        shp_blocks = ar.do("shape", new.blocks, like=self.backend)

        dc = self.order
        dB = shp_sectors[0]

        new_sectors = ar.do(
            "reshape", new.sectors, (dc, dB // dc, *shp_sectors[1:])
        )[charge]

        new_blocks = ar.do(
            "reshape", new.blocks, (dc, dB // dc, *shp_blocks[1:])
        )[charge]

        new._sectors = new_sectors
        new._blocks = new_blocks
        return new

    def align_axes(
        self: "AbelianArrayFlat",
        other: "AbelianArrayFlat",
        axes: tuple[tuple[int, ...], tuple[int, ...]],
        inplace=False,
    ) -> tuple["AbelianArrayFlat", "AbelianArrayFlat"]:
        """Align the axes of two arrays for contraction."""
        ndim_a = self.ndim
        ndim_b = other.ndim

        if ndim_a >= 2 and ndim_b >= 2:
            # ~ matmat: just need to sort sectors along common axes
            return (
                self.sort_stack(axes[0], inplace=inplace),
                other.sort_stack(axes[1], inplace=inplace),
            )

        elif ndim_a >= 2 and ndim_b == 1:
            # ~matvec: b has locked axis and only one charge
            (axis,) = axes[0]
            a = self.select_charge(axis, other.charge, inplace=inplace)
            return a, other

        elif ndim_a == 1 and ndim_b >= 2:
            # vec~mat: a has locked axis and only one charge
            (axis,) = axes[1]
            b = other.select_charge(axis, self.charge, inplace=inplace)
            return self, b

        else:
            raise NotImplementedError(
                "Cannot align axes for arrays with shapes: "
                f"{self.shape} and {other.shape} yet."
            )

    def __matmul__(
        self: "AbelianArrayFlat",
        other: "AbelianArrayFlat",
        preserve_array=False,
    ):
        import cotengra as ctg

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

        return a.__class__(
            blocks=new_blocks,
            sectors=new_sectors,
            indices=new_indices,
        )

    def trace(self):
        raise NotImplementedError()

    def multiply_diagonal(self, v, axis, inplace=False):
        raise NotImplementedError()

    def einsum(self, eq, preserve_array=False):
        raise NotImplementedError()

    def __reduce__(self):
        return (get_zn_array_flat_cls, (self.order,))

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"shape~{self.shape}:[{self.signature}], "
            f"num_blocks={self.num_blocks})"
        )


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


class Z2ArrayFlat(AbelianArrayFlat):
    static_symmetry = get_symmetry("Z2")


@functools.cache
def get_zn_array_flat_cls(n):
    """Get a block array class with ZN symmetry."""
    if n == 2:
        return Z2ArrayFlat

    return type(
        f"Z{n}ArrayFlat",
        (AbelianArrayFlat,),
        {"static_symmetry": get_symmetry(f"Z{n}")},
    )


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
            charge - last_sign * (ar.do("sum", signs * keys, axis=-1))
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
            c_total
            - last_sign * ar.do("sum", signs * keys, axis=-1, like=like)
        ) % order
    else:
        c_last = (c_total - ar.do("sum", keys, axis=-1, like=like)) % order
    keys[:, :, -1] = c_last

    return keys
