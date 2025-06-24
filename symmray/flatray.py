import functools
import math
from itertools import repeat

import autoray as ar

from symmray.abelian_core import (
    AbelianCommon,
    calc_fuse_group_info,
    get_zn_array_cls,
)


class FuseInfo:
    """Information required to unfuse a fused index."""

    def __init__(self, subkeys, extents, duals, fuseinfos=None):
        self.subkeys = subkeys
        subkey_shape = ar.do("shape", subkeys)
        # number of overall charges, e.g. {0, 1} -> 2
        # number of subsectors e.g. [000, 011, 101, 110] -> 4
        # number of subcharges, e.g. 3 for above
        self.ncharge, self.nsectors, self.nsubcharges = subkey_shape
        self.extents = tuple(extents)
        self.duals = tuple(duals)
        if fuseinfos is None:
            self.fuseinfos = (None,) * self.nsubcharges
        else:
            self.fuseinfos = tuple(fuseinfos)

    def conj(self):
        new_duals = tuple(not d for d in self.duals)
        new_fuseinfos = tuple(
            fi.conj() if fi is not None else None for fi in self.fuseinfos
        )
        return FuseInfo(
            subkeys=self.subkeys,
            extents=self.extents,
            duals=new_duals,
            fuseinfos=new_fuseinfos,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"ncharge={self.ncharge}, "
            f"nsectors={self.nsectors}, "
            f"nsubcharges={self.nsubcharges}, "
            f"extents={self.extents}, "
            f"duals={self.duals}, "
            f"fuseinfos={self.fuseinfos})"
        )


def lexsort_keys(vectors, stable=True):
    """Given a sequence of vectors of positive integers, find the indices that
    sort them lexicographically, such that the first vector is the most
    significant.

    Parameters
    ----------
    vectors : array_like or sequence of array_like
        The vectors to sort, each vector should be a 1D array of positive
        integers. Either supplied as a 2D array, or a sequence of 1D arrays.
    stable : bool, optional
        Whether to use a stable sort. Default is True, which uses the
        `argsort` function with the `stable` parameter set to True. If False,
        it will use the default sorting behavior which may not be stable.

    Returns
    -------
    array_like
        The indices that would sort the vectors lexicographically.

    Examples
    --------

        >>> vectors = np.array([[4, 1, 0], [3, 2, 1], [3, 1, 0], [2, 0, 1]])
        >>> k = lexsort_keys(vectors)
        >>> vectors[k]
        array([[2, 0, 1],
               [3, 1, 0],
               [3, 2, 1],
               [4, 1, 0]])
    """
    if not ar.is_array(vectors):
        vectors = ar.do("stack", vectors, axis=1)

    backend = ar.infer_backend(vectors)

    n = ar.do("shape", vectors, like=backend)[1]
    limits = ar.do("max", vectors, axis=0, like=backend) + 1
    ws = ar.do("ones", (1, n), dtype=vectors.dtype, like=backend)
    for ax in range(n - 2, -1, -1):
        # reverse cumulative product to get 'strides'
        ws[0, ax] = ws[0, ax + 1] * limits[ax]

    ranks = ar.do("sum", ws * vectors, axis=1, like=backend)
    return ar.do("argsort", ranks, stable=stable, like=backend)


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


def zn_combine(sectors, duals=None, order=2, like=None):
    """Implement vectorized addition modulo group order, with signature."""
    if order == 2:
        # self inverse, no need to check duals
        return ar.do("sum", sectors, axis=-1, like=like) % 2

    if duals is not None:
        signs = [(-1) ** dual for dual in duals]
        signs = ar.do("array", signs, like=like)
        signs = ar.do("reshape", signs, (1, -1))
        signed_sectors = sectors * signs
    else:
        signed_sectors = sectors

    return ar.do("sum", signed_sectors, axis=-1, like=like) % order


class AbelianArrayFlat(AbelianCommon):
    order = None

    def __init__(self, fkeys, fblock, duals, fuseinfos=None):
        self.fkeys = ar.do("array", fkeys)
        self.fblock = ar.do("array", fblock)
        self.backend = ar.infer_backend(self.fblock)
        self.duals = tuple(map(bool, duals))
        if fuseinfos is None:
            self.fuseinfos = (None,) * self.ndim
        else:
            self.fuseinfos = tuple(fuseinfos)

    def copy(self, deep=False):
        """Create a copy of the array."""
        if deep:
            fkeys = ar.do("copy", self.fkeys, like=self.backend)
            fblock = ar.do("copy", self.fblock, like=self.backend)
        else:
            fkeys = self.fkeys
            fblock = self.fblock
        return self.__class__(fkeys, fblock, self.duals, self.fuseinfos)

    def check(self):
        assert len(self.fkeys) == len(self.fblock)
        assert self.ndim == len(self.duals)
        assert self.ndim == len(self.fkeys[0])
        assert self.ndim == len(self.fuseinfos)
        # check blocks all have the same overall charge
        sector_charges = ar.do(
            "unique", zn_combine(self.fkeys, self.duals, self.order)
        )
        assert ar.do("size", sector_charges) == 1

    def _get_shape_fblock(self):
        """Get the full shape of the fblock"""
        return ar.do("shape", self.fblock, like=self.backend)

    @property
    def shape_block(self):
        """Get the shape of an individual block."""
        return self._get_shape_fblock()[1:]

    @property
    def ndim(self):
        """Get the number of effective dimensions of the array."""
        return len(self.shape_block)

    @property
    def shape(self):
        """Get the effective shape of the array."""
        return tuple(self.order * d for d in self.shape_block)

    @property
    def num_blocks(self):
        """Get the number of blocks in the array."""
        return self._get_shape_fblock()[0]

    @classmethod
    def from_blocks(cls, blocks, duals):
        fkeys = []
        fblock = None
        fshape = None
        for i, key in enumerate(sorted(blocks)):
            block = blocks[key]
            bshape = ar.do("shape", block)
            if fblock is None:
                fshape = (len(blocks), *bshape)
                fblock = ar.do("empty", fshape, like=block)
            fkeys.append(list(key))
            fblock[i] = block
        return cls(fkeys, fblock, duals)

    @classmethod
    def from_blocksparse(cls, x):
        return cls.from_blocks(blocks=x.blocks, duals=x.duals)

    def to_blocksparse(self):
        blocks = {}
        for i in range(self.num_blocks):
            sector = tuple(map(int, self.fkeys[i]))
            block = self.fblock[i]
            blocks[sector] = block
        cls = get_zn_array_cls(self.order)
        return cls.from_blocks(blocks, duals=self.duals)

    def is_fused(self, ax):
        """Does axis `ax` carry subindex information, i.e., is it a fused
        index?
        """
        return self.fuseinfos[ax] is not None

    def sort_stack(self, axes=(), all_axes=None):
        """Lexicgraphic sort the stack of blocks according to the values of
        charges in the specified axes, optionally filling in the rest of the
        axes with the remaining axes in the order they appear.
        """
        if all_axes:
            axes = (*axes, *(ax for ax in range(self.ndim) if ax not in axes))
        cols = self.fkeys[:, axes]
        kord = lexsort_keys(cols)
        self.fkeys = self.fkeys[kord]
        self.fblock = self.fblock[kord]

    def expand_dims(self, axis, c=None, dual=None, inplace=False):
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
                dual = self.duals[axis - 1]
            elif axis < self.ndim:
                # inherit from right
                dual = self.duals[axis]
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
        new_fkeys = ar.do(
            "concatenate",
            (self.fkeys[:, :axis], keys_new_col, self.fkeys[:, axis:]),
            axis=1,
            like=self.backend,
        )

        # expand the actual blocks (accounting for extra flat ax at start)
        selector = (
            (slice(None),) * (axis + 1)
            + (None,)
            + (slice(None),) * (self.ndim - axis - 1)
        )
        new_fblock = self.fblock[selector]

        # expand the index information
        new_duals = self.duals[:axis] + (dual,) + self.duals[axis:]
        new_fuseinfos = self.fuseinfos[:axis] + (None,) + self.fuseinfos[axis:]

        if inplace:
            self.fkeys = new_fkeys
            self.fblock = new_fblock
            self.duals = new_duals
            self.fuseinfos = new_fuseinfos
            return self
        else:
            return self.__class__(
                fkeys=new_fkeys,
                fblock=new_fblock,
                duals=new_duals,
                fuseinfos=new_fuseinfos,
            )

    def _fuse_core(
        self,
        *axes_groups,
        inplace=False,
    ):
        """ """
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
        new_fkeys = [self.fkeys[:, ax] for ax in axes_before]
        for axs, dg in zip(axes_groups, group_duals):
            # charges with opposite sign to overall group need to be flipped
            eff_duals = [self.duals[ax] != dg for ax in axs]
            new_fkeys.append(
                zn_combine(self.fkeys[:, axs], eff_duals, self.order)
            )
        # then we add the unfused axes after
        new_fkeys.extend(self.fkeys[:, ax] for ax in axes_after)
        # combine into single array
        new_fkeys = ar.do("stack", tuple(new_fkeys), axis=1)

        # then we find the correct order to sort the new keys
        sortingcols = (
            # first we sort by fused charge
            *(new_fkeys[:, pos + g] for g in range(num_groups)),
            # then we sort by the unfused axes
            *(self.fkeys[:, ax] for ax in axes_before),
            *(self.fkeys[:, ax] for ax in axes_after),
            # and finally by the fused charges within each group
            *(self.fkeys[:, ax] for group in axes_groups for ax in group),
        )
        kord = lexsort_keys(sortingcols)
        new_blocks = self.fblock[kord]
        new_fkeys = new_fkeys[kord]

        old_fkeys = self.fkeys[kord]

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
        new_fkeys = ar.do("reshape", new_fkeys, (*keys_reshaper, new_ndim))
        old_fkeys = ar.do("reshape", old_fkeys, (*keys_reshaper, self.ndim))

        # drop sub charge axes from new_fkeys
        new_fkeys = new_fkeys[
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
            subkey = old_fkeys[subkey_selector]
            # flatten into (overall_charge, subcharges, row)
            subkey = ar.do(
                "reshape", subkey, (keys_reshaper[g_lock], -1, self.ndim)
            )
            # just get axes relevant to this group
            subkey = subkey[:, :, axes_groups[g]]

            if g_lock != g:
                # this group axis is locked to the previous group,
                # so we need to do an additional sort the overall charge axis
                gcharges = new_fkeys[
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
        new_fkeys = ar.do(
            "reshape", new_fkeys, (-1, new_ndim), like=self.backend
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
        # new_fkeys = new_fkeys[::stride]

        # if num_groups == 1 and len(axes_groups[0]) == self.ndim:
        #     # full fuse, only one overall charge, subkeys are all current keys
        #     subkeys = {0: ar.do("reshape", self.fkeys, (1, -1, self.ndim))}
        # else:
        #     subkeys = {
        #         g: build_cyclic_keys_by_charge(
        #             ndim=len(axs),
        #             order=self.order,
        #             duals=[self.duals[ax] != group_duals[g] for ax in axs],
        #             # duals=[self.duals[ax] for ax in axs],
        #             like=self.fkeys,
        #         )
        #         for g, axs in enumerate(axes_groups)
        #     }

        # finally we construct info, including for unfusing
        new_duals = []
        new_fuseinfos = []

        for ax in axes_before:
            new_duals.append(self.duals[ax])
            new_fuseinfos.append(self.fuseinfos[ax])

        new_duals.extend(group_duals)

        old_duals = self.duals
        old_shape_block = self.shape_block
        old_fuseinfos = self.fuseinfos

        for g in range(num_groups):
            if g in group_singlets:
                # no new fuse -> just propagate any current info
                new_fuseinfos.append(self.fuseinfos[axes_groups[g][0]])
            else:
                # create a new fuseinfo for this group
                sub_extents = []
                sub_duals = []
                sub_fuseinfos = []

                for ax in axes_groups[g]:
                    sub_extents.append(old_shape_block[ax])
                    sub_duals.append(old_duals[ax])
                    sub_fuseinfos.append(old_fuseinfos[ax])

                finfo = FuseInfo(
                    subkeys=subkeys[g],
                    extents=sub_extents,
                    duals=sub_duals,
                    fuseinfos=sub_fuseinfos,
                )
                new_fuseinfos.append(finfo)

        for ax in axes_after:
            new_duals.append(self.duals[ax])
            new_fuseinfos.append(self.fuseinfos[ax])

        if inplace:
            self.fkeys = new_fkeys
            self.fblock = new_blocks
            self.duals = new_duals
            self.fuseinfos = new_fuseinfos
        else:
            return self.__class__(
                fkeys=new_fkeys,
                fblock=new_blocks,
                duals=new_duals,
                fuseinfos=new_fuseinfos,
            )

    def unfuse(self, ax, inplace=False):
        """Unfuse the ``axis`` index, which must carry subindex information,
        likely generated automatically from a fusing operation.

        Parameters
        ----------
        axis : int
            The axis to unfuse. It must have fuse information
            (`.fuseinfos[ax]`).
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianArrayFlat
        """
        from einops import rearrange, repeat

        new = self if inplace else self.copy()

        fi = new.fuseinfos[ax]
        if fi is None:
            raise ValueError(f"Axis {ax} is not fused in this array.")

        axs_rem = tuple(range(ax)) + tuple(range(ax + 1, self.ndim))
        new.sort_stack((ax, *axs_rem), True)

        # keys coming from remaining axes
        ka = repeat(
            new.fkeys, "(Bf B) s -> (Bf B x) s", Bf=fi.ncharge, x=fi.nsectors
        )[:, axs_rem]

        # keys coming from unfused axis
        kb = fi.subkeys
        kb = repeat(kb, "B Bu s -> (B x Bu) s", x=new.num_blocks // fi.ncharge)

        # concatenate into the full new keys!
        new_keys = ar.do("concatenate", (ka[:, :ax], kb, ka[:, ax:]), axis=-1)

        # now we need to unfuse the actual blocks an example pattern:
        #     B p0 ( Bu u0 u1 u2 ) p2 p3 -> (B Bu) p0 u0 u1 u2 p2 p3
        # i.e. we unfuse the current axis, and shift its internal sector index
        # (`Bu`) into the total sector index
        pattern = ["B "]
        rhs = ["(B Bu) "]
        sizes = {}
        for i in range(ax):
            pattern.append(f"p{i} ")
            rhs.append(f"p{i} ")
        pattern.append("( Bu ")
        for g, sz in enumerate(fi.extents):
            pattern.append(f"u{g} ")
            rhs.append(f"u{g} ")
            sizes[f"u{g}"] = sz
        pattern.append(") ")
        for i in range(ax + 1, new.ndim):
            pattern.append(f"p{i} ")
            rhs.append(f"p{i} ")
        pattern.append("-> ")
        pattern.extend(rhs)
        pattern = "".join(pattern)

        # perform the unfuse!
        new_blocks = rearrange(new.fblock, pattern, **sizes)

        new_duals = (*new.duals[:ax], *fi.duals, *new.duals[ax + 1 :])
        new_fuseinfos = (
            *new.fuseinfos[:ax],
            *fi.fuseinfos,
            *new.fuseinfos[ax + 1 :],
        )

        new.fkeys = new_keys
        new.fblock = new_blocks
        new.duals = new_duals
        new.fuseinfos = new_fuseinfos

        return new

    def conj(self, inplace=False):
        """Return the complex conjugate of this block array, including the
        indices."""
        new_fkeys = self.fkeys
        new_fblock = ar.do("conj", self.fblock, like=self.backend)
        new_duals = tuple(not d for d in self.duals)
        new_flatinfos = tuple(
            fi.conj() if fi is not None else None for fi in self.fuseinfos
        )
        if inplace:
            self.fkeys = new_fkeys
            self.fblock = new_fblock
            self.duals = new_duals
            self.fuseinfos = new_flatinfos
            return self
        else:
            return self.__class__(
                fkeys=new_fkeys,
                fblock=new_fblock,
                duals=new_duals,
                fuseinfos=new_flatinfos,
            )

    def transpose(self, axes=None, inplace=False):
        """Transpose this flat abelian array."""
        if axes is None:
            # reverse the axes
            axes = tuple(range(self.ndim - 1, -1, -1))

        axes = tuple(map(int, axes))

        # just swap columbs of sectors
        new_fkeys = self.fkeys[:, axes]

        # transpose block as usual, but broadcast block axis
        new_fblock = ar.do(
            "transpose",
            self.fblock,
            (0, *(ax + 1 for ax in axes)),
            like=self.backend,
        )

        new_duals = tuple(self.duals[ax] for ax in axes)
        new_fuseinfos = tuple(self.fuseinfos[ax] for ax in axes)

        if inplace:
            self.fkeys = new_fkeys
            self.fblock = new_fblock
            self.duals = new_duals
            self.fuseinfos = new_fuseinfos
            return self
        else:
            return self.__class__(
                fkeys=new_fkeys,
                fblock=new_fblock,
                duals=new_duals,
                fuseinfos=new_fuseinfos,
            )

    def trace(self):
        raise NotImplementedError()

    def multiply_diagonal(self, v, axis, inplace=False):
        raise NotImplementedError()

    def align_axes(self, other, axes):
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
    order = 2


@functools.cache
def get_zn_array_flat_cls(N):
    """Get a block array class with ZN symmetry."""
    if N == 2:
        return Z2ArrayFlat

    return type(
        f"Z{N}ArrayFlat",
        (AbelianArrayFlat,),
        {"order": N},
    )
