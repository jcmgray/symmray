"""Methods that apply to all Abelian symmetric arrays, regardless of
backend.
"""

import functools

from autoray.lazy.core import find_full_reshape

from .symmetries import Symmetry, get_symmetry


@functools.lru_cache(maxsize=2**15)
def calc_reshape_args(shape, newshape, subshapes):
    """Given a current block sparse shape ``shape`` a target shape ``newshape``
    and current sub index sizes ``subshapes`` (i.e. previously fused dimensions)
    compute the sequence of axes to unfuse, fuse and expand to reshape the
    array.

    Parameters
    ----------
    shape : tuple[int]
        The current shape of the array.
    newshape : tuple[int]
        The target shape of the array.
    subshapes : tuple[None or tuple[int]]
        The sizes of the subindices that were previously fused.

    Returns
    -------
    axs_unfuse : tuple[int]
        The axes to unfuse.
    axs_fuse : tuple[tuple[tuple[int]]]
        The axes (after unfusing) to fuse, grouped by contiguous groups.
    axs_expand : tuple[int]
        The axes (after unfusing and fusing) to expand.
    """
    # tracks position in input shape
    i = 0
    # tracks position in output shape
    j = 0
    # tracks position in post-fuse / pre-expand shape
    k = 0

    ndim_old = len(shape)
    ndim_new = len(newshape)

    term = []  # dnyamically updated labelled dimensions
    axs_squeeze = []
    unfuse_sizes = {}
    fuse_sizes = {}
    axs_expand = []
    any_singleton = False
    any_fused = False

    while i < ndim_old and j < ndim_new:
        di = shape[i]
        dj = newshape[j]

        if (subshapes[i] is not None) and (
            subshapes[i] == newshape[j : j + len(subshapes[i])]
        ):
            # unfuse, check first
            label = f"u{len(unfuse_sizes)}"
            s = 0
            for ds in subshapes[i]:
                dj = newshape[j]
                if ds != dj:
                    raise ValueError("Shape mismatch for unfuse.")
                s += 1
                j += 1
                k += 1
            unfuse_sizes[label] = s
            term.append(label)
            i += 1
        elif di == dj:
            # output dimension already
            term.append("o")
            i += 1
            j += 1
            k += 1
        elif di == 1:
            # have to handle squeezed dimensions after unfusing
            term.append("s")
            axs_squeeze.append(i)
            any_singleton = True
            i += 1
        elif dj == 1:
            # record expansion location relative to *post* fuse shape
            axs_expand.append(k)
            j += 1
        elif di < dj:
            # need to fuse
            label = f"g{len(fuse_sizes)}"
            term.append(label)
            s = 1
            i += 1
            while di < dj:
                di *= shape[i]
                term.append(label)
                i += 1
                s += 1
            if di != dj:
                raise ValueError("Shape mismatch for fuse.")
            fuse_sizes[label] = s
            any_fused = True
            j += 1
            k += 1
        else:
            raise ValueError("Shape mismatch.")

    # check trailing dimensions, which should be size 1
    for i in range(i, ndim_old):
        any_singleton = True
        term.append("s")
    for j in range(j, ndim_new):
        axs_expand.append(k)

    # first we handle unfusings
    axs_unfuse = []
    for label, s in unfuse_sizes.items():
        ax = term.index(label)
        axs_unfuse.append(ax)
        term = term[:ax] + ["o"] * s + term[ax + 1 :]

    # handle squeezes by converting them into fuse groups
    if any_singleton:
        i = 0
        label = term[i]
        if label == "s":
            # if we have squeeze axes on left, we have to group into right
            while label == "s":
                i += 1
                label = term[i]

            if label[0] == "g":
                # adjacent to existing group
                g = label
            elif label == "o":
                # or new group
                g = f"g{len(fuse_sizes)}"
                term[i] = g
                fuse_sizes[g] = 1

            # mark all axs up to this point
            for j in range(0, i):
                fuse_sizes[g] += 1
                term[j] = g

        # process rest of term, now preferring grouping into left
        i += 1
        while i < len(term):
            label = term[i]
            if label == "s":
                left = term[i - 1]
                if left[0] == "g":
                    g = left
                elif left == "o":
                    g = f"g{len(fuse_sizes)}"
                    term[i - 1] = g
                    fuse_sizes[g] = 1

                # update any right block of squeeze axs to g
                while label == "s":
                    term[i] = g
                    fuse_sizes[g] += 1
                    i += 1
                    if i == len(term):
                        break
                    label = term[i]
            i += 1

    # now we handle fusing
    axs_fuse = []
    if any_fused or any_singleton:
        # complexity here is we want to simulteneously fuse adjacent groups for
        # efficiency, but also need to handle non-adjacent groups
        current_groups = []
        i = 0
        while i < len(term):
            label = term[i]
            if label not in fuse_sizes:
                if current_groups:
                    # start of groups
                    i0 = i - sum(map(len, current_groups))
                    ng = len(current_groups)
                    term = term[:i0] + ["o"] * ng + term[i:]
                    axs_fuse.append(tuple(current_groups))
                    current_groups = []
                    # rewind to end of new group(s)
                    i = i0 + ng
                else:
                    i += 1
                continue

            s = fuse_sizes[label]
            current_groups.append(tuple(range(i, i + s)))
            i += s
        if current_groups:
            axs_fuse.append(tuple(current_groups))

    # handle expansion
    axs_expand.reverse()

    return tuple(axs_unfuse), tuple(axs_fuse), tuple(axs_expand)


class AbelianCommon:
    """Common base class for arrays with Abelian symmetry."""

    @property
    def symmetry(self) -> Symmetry:
        """The symmetry object of the array."""
        return self._symmetry

    @classmethod
    def get_class_symmetry(cls, symmetry=None) -> Symmetry:
        if symmetry is None:
            if cls.static_symmetry is None:
                # symmetry must be given if not static
                raise ValueError("Symmetry must be given.")
            symmetry = cls.static_symmetry
        elif cls.static_symmetry and symmetry != cls.static_symmetry:
            raise ValueError("Cannot override static symmetry of class.")

        return get_symmetry(symmetry)

    @property
    def signature(self):
        return "".join("-" if f else "+" for f in self.duals)

    @property
    def T(self):
        """The transpose of the block array."""
        return self.transpose()

    def dagger(self, inplace=False):
        """Return the adjoint of this block array."""
        return self.conj(inplace=inplace).transpose(inplace=True)

    @property
    def H(self):
        return self.dagger()

    def fuse(
        self,
        *axes_groups,
        expand_empty=True,
        inplace=False,
        **kwargs,
    ):
        """Fuse the given group or groups of axes. The new fused axes will be
        inserted at the minimum index of any fused axis (even if it is not in
        the first group). For example, ``x.fuse([5, 3], [7, 2, 6])`` will
        produce an array with axes like::

            groups inserted at axis 2, removed beyond that.
                   ......<--
            (0, 1, g0, g1, 4, 8, ...)
                   |   |
                   |   g1=(7, 2, 6)
                   g0=(5, 3)

        The fused axes will carry subindex information, which can be used to
        automatically unfuse them back into their original components.
        Depending on `expand_empty`, any empty groups can be expanded to new
        singlet dimensions, or simply ignored.

        Parameters
        ----------
        axes_groups : Sequence[Sequence[int]]
            The axes to fuse. Each group of axes will be fused into a single
            axis.
        expand_empty : bool, optional
            Whether to expand empty groups into new axes.
        mode : "auto", "insert", "concat", optional
            The method to use for fusing. `"insert"` creates the new fused
            blocks and insert the subblocks inplace. `"concat"` recursively
            concatenates the subblocks, which can be slightly slower but is
            more compatible with e.g. autodiff. `"auto"` will use `"insert"` if
            the backend is numpy, otherwise `"concat"`.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.
        kwargs : dict, optional
            Additional keyword arguments to pass to the core fusing method.

        Returns
        -------
        AbelianCommon
        """
        # handle empty groups and ensure hashable
        _axes_groups = []
        _axes_expand = []
        for ax, group in enumerate(axes_groups):
            if group:
                _axes_groups.append(tuple(group))
            else:
                _axes_expand.append(ax)

        if _axes_groups:
            xf = self._fuse_core(*_axes_groups, inplace=inplace, **kwargs)
        else:
            xf = self if inplace else self.copy()

        if expand_empty and _axes_expand:
            g0 = min(g for groups in _axes_groups for g in groups)
            for ax in _axes_expand:
                xf.expand_dims(g0 + ax, inplace=True)

        return xf

    def unfuse_all(self, inplace=False):
        """Unfuse all indices that carry subindex information, likely from a
        fusing operation.

        Parameters
        ----------
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianCommon
        """
        new = self if inplace else self.copy()
        for ax in reversed(range(self.ndim)):
            if new.is_fused(ax):
                new.unfuse(ax, inplace=True)
        return new

    def reshape(self, newshape, inplace=False):
        """Reshape this abelian array to ``newshape``, assuming it can be done
        by any mix of fusing, unfusing, and expanding new axes.

        Restrictions and complications vs normal array reshaping arise from the
        fact that

            A) only previously fused axes can be unfused, and their total size
               may not be the product of the individual sizes due to sparsity.
            B) the indices carry information beyond size and how they are
               grouped potentially matters, relevant for singleton dimensions.

        Accordingly the approach here is as follows:

            1. Unfuse any axes that match the new shape.

            2. If there are singleton dimensions that don't appear in the new
               shape, (i.e. are being 'squeezed') these are grouped with the
               axis to the their left to then be fused. If they are already
               left-most, they are grouped with the right.

            3. Fuse any groups of axes required to match the new shape.
               Adjacent groups are fused simultaneously for efficiency.

            4. Expand new axes required to match singlet dimensions in the new
               shape. By default these will have zero charge and dual-ness
               iherited from whichever axis is to their left, or right if they
               are the left-most axis already.

        To avoid the effective grouping of 'squeezed' axes you can explicitly
        squeeze them before reshaping. Similarly use ``expand_dims`` to add
        new axes with specific charges and dual-ness.

        Parameters
        ----------
        newshape : tuple[int]
            The new shape to reshape to.
        inplace : bool, optional
            Whether to perform the operation inplace or return a new array.

        Returns
        -------
        AbelianCommon

        See Also
        --------
        fuse, unfuse, squeeze, expand_dims
        """
        x = self if inplace else self.copy()

        if not isinstance(newshape, tuple):
            newshape = tuple(newshape)
        newshape = find_full_reshape(newshape, self.size)

        subshapes = tuple(ix.subshape for ix in x.indices)

        # cached parsing of reshape arguments
        axs_unfuse, axs_fuse_groupings, axs_expand = calc_reshape_args(
            x.shape, newshape, subshapes
        )

        for ax in axs_unfuse:
            x.unfuse(ax, inplace=True)
        for grouping in axs_fuse_groupings:
            x.fuse(*grouping, inplace=True)
        for ax in axs_expand:
            x.expand_dims(ax, inplace=True)

        return x

    def __str__(self):
        lines = [
            (
                f"{self.__class__.__name__}(ndim={self.ndim}, "
                f"charge={self.charge}, indices=["
            )
        ]
        for i in range(self.ndim):
            lines.extend(
                f"    {line}" for line in str(self.indices[i]).split("\n")
            )
        lines.append(
            (
                f"], num_blocks={self.num_blocks}, backend={self.backend}, "
                f"dtype={self.dtype})"
            )
        )
        return "\n".join(lines)

    def __repr__(self):
        if self.static_symmetry is not None:
            c = f"{self.__class__.__name__}("
        else:
            c = f"{self.__class__.__name__}{self.symmetry}("

        return "".join(
            [
                c,
                (
                    f"shape~{self.shape}:[{self.signature}]"
                    if self.indices
                    else f"{self.get_any_array()}"
                ),
                f", charge={self.charge}",
                f", num_blocks={self.num_blocks})",
            ]
        )
