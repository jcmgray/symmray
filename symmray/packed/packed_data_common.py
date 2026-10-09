"""Common interface methods for arrays with packed backend storage: every block
stored C contiguous, end to end, in one flat data buffer.
"""

import autoray as ar


class PackedCommon:
    """Mixin class for arrays with packed storage, numpy only for now. Copies
    and cached plans share the structure arrays, so never modify them inplace.
    The structure is

    - ``_data``: 1D buffer holding every block, in sector order.
    - ``_sectors``: integer array ``(num_blocks, ndim)`` of the charge id of
      each block on each axis, rows unique and lexicographically sorted.
    - ``_shapes``: integer array ``(num_blocks, ndim)`` of block shapes.
    - ``_offsets``: integer array ``(num_blocks + 1,)`` of where each block
      starts in ``_data``.
    """

    __slots__ = ("_data", "_offsets", "_sectors", "_shapes", "_skey")

    def _new_with_packedcommon(self, data, sectors, shapes, offsets):
        new = self.__new__(self.__class__)
        new._data = data
        new._sectors = sectors
        new._shapes = shapes
        new._offsets = offsets
        new._skey = None
        return new

    def _copy_packedcommon(self):
        new = self.__new__(self.__class__)
        new._data = self._data
        new._sectors = self._sectors
        new._shapes = self._shapes
        new._offsets = self._offsets
        new._skey = self._skey
        return new

    def _modify_packedcommon(
        self, data=None, sectors=None, shapes=None, offsets=None
    ):
        if data is not None:
            self._data = data
        if sectors is not None:
            self._sectors = sectors
            self._skey = None
        if shapes is not None:
            self._shapes = shapes
        if offsets is not None:
            self._offsets = offsets
        return self

    @property
    def data(self):
        """The flat buffer holding every block."""
        return self._data

    @property
    def sector_ids(self):
        """The charge id of each block on each axis, as an integer array
        ``(num_blocks, ndim)``.
        """
        return self._sectors

    @property
    def num_blocks(self):
        """The number of blocks."""
        return self._sectors.shape[0]

    @property
    def dtype(self):
        """The dtype name of the data."""
        return ar.get_dtype_name(self._data)

    @property
    def backend(self):
        """The backend of the data, always numpy for now."""
        return ar.infer_backend(self._data)

    def get_any_array(self):
        """Get the data buffer, e.g. to check dtype and backend."""
        return self._data

    def get_namespace(self):
        """Get the array namespace for the data."""
        return ar.get_namespace(self._data)

    def get_params(self):
        """Get the underlying data as a pytree."""
        return {"data": self._data}

    def set_params(self, params):
        """Set the underlying data from a pytree. The data must be a numpy
        array.
        """

        data = params["data"]
        if ar.infer_backend(data) != "numpy":
            raise TypeError(
                "The packed backend only supports numpy data, "
                f"got {ar.infer_backend(data)}."
            )
        self._data = data

    def apply_to_arrays(self, fn):
        """Apply ``fn`` inplace to all the data at once. ``fn`` should be
        elementwise, use `apply_to_blocks` otherwise.
        """
        self._data = fn(self._data)

    def apply_to_blocks(self, fn):
        """Apply ``fn`` inplace to the array of each block. ``fn`` need not be
        elementwise, e.g. a matrix exponential.
        """
        import numpy as np

        offsets = self._offsets.tolist()
        shapes = self._shapes.tolist()
        arrays = [
            np.asarray(fn(self._data[offsets[i] : offsets[i + 1]].reshape(s)))
            for i, s in enumerate(shapes)
        ]
        if arrays:
            self._data = np.concatenate([x.reshape(-1) for x in arrays])
        else:
            self._data = fn(self._data)

    def _do_reduction_blockwise(self, fn):
        if isinstance(fn, str):
            fn = ar.get_lib_fn(self.backend, fn)
        return fn(self._data)

    def norm(self):
        """Get the frobenius norm of the array."""
        return ar.do("linalg.norm", self._data)

    def is_zero(self, tol=1e-12):
        """Check if all elements are zero up to a tolerance."""
        return bool(ar.do("allclose", self._data, 0.0, atol=tol))

    def _item_blockwise(self):
        return self._data.item()
