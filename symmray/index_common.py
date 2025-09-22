"""General interface for index objects."""


class Index:
    @property
    def dual(self) -> bool:
        """Whether the index flows 'outwards' / (+ve) / ket-like = ``False`` or
        'inwards' / (-ve) / bra-like= ``True``. The charge sign is given by
        ``(-1) ** dual``.
        """
        return self._dual

    @property
    def subinfo(self) -> "SubInfo":
        """Information about the subindices of this index and their extents if
        this index was formed from fusing.
        """
        return self._subinfo


class SubInfo:
    @property
    def indices(self) -> tuple[Index, ...]:
        """The indices that were fused to make this index."""
        return self._indices
