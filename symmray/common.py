"""Common interfaces for all symmray array objects."""

_symmray_namespace = None


class SymmrayCommon:
    """Common functionality for all symmray array like objects."""

    def __array_namespace__(self, api_version=None):
        """Return the namespace for the symmray module."""
        global _symmray_namespace
        if _symmray_namespace is None:
            import symmray

            _symmray_namespace = symmray
        return _symmray_namespace
