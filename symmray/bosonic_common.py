"""Common methods for any 'bosonic' (non-fermionic) arrays."""


class BosonicCommon:
    """Mixin for bosonic (non-fermionic) arrays."""

    def _split(self, *args, **kwargs):
        return self._split_abelian(*args, **kwargs)
