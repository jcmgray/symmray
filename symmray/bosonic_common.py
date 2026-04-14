"""Common methods for any 'bosonic' (non-fermionic) arrays."""


class BosonicCommon:
    """Mixin for bosonic (non-fermionic) arrays."""

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
