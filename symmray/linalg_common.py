"""Common linear algebra utilities shared across backends."""


class Absorb:
    """Absorb mode constants and parsing for SVD-like decompositions.

    Attributes give canonical integer codes for each mode. Use
    ``Absorb.parse`` to normalize user-facing string or integer aliases
    to the canonical code.

    Modes
    -----
    U_s_VH : None
        Return all three factors unmodified ('full').
    s : 2
        Return only the singular values ('svals').
    Usq : -12
        Absorb sqrt(s) into U, return (U√s, None, None) ('lsqrt').
    VH : -11
        Return only VH ('rorthog').
    Us : -10
        Absorb s into U, return (Us, None, None) ('lfactor').
    Us_VH : -1
        Absorb s into U, return (Us, None, VH) ('left').
    Usq_sqVH : 0
        Absorb sqrt(s) into both, return (U√s, None, √sVH) ('both').
    U_sVH : 1
        Absorb s into VH, return (U, None, sVH) ('right').
    U : 10
        Return only U ('lorthog').
    sVH : 11
        Absorb s into VH, return (None, None, sVH) ('rfactor').
    sqVH : 12
        Absorb sqrt(s) into VH, return (None, None, √sVH) ('rsqrt').
    """

    U_s_VH = None  # 'full'
    s = 2  # 'svals'
    Usq = -12  # 'lsqrt'
    VH = -11  # 'rorthog'
    Us = -10  # 'lfactor'
    Us_VH = -1  # 'left'
    Usq_sqVH = 0  # 'both'
    U_sVH = 1  # 'right'
    U = 10  # 'lorthog'
    sVH = 11  # 'rfactor'
    sqVH = 12  # 'rsqrt'

    _map = {}

    @classmethod
    def parse(cls, absorb):
        """Normalize a user-facing absorb value to a canonical mode code.

        Parameters
        ----------
        absorb : int, str, or None
            Any valid absorb specification: a canonical integer code,
            a string alias (e.g. ``'left'``, ``'both'``, ``'rfactor'``),
            or ``None`` for full (no absorption).

        Returns
        -------
        int or None
            The canonical absorb mode code.

        Raises
        ------
        KeyError
            If ``absorb`` is not a recognized mode or alias.
        """
        return cls._map[absorb]


# populate the map once at import time
for _mode, _aliases in [
    (Absorb.U_s_VH, ["U,s,VH"]),
    (Absorb.s, ["s"]),
    (Absorb.Usq, ["lsqrt"]),
    (Absorb.VH, ["VH", "rorthog"]),
    (Absorb.Us, ["Us", "lfactor"]),
    (Absorb.Us_VH, ["Us,VH", "left"]),
    (Absorb.Usq_sqVH, ["Usq,sqVH", "both"]),
    (Absorb.U_sVH, ["U,sVH", "right"]),
    (Absorb.U, ["U", "lorthog"]),
    (Absorb.sVH, ["sVH", "rfactor"]),
    (Absorb.sqVH, ["sqVH", "rsqrt"]),
]:
    Absorb._map[_mode] = _mode
    for _alias in _aliases:
        Absorb._map[_alias] = _mode
del _mode, _aliases, _alias  # noqa: F821


def absorb_svd_result(U, s, VH, absorb):
    """Apply absorption of singular values into U and/or VH.

    Works on any symmray array objects that support ``multiply_diagonal``
    and vectors that support ``.sqrt()``.

    Parameters
    ----------
    U : SymmrayCommon
        Left singular vectors.
    s : VectorCommon
        Singular values.
    VH : SymmrayCommon
        Right singular vectors.
    absorb : int or None
        Absorption mode code (should already be parsed via ``Absorb.parse``).

    Returns
    -------
    U : SymmrayCommon or None
    s : VectorCommon or None
    VH : SymmrayCommon or None
    """
    if absorb is Absorb.U_s_VH:  # None - 'full'
        return U, s, VH
    if absorb == Absorb.s:
        return None, s, None
    if absorb == Absorb.U:
        return U, None, None
    if absorb == Absorb.VH:
        return None, None, VH
    if absorb == Absorb.Us_VH:
        U.multiply_diagonal(s, axis=1, inplace=True)
        return U, None, VH
    if absorb == Absorb.U_sVH:
        VH.multiply_diagonal(s, axis=0, inplace=True)
        return U, None, VH
    if absorb == Absorb.Usq_sqVH:
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        return U, None, VH
    if absorb == Absorb.Us:
        U.multiply_diagonal(s, axis=1, inplace=True)
        return U, None, None
    if absorb == Absorb.sVH:
        VH.multiply_diagonal(s, axis=0, inplace=True)
        return None, None, VH
    if absorb == Absorb.Usq:
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        return U, None, None
    if absorb == Absorb.sqVH:
        s_sqrt = s.sqrt()
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        return None, None, VH
    raise ValueError(f"Invalid absorb mode: {absorb}")
