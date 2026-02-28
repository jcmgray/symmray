"""Common linear algebra utilities shared across backends."""

# absorb mode constants (matching quimb convention)
get_U_s_VH = None  # 'full'
get_s = 2  # 'svals'
get_Usq = -12  # 'lsqrt'
get_VH = -11  # 'rorthog'
get_Us = -10  # 'lfactor'
get_Us_VH = -1  # absorb 'left'
get_Usq_sqVH = 0  # absorb 'both'
get_U_sVH = 1  # absorb 'right'
get_U = 10  # 'lorthog'
get_sVH = 11  # 'rfactor'
get_sqVH = 12  # 'rsqrt'
_ABSORB_MAP = {}
for mode, aliases in [
    (None, ["U,s,VH"]),
    (get_s, ["s"]),  # 2
    (get_Usq, ["lsqrt"]),  # -12
    (get_VH, ["VH", "rorthog"]),  # -11
    (get_Us, ["Us", "lfactor"]),  # -10
    (get_Us_VH, ["Us,VH", "left"]),  # -1
    (get_Usq_sqVH, ["Usq,sqVH", "both"]),  # 0
    (get_U_sVH, ["U,sVH", "right"]),  # 1
    (get_U, ["U", "lorthog"]),  # 10
    (get_sVH, ["sVH", "rfactor"]),  # 11
    (get_sqVH, ["sqVH", "rsqrt"]),  # 12
]:
    _ABSORB_MAP[mode] = mode
    for alias in aliases:
        _ABSORB_MAP[alias] = mode


def _do_absorb(U, s, VH, absorb):
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
        Absorption mode code.

    Returns
    -------
    U : SymmrayCommon or None
    s : VectorCommon or None
    VH : SymmrayCommon or None
    """
    if absorb is None:  # get_U_s_VH - 'full'
        return U, s, VH
    if absorb == get_s:  # 'svals'
        return None, s, None
    if absorb == get_U:  # 'lorthog'
        return U, None, None
    if absorb == get_VH:  # 'rorthog'
        return None, None, VH
    if absorb == get_Us_VH:  # 'left'
        U.multiply_diagonal(s, axis=1, inplace=True)
        return U, None, VH
    if absorb == get_U_sVH:  # 'right'
        VH.multiply_diagonal(s, axis=0, inplace=True)
        return U, None, VH
    if absorb == get_Usq_sqVH:  # 'both'
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        return U, None, VH
    if absorb == get_Us:  # 'lfactor'
        U.multiply_diagonal(s, axis=1, inplace=True)
        return U, None, None
    if absorb == get_sVH:  # 'rfactor'
        VH.multiply_diagonal(s, axis=0, inplace=True)
        return None, None, VH
    if absorb == get_Usq:  # 'lsqrt'
        s_sqrt = s.sqrt()
        U.multiply_diagonal(s_sqrt, axis=1, inplace=True)
        return U, None, None
    if absorb == get_sqVH:  # 'rsqrt'
        s_sqrt = s.sqrt()
        VH.multiply_diagonal(s_sqrt, axis=0, inplace=True)
        return None, None, VH
    raise ValueError(f"Invalid absorb mode: {absorb}")
