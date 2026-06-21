"""Helper functions for building local spin-1/2 operators from a symbolic
description, without any external dependencies. Unlike the fermionic case there
are no 'internal' signs to compute, so each named operator is simply given its
2x2 matrix representation and terms are combined via tensor (Kronecker)
products.
"""

import functools

import autoray as ar


@functools.cache
def _spin_matrices():
    """The 2x2 matrix representations of the supported single-site spin-1/2
    operators, as a dict of numpy arrays. Cached so they are built only once.
    """
    import numpy as np

    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.array([[1.0, 0.0], [0.0, -1.0]])

    return {
        # identity
        "I": np.array([[1.0, 0.0], [0.0, 1.0]]),
        # pauli operators
        "x": x,
        "y": y,
        "z": z,
        # spin-1/2 operators
        "sx": x / 2,
        "sy": y / 2,
        "sz": z / 2,
        # raising and lowering operators
        "+": np.array([[0.0, 1.0], [0.0, 0.0]]),
        "-": np.array([[0.0, 0.0], [1.0, 0.0]]),
    }


def get_spin_operator(label, like="numpy"):
    """Get the 2x2 matrix representation of a single-site spin-1/2 operator.

    Parameters
    ----------
    label : str
        The operator to get, one of "I", "x", "y", "z", "sx", "sy", "sz", "+"
        or "-".
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array
        The 2x2 matrix representation.
    """
    matrices = _spin_matrices()
    try:
        matrix = matrices[label]
    except KeyError:
        raise ValueError(
            f"Unknown spin operator {label!r}. Valid operators are "
            f"{sorted(matrices)}."
        )
    return ar.do("array", matrix, like=like)


def get_spinhalf_charge_indexmap(symmetry):
    """Get a mapping of linear index to charge sector for a spin-1/2 site.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", a "ZN" group, or "U1".

    Returns
    -------
    list[hashable]
    """
    if symmetry == "U1" or (
        symmetry.startswith("Z") and symmetry[1:].isdigit()
    ):
        return [0, 1]
    else:
        raise ValueError(f"Invalid symmetry: {symmetry}")


def _spinhalf_index_maps(symmetry, nsites):
    """Per-site spin-1/2 index maps, or None when ``symmetry`` is None."""
    if symmetry is None:
        return None
    return [get_spinhalf_charge_indexmap(symmetry)] * nsites


def build_local_spin_dense(terms, like="numpy"):
    """Compute the dense matrix of a local spin-1/2 operator from a symbolic
    description.

    Parameters
    ----------
    terms : Sequence[tuple[float, tuple[str, ...]]]
        The terms in the operator, each a tuple of a coefficient and a tuple of
        single-site operator labels (see `get_spin_operator`), one per site.
        All terms should act on the same number of sites.
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array
        The dense operator, of shape ``(2,) * (2 * nsites)`` with axes ordered
        as ``(ket_0, ..., ket_{n-1}, bra_0, ..., bra_{n-1})``.

    Examples
    --------
    The transverse field Ising local term::

        build_local_spin_dense(
            [(-1.0, ("x", "x")), (-3.0, ("z", "I")), (-3.0, ("I", "z"))]
        )

    """
    import numpy as np

    matrices = _spin_matrices()
    nsites = len(terms[0][1])

    hij = None
    for coeff, ops in terms:
        if coeff == 0.0:
            continue
        matrix = matrices[ops[0]]
        for op in ops[1:]:
            matrix = np.kron(matrix, matrices[op])
        contrib = coeff * matrix
        hij = contrib if hij is None else hij + contrib

    if hij is None:
        # no non-zero terms
        hij = np.zeros((2**nsites, 2**nsites))

    # drop a negligible imaginary part (e.g. from y . y -> real)
    if np.iscomplexobj(hij) and not np.any(np.abs(hij.imag) > 1e-12):
        hij = hij.real

    hij = hij.reshape((2,) * (2 * nsites))

    return ar.do("array", hij, like=like)


def build_local_spin_array(
    terms,
    symmetry,
    index_maps=None,
    charge=None,
    like="numpy",
    flat=False,
):
    """Compute a local spin-1/2 operator as an `AbelianArray`.

    Parameters
    ----------
    terms : Sequence[tuple[float, tuple[str, ...]]]
        The terms in the operator, each a tuple of a coefficient and a tuple of
        single-site operator labels (see `get_spin_operator`), one per site.
    symmetry : str or None
        The symmetry of the operator. Either "Z2", a "ZN" group, or "U1". If
        None, the raw dense array is returned instead of a symmetric array.
    index_maps : Sequence[Sequence[hashable]], optional
        For each site, the sequence mapping linear index to charge sector.
        Required unless ``symmetry`` is None.
    charge : hashable, optional
        The total charge of the array. If not given it is taken as the identity
        / zero element, suitable for charge-conserving operators.
    like : str, optional
        The backend to use, by default "numpy".
    flat : bool, optional
        Whether to return a flat array, by default False.

    Returns
    -------
    AbelianArray or AbelianArrayFlat or array
        The local operator, as a dense array if ``symmetry`` is None.
    """
    from .utils import from_dense

    dense = build_local_spin_dense(terms, like=like)
    if symmetry is None:
        return dense

    nsites = len(terms[0][1])
    duals = [False] * nsites + [True] * nsites

    return from_dense(
        dense,
        symmetry=symmetry,
        index_maps=list(index_maps) * 2,
        duals=duals,
        fermionic=False,
        charge=charge,
        flat=flat,
    )


def tfim_local_array(
    symmetry,
    jx=-1.0,
    hz=-3.0,
    coordinations=(1, 1),
    like="numpy",
    flat=False,
):
    """Build an abelian symmetric local operator for the transverse field
    Ising model::

        H = jx * sum_ij X_i X_j + hz * sum_i Z_i

    Note that its rotated into the x-basis so that the Z2 symmetry is manifest.

    Parameters
    ----------
    symmetry : str or None
        The symmetry of the model. Should be "Z2", or None to return the raw
        dense array.
    jx : float
        The coupling strength for the X-X interactions, by default -1.0.
    hz : float or tuple[float, float]
        The coupling strength for the Z interactions, by default -3.0. If a
        tuple is given it should contain the fields for the two sites.
    coordinations : tuple[int, int], optional
        The coordinations of the two sites, by default (1, 1). The fields
        are divided by these values to account for double counting.
    like : str, optional
        The backend to use, by default "numpy".
    flat : bool, optional
        Whether to return a flat array, by default False.

    Returns
    -------
    Z2Array or Z2ArrayFlat or array
        The local Hamiltonian term, dense if ``symmetry`` is None.
    """
    if symmetry is not None and symmetry != "Z2":
        raise ValueError(f"Symmetry {symmetry} not supported for TFIM.")

    try:
        ha, hb = hz
    except TypeError:
        ha = hb = hz

    terms = [
        (jx, ("x", "x")),
        (ha / coordinations[0], ("z", "I")),
        (hb / coordinations[1], ("I", "z")),
    ]

    return build_local_spin_array(
        terms,
        symmetry,
        index_maps=_spinhalf_index_maps(symmetry, 2),
        like=like,
        flat=flat,
    )


def heisenberg_local_array(
    symmetry,
    j=1.0,
    b=0.0,
    coordinations=(1, 1),
    like="numpy",
    flat=False,
):
    """Build an abelian symmetric local operator for the Heisenberg model::

        H = sum_ij (jx Sx_i Sx_j + jy Sy_i Sy_j + jz Sz_i Sz_j)
            - sum_i bz Sz_i

    where the spin operators are the spin-1/2 operators (eigenvalues +/- 1/2).

    Parameters
    ----------
    symmetry : str or None
        The symmetry of the model. Either "Z2" or "U1", or None to return the
        raw dense array. Note "U1" requires the XY couplings to be equal
        (``jx == jy``), so that total magnetization is conserved.
    j : float or tuple[float, float, float], optional
        The coupling strength, by default 1.0. If a tuple is given it should
        contain the ``(jx, jy, jz)`` couplings.
    b : float or tuple[float, float], optional
        The magnetic field along the z-axis, by default 0.0. A scalar is
        applied uniformly to both sites, while a pair ``(ba, bb)`` sets the two
        sites' fields independently. Only z-fields are supported as transverse
        fields would not conserve the symmetry.
    coordinations : tuple[int, int], optional
        The coordinations of the two sites, by default (1, 1). The fields
        are divided by these values to account for double counting.
    like : str, optional
        The backend to use, by default "numpy".
    flat : bool, optional
        Whether to return a flat array, by default False.

    Returns
    -------
    AbelianArray or AbelianArrayFlat or array
        The local Hamiltonian term, dense if ``symmetry`` is None.
    """
    try:
        jx, jy, jz = j
    except TypeError:
        jx = jy = jz = j

    try:
        ba, bb = b
    except TypeError:
        ba = bb = b

    terms = [
        (jx, ("sx", "sx")),
        (jy, ("sy", "sy")),
        (jz, ("sz", "sz")),
    ]

    # per-site z-fields, divided by coordination to avoid double counting
    for site, (bz, ops) in enumerate([(ba, ("sz", "I")), (bb, ("I", "sz"))]):
        coeff = -bz / coordinations[site]
        if coeff != 0.0:
            terms.append((coeff, ops))

    return build_local_spin_array(
        terms,
        symmetry,
        index_maps=_spinhalf_index_maps(symmetry, 2),
        like=like,
        flat=flat,
    )


def spin_operator_local_array(symmetry, op="z", like="numpy", flat=False):
    """Build a single-site abelian symmetric spin-1/2 operator.

    Parameters
    ----------
    symmetry : str or None
        The symmetry of the model. Either "Z2", a "ZN" group, or "U1", or None
        to return the raw dense array.
    op : str, optional
        The operator to build, by default "z". One of "I", "x", "y", "z",
        "sx", "sy", "sz", "+" or "-". Note charge-changing operators ("+",
        "-", and for "U1" also "x", "y") yield an array with non-identity total
        charge.
    like : str, optional
        The backend to use, by default "numpy".
    flat : bool, optional
        Whether to return a flat array, by default False.

    Returns
    -------
    AbelianArray or AbelianArrayFlat or array
        The single-site operator, dense if ``symmetry`` is None.
    """
    from .symmetries import get_symmetry

    terms = [(1.0, (op,))]
    if symmetry is None:
        return build_local_spin_array(terms, None, like=like)

    indexmap = get_spinhalf_charge_indexmap(symmetry)

    # derive the total charge from which sectors the operator connects
    matrix = get_spin_operator(op, like="numpy")
    sym = get_symmetry(symmetry)
    charges = set()
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            if abs(matrix[r, c]) > 1e-12:
                charges.add(
                    sym.combine(
                        sym.sign(indexmap[r], False),
                        sym.sign(indexmap[c], True),
                    )
                )

    if len(charges) > 1:
        raise ValueError(
            f"Operator {op!r} is not well defined under symmetry "
            f"{symmetry!r}: it connects sectors with differing total charge "
            f"{sorted(charges)}."
        )
    charge = charges.pop() if charges else sym.combine()

    return build_local_spin_array(
        terms,
        symmetry,
        index_maps=[indexmap],
        charge=charge,
        like=like,
        flat=flat,
    )
