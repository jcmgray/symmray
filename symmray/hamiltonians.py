"""Abelian symmetric and fermionic Hamiltonians in terms of local operators."""


def make_edge_factory(coeff):
    """Ensure `coeff` is a function that takes two sites and returns an edge
    coeff."""
    if isinstance(coeff, dict):

        def edge_factory(cooa, coob):
            try:
                return coeff[(cooa, coob)]
            except KeyError:
                return coeff[(coob, cooa)]

    elif callable(coeff):
        edge_factory = coeff

    else:

        def edge_factory(cooa, coob):
            # constant
            return coeff

    return edge_factory


def make_node_factory(coeff):
    """Ensure `coeff` is a function that takes a site and returns a node
    coeff."""
    if isinstance(coeff, dict):

        def node_factory(coo):
            return coeff[coo]

    elif callable(coeff):
        node_factory = coeff

    else:

        def node_factory(coo):
            # constant
            return coeff

    return node_factory


def ham_tfim_from_edges(
    symmetry,
    edges,
    jx=-1.0,
    hz=-3.0,
    flat=False,
):
    """Return a dict of local 2-body Hamiltonian abelian symmetric terms for
    the transverse field Ising model on the given lattice defined by `edges`::

        H = jx * sum_ij X_i X_j + hz * sum_i Z_i

    Note that its rotated into the x-basis so that the Z2 symmetry is manifest.

    Parameters
    ----------
    symmetry : str or None
        The symmetry of the model. Should be "Z2", or None to return raw dense
        arrays.
    edges : Sequence[tuple[Hashable, Hashable]]
        A list of edges representing the lattice, each edge is a tuple of two
        nodes, each node is some hashable label.
    jx : float, dict, or callable, optional
        The coupling strength for the X-X interactions, by default -1.0. If a
        dict is given it should map edges to values, if a callable it should
        take the two sites as input.
    hz : float, dict or callable, optional
        The coupling strength for the Z interactions, by default -3.0. If a
        dict is given it should map sites to values, if a callable it should
        take the site as input.
    flat : bool, optional
        Whether to return flat arrays, by default False.

    Returns
    -------
    dict[tuple[Hashable, Hashable], AbelianArray | AbelianArrayFlat]
        A dictionary mapping edges to local Hamiltonian terms.
    """
    from .spin_local_operators import tfim_local_array

    coordinations = {}
    for cooa, coob in edges:
        coordinations[cooa] = coordinations.setdefault(cooa, 0) + 1
        coordinations[coob] = coordinations.setdefault(coob, 0) + 1

    jx_factory = make_edge_factory(jx)
    hz_factory = make_node_factory(hz)

    return {
        (cooa, coob): tfim_local_array(
            symmetry,
            jx=jx_factory(cooa, coob),
            hz=(hz_factory(cooa), hz_factory(coob)),
            coordinations=(coordinations[cooa], coordinations[coob]),
            flat=flat,
        )
        for cooa, coob in edges
    }


def ham_heisenberg_from_edges(
    symmetry,
    edges,
    j=1.0,
    b=0.0,
    flat=False,
):
    """Return a dict of local 2-body Hamiltonian abelian symmetric terms for
    the Heisenberg model on the given lattice defined by `edges`::

        H = sum_ij (jx Sx_i Sx_j + jy Sy_i Sy_j + jz Sz_i Sz_j)
            - sum_i bz Sz_i

    where the spin operators are the spin-1/2 operators (eigenvalues +/- 1/2).

    Parameters
    ----------
    symmetry : str or None
        The symmetry of the model. Either "Z2" or "U1", or None to return raw
        dense arrays. Note "U1" requires the XY couplings to be equal
        (``jx == jy``), so that total magnetization is conserved.
    edges : Sequence[tuple[Hashable, Hashable]]
        A list of edges representing the lattice, each edge is a tuple of two
        nodes, each node is some hashable label.
    j : float, tuple, dict, or callable, optional
        The coupling strength, by default 1.0. A tuple is interpreted as the
        ``(jx, jy, jz)`` couplings. If a dict is given it should map edges to
        values, if a callable it should take the two sites as input.
    b : float, dict, or callable, optional
        The magnetic field along the z-axis, by default 0.0. If a dict is given
        it should map sites to values, if a callable it should take the site as
        input, in either case allowing a different field per site. The field is
        divided by each site's coordination to avoid double counting. Only
        z-fields are supported as transverse fields would not conserve the
        symmetry.
    flat : bool, optional
        Whether to return flat arrays, by default False.

    Returns
    -------
    dict[tuple[Hashable, Hashable], AbelianArray | AbelianArrayFlat]
        A dictionary mapping edges to local Hamiltonian terms.
    """
    from .spin_local_operators import heisenberg_local_array

    coordinations = {}
    for cooa, coob in edges:
        coordinations[cooa] = coordinations.setdefault(cooa, 0) + 1
        coordinations[coob] = coordinations.setdefault(coob, 0) + 1

    j_factory = make_edge_factory(j)
    b_factory = make_node_factory(b)

    return {
        (cooa, coob): heisenberg_local_array(
            symmetry,
            j=j_factory(cooa, coob),
            b=(b_factory(cooa), b_factory(coob)),
            coordinations=(coordinations[cooa], coordinations[coob]),
            flat=flat,
        )
        for cooa, coob in edges
    }


def ham_fermi_hubbard_from_edges(
    symmetry,
    edges,
    t=1.0,
    U=8.0,
    mu=0.0,
    like="numpy",
    flat=False,
):
    """Return a dict of local 2-body Hamiltonian terms for the Fermi-Hubbard
    model on the given lattice defined by `edges`. The indices are ordered as
    (a, b, a', b'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively and similar for site b.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    edges : Sequence[tuple[Hashable, Hashable]]
        A list of edges representing the lattice, each edge is a tuple of two
        nodes, each node is some hashable label.
    t : float, dict or callable, optional
        The hopping parameter, by default 1.0. If a dict is given it should map
        edges to values, if a callable it should take the two sites as input.
    U : float dict or callable, optional
        The interaction parameter, by default 8.0. If a dict is given it should
        map sites to values, if a callable it should take the site as input.
    mu : float, optional
        The chemical potential, by default 0.0.
    like : str, optional
        The backend to use, by default "numpy".
    flat : bool, optional
        Whether to return flat arrays, by default False.

    Returns
    -------
    dict[tuple[Hashable, Hashable], FermionicArray | FermionicArrayFlat]
        A dictionary mapping edges to local Hamiltonian terms.
    """
    from .fermionic_local_operators import fermi_hubbard_local_array

    coordinations = {}
    for cooa, coob in edges:
        coordinations[cooa] = coordinations.setdefault(cooa, 0) + 1
        coordinations[coob] = coordinations.setdefault(coob, 0) + 1

    t_factory = make_edge_factory(t)
    U_factory = make_node_factory(U)
    mu_factory = make_node_factory(mu)

    return {
        (cooa, coob): fermi_hubbard_local_array(
            symmetry,
            t=t_factory(cooa, coob),
            U=(U_factory(cooa), U_factory(coob)),
            mu=(mu_factory(cooa), mu_factory(coob)),
            coordinations=(coordinations[cooa], coordinations[coob]),
            like=like,
            flat=flat,
        )
        for cooa, coob in edges
    }


def ham_fermi_hubbard_spinless_from_edges(
    symmetry,
    edges,
    t=1.0,
    V=0.0,
    mu=0.0,
    delta=0.0,
    like="numpy",
    flat=False,
):
    """Return a dict of local 2-body Hamiltonian terms for the 'spinless
    Fermi-Hubbard' or ('t-V') model on the given lattice defined by `edges`.
    model on the given lattice defined by `edges`.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2" or "U1".
    edges : Sequence[tuple[Hashable, Hashable]]
        A list of edges representing the lattice, each edge is a tuple of two
        nodes, each node is some hashable label.
    t : float, optional
        The hopping parameter, by default 1.0.
    V : float, optional
        The nearest neighbor interaction parameter, by default 0.0.
    mu : float, optional
        The chemical potential, by default 0.0.
    delta : float, optional
        The nearest neighbor superconducting pairing parameter, by default 0.0.
    like : str, optional
        The backend to use, by default "numpy".
    flat : bool, optional
        Whether to return flat arrays, by default False.

    Returns
    -------
    dict[tuple[Hashable, Hashable], FermionicArray | FermionicArrayFlat]
        A dictionary mapping edges to local Hamiltonian terms.
    """
    from .fermionic_local_operators import fermi_hubbard_spinless_local_array

    coordinations = {}
    for cooa, coob in edges:
        coordinations[cooa] = coordinations.setdefault(cooa, 0) + 1
        coordinations[coob] = coordinations.setdefault(coob, 0) + 1

    t_factory = make_edge_factory(t)
    V_factory = make_edge_factory(V)
    mu_factory = make_node_factory(mu)
    delta_factory = make_edge_factory(delta)

    return {
        (cooa, coob): fermi_hubbard_spinless_local_array(
            symmetry,
            t=t_factory(cooa, coob),
            V=V_factory(cooa, coob),
            mu=(mu_factory(cooa), mu_factory(coob)),
            delta=delta_factory(cooa, coob),
            coordinations=(coordinations[cooa], coordinations[coob]),
            like=like,
            flat=flat,
        )
        for cooa, coob in edges
    }
