"""Abelian symmetric and fermionic Hamiltonians in terms of local operators."""


def make_edge_factory(t):
    """Ensure `t` is a function that takes two sites and returns an edge coeff."""
    if isinstance(t, dict):

        def edge_factory(cooa, coob):
            try:
                return t[(cooa, coob)]
            except KeyError:
                return t[(coob, cooa)]

    elif callable(t):
        edge_factory = t

    else:

        def edge_factory(cooa, coob):
            return t

    return edge_factory


def make_node_factory(U):
    """Ensure `U` is a function that takes a site and returns a node coeff."""
    if isinstance(U, dict):

        def node_factory(coo):
            return U[coo]

    elif callable(U):
        node_factory = U

    else:

        def node_factory(coo):
            return U

    return node_factory


def tfim_local_array(symmetry, jx=-1.0, hz=-3.0, coordinations=(1, 1)):
    """Build an abelian symmetric local operator for the transverse field
    Ising model::

        H = jx * sum_ij X_i X_j + hz * sum_i Z_i

    Note that its rotated into the x-basis so that the Z2 symmetry is manifest.
    """
    import quimb as qu

    from .utils import from_dense

    if symmetry != "Z2":
        raise ValueError(f"Symmetry {symmetry} not supported for TFIM.")

    index_map = [0, 1]

    try:
        ha, hb = hz
    except TypeError:
        ha = hb = hz

    I, Z, X = (qu.pauli(s, dtype="float64") for s in "IZX")

    h2 = (
        jx * (X & X)
        + (ha / coordinations[0]) * (Z & I)
        + (hb / coordinations[1]) * (I & Z)
    )

    return from_dense(
        h2.reshape(2, 2, 2, 2),
        symmetry=symmetry,
        index_maps=[index_map] * 4,
        duals=[False, False, True, True],
    )


def ham_tfim_from_edges(
    symmetry,
    edges,
    jx=-1.0,
    hz=-3.0,
):
    """Return a dict of local 2-body Hamiltonian abelian symmetric terms for
    the transverse field Ising model on the given lattice defined by `edges`::

        H = jx * sum_ij X_i X_j + hz * sum_i Z_i

    Note that its rotated into the x-basis so that the Z2 symmetry is manifest.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Should be "Z2".
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

    Returns
    -------
    dict[tuple[Hashable, Hashable], AbelianArray]
        A dictionary mapping edges to local Hamiltonian terms.
    """
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
        )
        for cooa, coob in edges
    }


def ham_heisenberg_from_edges(
    symmetry,
    edges,
    **kwargs,
):
    import quimb as qu

    from .utils import from_dense

    h2 = qu.ham_heis(2, **kwargs)
    index_map = [0, 1]

    h2 = from_dense(
        h2.reshape(2, 2, 2, 2),
        symmetry=symmetry,
        index_maps=[index_map] * 4,
        duals=[False, False, True, True],
    )

    return {(a, b): h2 for a, b in edges}


def ham_fermi_hubbard_from_edges(
    symmetry,
    edges,
    t=1.0,
    U=8.0,
    mu=0.0,
    like="numpy",
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

    Returns
    -------
    dict[tuple[Hashable, Hashable], FermionicArray]
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
        )
        for cooa, coob in edges
    }


def ham_fermi_hubbard_spinless_from_edges(
    symmetry,
    edges,
    t=1.0,
    V=0.0,
    mu=0.0,
    like="numpy",
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
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    dict[tuple[Hashable, Hashable], FermionicArray]
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

    return {
        (cooa, coob): fermi_hubbard_spinless_local_array(
            symmetry,
            t=t_factory(cooa, coob),
            V=V_factory(cooa, coob),
            mu=(mu_factory(cooa), mu_factory(coob)),
            coordinations=(coordinations[cooa], coordinations[coob]),
            like=like,
        )
        for cooa, coob in edges
    }
