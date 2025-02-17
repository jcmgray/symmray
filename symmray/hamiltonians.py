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
    t : float, optional
        The hopping parameter, by default 1.0.
    U : float, optional
        The interaction parameter, by default 8.0.
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

    return {
        (cooa, coob): fermi_hubbard_local_array(
            symmetry,
            t=t,
            U=U,
            mu=mu,
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

    return {
        (cooa, coob): fermi_hubbard_spinless_local_array(
            symmetry,
            t=t,
            V=V,
            mu=mu,
            coordinations=(coordinations[cooa], coordinations[coob]),
            like=like,
        )
        for cooa, coob in edges
    }
