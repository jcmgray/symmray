"""Functions to create full tensor networks backed by `symmray`."""

import numbers


def parse_edges_to_site_info(
    edges,
    bond_dim,
    phys_dim=2,
    site_ind_id="k{}",
    bond_ind_id="b{}-{}",
    site_tag_id="I{}",
):
    """Given a list of edges, return a dictionary of site information, each
    specifying the local shape, index identifiers, index dualnesses, and tags.
    The dualnesses of the bonds are set in a canonical order corresponding to
    sorting all the sites and the edges.

    Parameters
    ----------
    edges : Sequence[Tuple[hashable, hashable]]
        The edges of the graph.
    bond_dim : int
        The internal bond dimension.
    phys_dim : int, optional
        The physical dimension of the sites.
    site_ind_id : str, optional
        The identifier for the site indices.
    bond_ind_id : str, optional
        The identifier for the bond indices.
    site_tag_id : str, optional
        The identifier for the site tags.

    Returns
    -------
    Dict[hashable, Dict[str, Any]]
    """
    site_info = {}

    starmap_ind = site_ind_id.count("{}") > 1
    starmap_tag = site_tag_id.count("{}") > 1

    # create bonds
    for sitea, siteb in sorted(edges):
        if sitea > siteb:
            sitea, siteb = siteb, sitea

        ind = bond_ind_id.format(sitea, siteb)
        infoa = site_info.setdefault(sitea, {})
        infob = site_info.setdefault(siteb, {})

        infoa.setdefault("inds", []).append(ind)
        infob.setdefault("inds", []).append(ind)

        infoa.setdefault("duals", []).append(0)
        infob.setdefault("duals", []).append(1)

        infoa.setdefault("shape", []).append(bond_dim)
        infob.setdefault("shape", []).append(bond_dim)

    # create physical inds
    for site in site_info:
        site_info[site]["coordination"] = len(site_info[site]["inds"])

        if starmap_tag:
            site_tag = site_tag_id.format(*site)
        else:
            site_tag = site_tag_id.format(site)

        site_info[site]["tags"] = (site_tag,)

        if phys_dim is not None:
            if starmap_ind:
                site_ind = site_ind_id.format(*site)
            else:
                site_ind = site_ind_id.format(site)

            site_info[site]["inds"].append(site_ind)
            site_info[site]["duals"].append(0)
            site_info[site]["shape"].append(phys_dim)

    # put in canonical sorted by site order
    site_info = {k: site_info[k] for k in sorted(site_info)}

    return site_info


_DEFAULT_PHYS_CHARGEMAPS = {
    ("Z2", 2): {0: 1, 1: 1},
    ("U1", 2): {0: 1, 1: 1},
    ("Z2", 4): {0: 2, 1: 2},
    ("U1", 4): {0: 1, 1: 2, 2: 1},
    ("U1U1", 4): {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
    ("Z2Z2", 4): {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1},
}


def TN_abelian_from_edges_rand(
    symmetry,
    edges,
    bond_dim,
    phys_dim=None,
    seed=None,
    dtype="float64",
    site_tag_id="I{}",
    site_ind_id="k{}",
    fermionic=False,
    flat=False,
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Build a random abelian symmetric `quimb.tensor` amplitude or PEPS from
    edges.

    Parameters
    ----------
    symmetry : {"Z2", "U1", "Z2Z2", "U1U1", ZN}
        The symmetry of the PEPS. Currently only "Z2" and "U1" are supported.
    edges : list of tuples
        The edges of the PEPS. Each edge is a tuple of the form `(cooa, coob)`
        where cooa and coob are hashable labels of the two sites.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the PEPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : None, int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, all sites have the same dimension. If a
        dictionary, a custom map of physical charges to sizes.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    fermionic : bool, optional
        Whether to generate fermionic tensors.
    flat : bool, optional
        Whether to generate 'flat' backend arrays (True) or the default
        block-sparse backend arrays (False).
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alternate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.TensorNetworkGen or quimb.tensor.TensorNetworkGenVector
    """
    import quimb.tensor as qtn

    import symmray as sr

    site_info = parse_edges_to_site_info(
        edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        site_ind_id=site_ind_id,
        site_tag_id=site_tag_id,
    )
    sites = tuple(site_info.keys())

    if phys_dim is not None:
        # have physical sites
        tn = qtn.tensor_arbgeom.TensorNetworkGenVector.new(
            sites=sites,
            site_ind_id=site_ind_id,
            site_tag_id=site_tag_id,
        )

        if isinstance(phys_dim, dict):
            # custom physical charge map
            phys_chargemap = phys_dim
        elif isinstance(phys_dim, numbers.Integral):
            phys_chargemap = _DEFAULT_PHYS_CHARGEMAPS[symmetry, phys_dim]
        else:
            phys_chargemap = phys_dim
    else:
        # no physical sites

        tn = qtn.tensor_arbgeom.TensorNetworkGen.new(
            sites=sites,
            site_tag_id=site_tag_id,
        )

        phys_chargemap = None

    if site_charge is None:
        from symmray.symmetries import ZN, get_symmetry

        if symmetry == "U1":
            even_sites = set(sites[::2])

            def site_charge(site):
                return 0 if site in even_sites else 1

        elif isinstance(get_symmetry(symmetry), ZN):

            def site_charge(site):
                return 0

        else:
            raise ValueError(
                f"symmetry={symmetry} has no default "
                "`site_charge`. Please provide one."
            )

    rng = sr.utils.get_rng(seed)

    index_store = {}
    for site, info in site_info.items():
        if phys_dim is None:
            shape = info["shape"]
        else:
            shape = info["shape"][:-1] + [phys_chargemap]
        duals = info["duals"]

        # resolve shape sizes into explicit indices ahead of time so that we
        # can build them in conj-pairs with matching subsizes even if random
        shape_parsed = []
        for ix, size, dual in zip(info["inds"], shape, duals):
            if ix in index_store:
                shape_parsed.append(index_store[ix].conj())
            else:
                index_store[ix] = sr.utils.rand_index(
                    symmetry,
                    size,
                    dual=dual,
                    subsizes=subsizes,
                    seed=rng,
                )
                shape_parsed.append(index_store[ix])

        tn |= qtn.Tensor(
            data=sr.utils.get_rand(
                shape=shape_parsed,
                duals=duals,
                symmetry=symmetry,
                charge=site_charge(site),
                fermionic=fermionic,
                flat=flat,
                subsizes=subsizes,
                seed=rng,
                dtype=dtype,
                # possibly needed for odd parity fermionic tensors
                label=site,
                **kwargs,
            ),
            inds=info["inds"],
            tags=info["tags"],
        )

    return tn


def TN_fermionic_from_edges_rand(
    symmetry,
    edges,
    bond_dim,
    phys_dim=None,
    seed=None,
    dtype="float64",
    site_tag_id="I{}",
    site_ind_id="k{}",
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random fermionic tensor network from edges. This is a wrapper
    around :func:`TN_abelian_from_edges_rand` with `fermionic=True`.

    Parameters
    ----------
    symmetry : {"Z2", "U1", "Z2Z2", "U1U1", ZN}
        The symmetry of the PEPS. Currently only "Z2" and "U1" are supported.
    edges : list of tuples
        The edges of the PEPS. Each edge is a tuple of the form `(cooa, coob)`
        where cooa and coob are hashable labels of the two sites.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the PEPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : None, int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, all sites have the same dimension. If a
        dictionary, a custom map of physical charges to sizes.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.TensorNetworkGen or quimb.tensor.TensorNetworkGenVector
    """
    return TN_abelian_from_edges_rand(
        symmetry=symmetry,
        edges=edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        fermionic=True,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )


def MPS_abelian_rand(
    symmetry,
    L,
    bond_dim,
    phys_dim=2,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{}",
    site_ind_id="k{}",
    fermionic=False,
    flat=False,
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random MPS with abelian symmetry.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the MPS.
    L : int
        The number of sites.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the MPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, a default charge distribution is chosen. If a
        dictionary, a custom map of physical charges to sizes.
    cyclic : bool, optional
        Whether the MPS is cyclic.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    fermionic : bool, optional
        Whether to generate fermionic tensors.
    flat : bool, optional
        Whether to generate 'flat' backend arrays (True) or the default
        block-sparse backend arrays (False).
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.MatrixProductState
    """
    import quimb.tensor as qtn

    edges = qtn.edges_1d_chain(L, cyclic=cyclic)

    mps = TN_abelian_from_edges_rand(
        symmetry,
        edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
        site_ind_id=site_ind_id,
        site_tag_id=site_tag_id,
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )

    return mps.view_as_(
        qtn.MatrixProductState,
        L=L,
        cyclic=cyclic,
    )


def PEPS_abelian_rand(
    symmetry,
    Lx,
    Ly,
    bond_dim,
    phys_dim=2,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{},{}",
    site_ind_id="k{},{}",
    x_tag_id="X{}",
    y_tag_id="Y{}",
    fermionic=False,
    flat=False,
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random 2D PEPS with abelian symmetry.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the PEPS.
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the PEPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, a default charge distribution is chosen. If a
        dictionary, a custom map of physical charges to sizes.
    cyclic : bool, optional
        Whether the PEPS is cyclic in the x-direction.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    fermionic : bool, optional
        Whether to generate fermionic tensors.
    flat : bool, optional
        Whether to generate 'flat' backend arrays (True) or the default
        block-sparse backend arrays (False).
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS
    """
    import quimb.tensor as qtn

    edges = qtn.edges_2d_square(Lx, Ly, cyclic=cyclic)

    peps = TN_abelian_from_edges_rand(
        symmetry,
        edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
        site_ind_id=site_ind_id,
        site_tag_id=site_tag_id,
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )

    for i in range(Lx):
        for j in range(Ly):
            t = peps[site_tag_id.format(i, j)]
            t.add_tag(x_tag_id.format(i))
            t.add_tag(y_tag_id.format(j))

    return peps.view_as_(
        qtn.PEPS, Lx=Lx, Ly=Ly, x_tag_id=x_tag_id, y_tag_id=y_tag_id
    )


def PEPS3D_abelian_rand(
    symmetry,
    Lx,
    Ly,
    Lz,
    bond_dim,
    phys_dim=2,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{},{},{}",
    site_ind_id="k{},{},{}",
    x_tag_id="X{}",
    y_tag_id="Y{}",
    z_tag_id="Z{}",
    fermionic=False,
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random 3D PEPS with abelian symmetry.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the PEPS.
    Lx : int
        Length of the PEPS in the x-direction.
    Ly : int
        Length of the PEPS in the y-direction.
    Lz : int
        Length of the PEPS in the z-direction.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the PEPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, a default charge distribution is chosen. If a
        dictionary, a custom map of physical charges to sizes.
    cyclic : bool, optional
        Whether the PEPS is cyclic in the x-direction.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    fermionic : bool, optional
        Whether to generate fermionic tensors.
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS3D
    """
    import quimb.tensor as qtn

    edges = qtn.edges_3d_cubic(Lx, Ly, Lz, cyclic=cyclic)

    peps = TN_abelian_from_edges_rand(
        symmetry,
        edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
        site_ind_id=site_ind_id,
        site_tag_id=site_tag_id,
        fermionic=fermionic,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )

    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                t = peps[site_tag_id.format(i, j, k)]
                t.add_tag(x_tag_id.format(i))
                t.add_tag(y_tag_id.format(j))
                t.add_tag(z_tag_id.format(k))

    return peps.view_as_(
        qtn.PEPS3D,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        x_tag_id=x_tag_id,
        y_tag_id=y_tag_id,
        z_tag_id=z_tag_id,
    )


def MPS_fermionic_rand(
    symmetry,
    L,
    bond_dim,
    phys_dim=2,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{}",
    site_ind_id="k{}",
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random fermionic MPS. This is a wrapper around
    :func:`MPS_abelian_rand` with `fermionic=True`.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the MPS.
    L : int
        The number of sites.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the MPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, a default charge distribution is chosen. If a
        dictionary, a custom map of physical charges to sizes.
    cyclic : bool, optional
        Whether the MPS is cyclic.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.MatrixProductState
    """
    return MPS_abelian_rand(
        symmetry=symmetry,
        L=L,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        cyclic=cyclic,
        seed=seed,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        fermionic=True,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )


def PEPS_fermionic_rand(
    symmetry,
    Lx,
    Ly,
    bond_dim,
    phys_dim=2,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{},{}",
    site_ind_id="k{},{}",
    x_tag_id="X{}",
    y_tag_id="Y{}",
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random 2D fermionic PEPS. This is a wrapper around
    :func:`PEPS_abelian_rand` with `fermionic=True`.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the PEPS.
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the PEPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, a default charge distribution is chosen. If a
        dictionary, a custom map of physical charges to sizes.
    cyclic : bool, optional
        Whether the PEPS is cyclic in the x-direction.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS
    """
    return PEPS_abelian_rand(
        symmetry=symmetry,
        Lx=Lx,
        Ly=Ly,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        cyclic=cyclic,
        seed=seed,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        x_tag_id=x_tag_id,
        y_tag_id=y_tag_id,
        fermionic=True,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )


def PEPS3D_fermionic_rand(
    symmetry,
    Lx,
    Ly,
    Lz,
    bond_dim,
    phys_dim=2,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{},{},{}",
    site_ind_id="k{},{},{}",
    x_tag_id="X{}",
    y_tag_id="Y{}",
    z_tag_id="Z{}",
    site_charge=None,
    subsizes="maximal",
    **kwargs,
):
    """Create a random 3D fermionic PEPS. This is a wrapper around
    :func:`PEPS3D_abelian_rand` with `fermionic=True`.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the PEPS.
    Lx : int
        Length of the PEPS in the x-direction.
    Ly : int
        Length of the PEPS in the y-direction.
    Lz : int
        Length of the PEPS in the z-direction.
    bond_dim : int or dict
        The total (sum of charge sizes) bond dimension of the PEPS. You can
        also provide an explicit map of bond charges to sizes.
    phys_dim : int or dict, optional
        The physical dimension of each site. If None, no physical sites are
        included. If an integer, a default charge distribution is chosen. If a
        dictionary, a custom map of physical charges to sizes.
    cyclic : bool, optional
        Whether the PEPS is cyclic in the x-direction.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS3D
    """
    return PEPS3D_abelian_rand(
        symmetry=symmetry,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        cyclic=cyclic,
        seed=seed,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        x_tag_id=x_tag_id,
        y_tag_id=y_tag_id,
        z_tag_id=z_tag_id,
        fermionic=True,
        site_charge=site_charge,
        subsizes=subsizes,
        **kwargs,
    )
