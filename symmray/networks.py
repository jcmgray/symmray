"""Functions to create full tensor networks backed by `symmray`."""

from collections.abc import Mapping

import autoray as ar


def parse_edges_to_site_info(
    edges,
    bond_dim,
    phys_dim=2,
    site_ind_id="k{}",
    bond_ind_id="b{}-{}",
    site_tag_id="I{}",
    duals="reversed",
    seed=None,
):
    """Given a list of edges, return a dictionary of site information, each
    specifying the local shape, index identifiers, index dualnesses, and tags.
    The dualnesses of the bonds can be set in reversed, canonical, random, or
    explicitly mapped orientations. The default is reversed for backwards
    compatibility.

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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use when ``duals="random"``.

    Returns
    -------
    Dict[hashable, Dict[str, Any]]
    """
    site_info = {}

    if isinstance(duals, Mapping):
        duals_unused = set(duals)
    elif duals not in ("reversed", "canonical", "random"):
        raise ValueError(f"Unrecognized duals: {duals}.")

    if duals == "random":
        from symmray.utils import get_rng

        rng = get_rng(seed)

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

        if duals == "reversed":
            duala = False
        elif duals == "canonical":
            duala = True
        elif duals == "random":
            duala = bool(rng.choice([False, True]))
        else:
            duala = False
            for key in (ind, (sitea, siteb), (siteb, sitea)):
                if key in duals:
                    duala = bool(duals[key])
                    duals_unused.discard(key)
                    break

        infoa.setdefault("duals", []).append(duala)
        infob.setdefault("duals", []).append(not duala)

        infoa.setdefault("shape", []).append(bond_dim)
        infob.setdefault("shape", []).append(bond_dim)

    if isinstance(duals, Mapping) and duals_unused:
        raise ValueError(f"Duals keys matched no bond: {duals_unused}.")

    # create physical inds
    for site, sinfo in site_info.items():
        sinfo["coordination"] = len(sinfo["inds"])

        if starmap_tag:
            site_tag = site_tag_id.format(*site)
        else:
            site_tag = site_tag_id.format(site)

        sinfo["tags"] = (site_tag,)

        if phys_dim is not None:
            if starmap_ind:
                site_ind = site_ind_id.format(*site)
            else:
                site_ind = site_ind_id.format(site)

            sinfo["inds"].append(site_ind)
            sinfo["duals"].append(0)
            sinfo["shape"].append(phys_dim)

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
    bond_ind_id="b{}-{}",
    fermionic=False,
    flat=False,
    site_charge=None,
    subsizes="maximal",
    duals="reversed",
    **kwargs,
):
    """Build a random abelian symmetric `quimb.tensor` amplitude or PEPS from
    edges.

    Parameters
    ----------
    symmetry : {"Z2", "U1", "Z2Z2", "U1U1", ZN}
        The symmetry of the PEPS.
    edges : list of tuples
        The edges of the PEPS. Each edge is a tuple of the form `(cooa, coob)`
        where cooa and coob are hashable, comparable labels of the two sites.
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
        The data type of the tensors, default is "float64".
    site_tag_id : str, optional
        The tag format for each site tensor, default is "I{}".
    site_ind_id : str, optional
        The index format for each site tensor, if physical sites are included.
        Default is "k{}".
    bond_ind_id : str, optional
        The index format for each bond, default is "b{}-{}".
    fermionic : bool, optional
        Whether to generate fermionic tensors, default is False.
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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.TensorNetworkGen or quimb.tensor.TensorNetworkGenVector
    """
    import quimb.tensor as qtn

    import symmray as sr

    rng = sr.utils.get_rng(seed)

    site_info = parse_edges_to_site_info(
        edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        site_ind_id=site_ind_id,
        site_tag_id=site_tag_id,
        bond_ind_id=bond_ind_id,
        duals=duals,
        seed=rng,
    )
    sites = tuple(site_info.keys())

    if phys_dim is not None:
        # have physical sites
        tn = qtn.TensorNetworkGenVector.new(
            sites=sites,
            site_ind_id=site_ind_id,
            site_tag_id=site_tag_id,
        )

        if isinstance(phys_dim, dict):
            # custom physical charge map
            phys_chargemap = phys_dim
        elif ar.is_scalar(phys_dim):
            # total physical dimension
            phys_chargemap = _DEFAULT_PHYS_CHARGEMAPS[symmetry, phys_dim]
        else:
            phys_chargemap = phys_dim
    else:
        # no physical sites
        tn = qtn.TensorNetworkGen.new(
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
                + "`site_charge`. Please provide one."
            )

    index_store = {}
    for site, info in site_info.items():
        if phys_dim is None:
            shape = info["shape"]
        else:
            shape = info["shape"][:-1] + [phys_chargemap]
        site_duals = info["duals"]

        # resolve shape sizes into explicit indices ahead of time so that we
        # can build them in conj-pairs with matching subsizes even if random
        shape_parsed = []
        for ix, size, dual in zip(info["inds"], shape, site_duals):
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
                duals=site_duals,
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
    duals="reversed",
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
        duals=duals,
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
    duals="reversed",
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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
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
        duals=duals,
        **kwargs,
    )

    return mps.view_as_(
        qtn.MatrixProductState,
        L=L,
        cyclic=cyclic,
    )


def TN2D_abelian_rand(
    symmetry,
    Lx,
    Ly,
    bond_dim,
    phys_dim=None,
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
    duals="reversed",
    **kwargs,
):
    """Create a random 2D symmetric tensor network.

    This returns a scalar-valued :class:`quimb.tensor.TensorNetwork2D` when
    ``phys_dim=None`` and a :class:`quimb.tensor.PEPS` otherwise.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the network.
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    bond_dim : int or dict
        The total bond dimension, or an explicit bond charge map.
    phys_dim : None, int or dict, optional
        The physical dimension of each site. If None, construct a scalar
        network without physical indices.
    cyclic : bool, optional
        Whether to make the network cyclic in the x-direction.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each physical site index.
    x_tag_id : str, optional
        The tag format for each x-coordinate.
    y_tag_id : str, optional
        The tag format for each y-coordinate.
    fermionic : bool, optional
        Whether to generate fermionic tensors.
    flat : bool, optional
        Whether to generate flat backend arrays.
    site_charge : callable, optional
        A function mapping each site coordinate to its total charge.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors.
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.TensorNetwork2D or quimb.tensor.PEPS
    """
    import quimb.tensor as qtn

    edges = qtn.edges_2d_square(Lx, Ly, cyclic=cyclic)
    tn = TN_abelian_from_edges_rand(
        symmetry=symmetry,
        edges=edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
        subsizes=subsizes,
        duals=duals,
        **kwargs,
    )

    starmap_tag = site_tag_id.count("{}") > 1
    for i in range(Lx):
        for j in range(Ly):
            if starmap_tag:
                site_tag = site_tag_id.format(i, j)
            else:
                site_tag = site_tag_id.format((i, j))
            tn[site_tag].add_tag(x_tag_id.format(i))
            tn[site_tag].add_tag(y_tag_id.format(j))

    if phys_dim is None:
        cls = qtn.TensorNetwork2D
    else:
        cls = qtn.PEPS

    return tn.view_as_(cls, Lx=Lx, Ly=Ly, x_tag_id=x_tag_id, y_tag_id=y_tag_id)


def TN2D_fermionic_rand(
    symmetry,
    Lx,
    Ly,
    bond_dim,
    phys_dim=None,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{},{}",
    site_ind_id="k{},{}",
    x_tag_id="X{}",
    y_tag_id="Y{}",
    site_charge=None,
    subsizes="maximal",
    duals="reversed",
    **kwargs,
):
    """Create a random 2D fermionic symmetric tensor network.

    This is a wrapper around :func:`TN2D_abelian_rand` with
    ``fermionic=True``.
    """
    return TN2D_abelian_rand(
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
        duals=duals,
        **kwargs,
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
    duals="reversed",
    **kwargs,
):
    """Create a random 2D PEPS with abelian symmetry.

    This is a wrapper around :func:`TN2D_abelian_rand` with a default
    physical dimension of 2.

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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS or quimb.tensor.TensorNetwork2D
    """
    return TN2D_abelian_rand(
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
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
        subsizes=subsizes,
        duals=duals,
        **kwargs,
    )


def TN3D_abelian_rand(
    symmetry,
    Lx,
    Ly,
    Lz,
    bond_dim,
    phys_dim=None,
    cyclic=False,
    seed=None,
    dtype="float64",
    site_tag_id="I{},{},{}",
    site_ind_id="k{},{},{}",
    x_tag_id="X{}",
    y_tag_id="Y{}",
    z_tag_id="Z{}",
    fermionic=False,
    flat=False,
    site_charge=None,
    subsizes="maximal",
    duals="reversed",
    **kwargs,
):
    """Create a random 3D symmetric tensor network.

    This returns a scalar-valued :class:`quimb.tensor.TensorNetwork3D` when
    ``phys_dim=None`` and a :class:`quimb.tensor.PEPS3D` otherwise.

    Parameters
    ----------
    symmetry : str or Symmetry
        The symmetry of the network.
    Lx : int
        The size of the network in the x-direction.
    Ly : int
        The size of the network in the y-direction.
    Lz : int
        The size of the network in the z-direction.
    bond_dim : int or dict
        The total bond dimension, or an explicit bond charge map.
    phys_dim : None, int or dict, optional
        The physical dimension of each site. If None, construct a scalar
        network without physical indices.
    cyclic : bool, optional
        Whether to make the network cyclic in the x-direction.
    seed : None, int or np.random.Generator, optional
        The random seed or generator to use.
    dtype : str, optional
        The data type of the tensors.
    site_tag_id : str, optional
        The tag format for each site tensor.
    site_ind_id : str, optional
        The index format for each physical site index.
    x_tag_id : str, optional
        The tag format for each x-coordinate.
    y_tag_id : str, optional
        The tag format for each y-coordinate.
    z_tag_id : str, optional
        The tag format for each z-coordinate.
    fermionic : bool, optional
        Whether to generate fermionic tensors.
    flat : bool, optional
        Whether to generate flat backend arrays.
    site_charge : callable, optional
        A function mapping each site coordinate to its total charge.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors.
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.TensorNetwork3D or quimb.tensor.PEPS3D
    """
    import quimb.tensor as qtn

    edges = qtn.edges_3d_cubic(Lx, Ly, Lz, cyclic=cyclic)
    tn = TN_abelian_from_edges_rand(
        symmetry=symmetry,
        edges=edges,
        bond_dim=bond_dim,
        phys_dim=phys_dim,
        seed=seed,
        dtype=dtype,
        site_tag_id=site_tag_id,
        site_ind_id=site_ind_id,
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
        subsizes=subsizes,
        duals=duals,
        **kwargs,
    )

    starmap_tag = site_tag_id.count("{}") > 1
    for i in range(Lx):
        for j in range(Ly):
            for k in range(Lz):
                if starmap_tag:
                    site_tag = site_tag_id.format(i, j, k)
                else:
                    site_tag = site_tag_id.format((i, j, k))
                tn[site_tag].add_tag(x_tag_id.format(i))
                tn[site_tag].add_tag(y_tag_id.format(j))
                tn[site_tag].add_tag(z_tag_id.format(k))

    if phys_dim is None:
        cls = qtn.TensorNetwork3D
    else:
        cls = qtn.PEPS3D

    return tn.view_as_(
        cls,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        x_tag_id=x_tag_id,
        y_tag_id=y_tag_id,
        z_tag_id=z_tag_id,
    )


def TN3D_fermionic_rand(
    symmetry,
    Lx,
    Ly,
    Lz,
    bond_dim,
    phys_dim=None,
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
    duals="reversed",
    **kwargs,
):
    """Create a random 3D fermionic symmetric tensor network.

    This is a wrapper around :func:`TN3D_abelian_rand` with
    ``fermionic=True``.
    """
    return TN3D_abelian_rand(
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
        duals=duals,
        **kwargs,
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
    flat=False,
    site_charge=None,
    subsizes="maximal",
    duals="reversed",
    **kwargs,
):
    """Create a random 3D PEPS with abelian symmetry.

    This is a wrapper around :func:`TN3D_abelian_rand` with a default
    physical dimension of 2.

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
    flat : bool, optional
        Whether to generate flat backend arrays.
    site_charge : callable, optional
        A function that takes a site index and returns the charge of that site.
        By default it will create all even parity tensors if Z2=0 or it will
        alernate between 0 and 1 for U1.
    subsizes : {"maximal", "equal"}, optional
        The sizes of the charge sectors. If None, the sizes are randomly
        determined. If "equal", the sizes are equal (up to remainders). If
        "maximal", as many charges as possible will be chosen.
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS3D or quimb.tensor.TensorNetwork3D
    """
    return TN3D_abelian_rand(
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
        fermionic=fermionic,
        flat=flat,
        site_charge=site_charge,
        subsizes=subsizes,
        duals=duals,
        **kwargs,
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
    duals="reversed",
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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
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
        duals=duals,
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
    duals="reversed",
    **kwargs,
):
    """Create a random 2D fermionic PEPS. This is a wrapper around
    :func:`TN2D_fermionic_rand` with a default physical dimension of 2.

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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS or quimb.tensor.TensorNetwork2D
    """
    return TN2D_fermionic_rand(
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
        site_charge=site_charge,
        subsizes=subsizes,
        duals=duals,
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
    duals="reversed",
    **kwargs,
):
    """Create a random 3D fermionic PEPS. This is a wrapper around
    :func:`TN3D_fermionic_rand` with a default physical dimension of 2.

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
    duals : {"reversed", "canonical", "random"} or dict, optional
        The bond orientation convention. "reversed" assigns the lower
        site a non-dual index and is the backwards-compatible default;
        "canonical" assigns it a dual index; "random" chooses each
        bond orientation independently; and a mapping can override individual
        bonds, keyed by bond index name or by the edge in either order. Bonds
        the mapping does not name use "reversed", and keys matching no bond
        raise a `ValueError`.
    kwargs
        Additional arguments to pass to :func:`symmray.utils.get_rand`.

    Returns
    -------
    quimb.tensor.PEPS3D or quimb.tensor.TensorNetwork3D
    """
    return TN3D_fermionic_rand(
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
        site_charge=site_charge,
        subsizes=subsizes,
        duals=duals,
        **kwargs,
    )
