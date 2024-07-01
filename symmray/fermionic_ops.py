"""Helper functions for building local fermionic operators, with 'internal'
signs pre-computed.
"""

import itertools

import autoray as ar

from .utils import from_dense


class FermionicOperator:
    """Simple class to represent a fermionic operator with a label and a
    dagger flag.
    """

    def __init__(self, label, dagger=False):
        self.label = label
        self.dagger = dagger

    @property
    def dag(self):
        return FermionicOperator(self.label, not self.dagger)

    def __repr__(self):
        return f"{self.label}{'+' if self.dagger else '-'}"


def _dagger_basis(basis):
    """Return the conjugate basis of a given basis."""
    return tuple(tuple(op.dag for op in reversed(x)) for x in basis)


def _ensure_fermionic_operator(op):
    """Possibly convert a tuple of (label, symbol) to a FermionicOperator."""
    if isinstance(op, FermionicOperator):
        return op
    label, symbol = op
    return FermionicOperator(label, {"+": True, "-": False}[symbol])


def _parse_terms(terms):
    """Allow ops to be specified as tuples of (site, symbol) in addition to
    FermionicOperator instances. E.g. ``(-t, [('a', '+'), ('b', '-')])``.
    """
    return tuple(
        (coeff, tuple(map(_ensure_fermionic_operator, term)))
        for coeff, term in terms
    )


def _parse_bases(bases):
    """Allow ops to be specified as tuples of (site, symbol) in addition to
    FermionicOperator instances.
    """
    return tuple(
        tuple(tuple(map(_ensure_fermionic_operator, ops)) for ops in basis)
        for basis in bases
    )


def compute_local_fermionic_elements(terms, bases):
    """Compute the elements of a local fermionic operator in a given tensor
    basis, including 'internal' signs.

    Parameters
    ----------
    terms : tuple[tuple[float, tuple[FermionicOperator, ...]]]
        The terms in the operator, each a tuple of a coefficient and a tuple
        of FermionicOperator or tuple[label, op] instances.
    bases : tuple[tuple[tuple[FermionicOperator]]]
        The tensor bases to compute the operator elements in. Each basis is a
        sequence of multiple FermionicOperator instancess acting on the vacuum.

    Returns
    -------
    list[tuple[tuple[int], float]]
        A list of tuples of tensor indices and the corresponding tensor
        element, including phases.

    Examples
    --------
    Compute the elements of a local fermionic operator in a tensor basis::

        a, b = map(FermionicOperator, "ab")
        basis_a = [(), (a.dag,)]
        basis_b = [(), (b.dag,)]
        bases = (basis_a, basis_b)

        t = 1.0
        U = 8.0

        terms = (
            (-t, (a.dag, b)),
            (-t, (b.dag, a)),
            (U, (a.dag, a, b.dag, b)),
        )

        compute_local_fermionic_elements(terms, bases)
        # [((0, 1, 1, 0), -1.0), ((1, 0, 0, 1), -1.0), ((1, 1, 1, 1), -8.0)]

    """
    terms = _parse_terms(terms)
    bases = _parse_bases(bases)

    # construct |i>|j>|k> along with linear tensor indices
    enum_right_bases = [tuple(enumerate(x)) for x in bases]
    # and <i'|<j'|<k'|  n.b. not <k'|<j'|<i'|
    enum_left_bases = [tuple(enumerate(_dagger_basis(x))) for x in bases]

    all_locations = itertools.product(
        itertools.product(*enum_left_bases),
        itertools.product(*enum_right_bases),
    )

    entries = []
    for l, r in all_locations:
        # get tensor index, and left and right
        # basis operators it corresponds to
        left_indices, left_basis_ops = zip(*l)
        right_indices, right_basis_ops = zip(*r)

        # flatten into single lists of operators
        left_basis_ops = [op for ops in left_basis_ops for op in ops]
        right_basis_ops = [op for ops in right_basis_ops for op in ops]

        # process all terms at this tensor index
        for coeff, term in terms:
            if coeff == 0.0:
                continue

            # sandwich terms between basis operators
            element = [*left_basis_ops, *term, *right_basis_ops]

            # phased sort by labels
            phase = 1
            any_moves = True
            while any_moves:
                any_moves = False
                for k in range(len(element) - 1):
                    # check each pair of neighboring operators
                    opl = element[k]
                    opr = element[k + 1]
                    if opl.label > opr.label:
                        # swap if not grouped contiguously yet
                        element[k] = opr
                        element[k + 1] = opl
                        # flip phase from fermionic anticommutation
                        phase = -phase
                        any_moves = True

            # group by label
            groups = {}
            for x in element:
                groups.setdefault(x.label, []).append(x)

            # all groups must be like <0|(- + - + ... - +)|0> to not vanish
            nonvanishing = all(
                (
                    (len(group) % 2 == 0)
                    and all(not op.dagger for op in group[::2])
                    and all(op.dagger for op in group[1::2])
                )
                for group in groups.values()
            )

            if nonvanishing:
                # a non-vanishing element
                index = (*left_indices, *right_indices)
                entries.append((index, phase * coeff))

    return entries


def compute_local_fermionic_dense(terms, bases, like="numpy"):
    hij = ar.do("zeros", tuple(len(b) for b in bases) * 2, like=like)

    for idx, val in compute_local_fermionic_elements(terms, bases):
        hij[idx] += val

    return hij


def fermi_hubbard_spinless_local_dense(
    t=1.0,
    V=8.0,
    mu=0.0,
    coordinations=(1, 1),
    like="numpy",
):
    """Construct the dense local 4-tensor for the spinless Fermi-Hubbard model,
    with internal signs precomputed. The indices are ordered as (a, b, a', b').

    Parameters
    ----------
    t : float, optional
        The hopping parameter, by default 1.0.
    V : float, optional
        The nearest-neighbor interaction parameter, by default 8.0.
    mu : float, optional
        The chemical potential, by default 0.0.
    coordinations : tuple[int, int], optional
        The coordinations of the sites, by default (1, 1). If applying this
        local operator to every edge in a graph, then the single site
        contributions can be properly accounted for if the coordinations are
        provided.
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array
        The local operator in dense form.
    """
    a, b = map(FermionicOperator, "ab")
    basis_a = ((), (a.dag,))
    basis_b = ((), (b.dag,))
    bases = (basis_a, basis_b)

    terms = (
        # hopping
        (-t, (a.dag, b)),
        (-t, (b.dag, a)),
        # nearest-neighbor interaction
        (V, (a.dag, a, b.dag, b)),
        # chemical potential
        # mu is single site and will be overcounted without coordinations
        (-mu / coordinations[0], (a.dag, a)),
        (-mu / coordinations[1], (b.dag, b)),
    )

    return compute_local_fermionic_dense(terms, bases, like=like)


def fermi_hubbard_spinless_local_tensor(
    symmetry,
    t=1.0,
    V=8.0,
    mu=0.0,
    coordinations=(1, 1),
    like="numpy",
):
    """Construct the fermionic local tensor for the spinless Fermi-Hubbard
    model. The indices are ordered as (a, b, a', b').

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2" or "U1".
    t : float, optional
        The hopping parameter, by default 1.0.
    V : float, optional
        The nearest-neighbor interaction parameter, by default 8.0.
    mu : float, optional
        The chemical potential, by default 0.0.
    coordinations : tuple[int, int], optional
        The coordinations of the sites, by default (1, 1). If applying this
        local operator to every edge in a graph, then the single site
        contributions can be properly accounted for if the coordinations are
        provided.
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array
        The local operator in fermionic array form.
    """
    hij = fermi_hubbard_spinless_local_dense(
        t, V, mu, coordinations=coordinations, like=like
    )

    if symmetry == "Z2" or symmetry == "U1":
        indexmap = [0, 1]
    else:
        raise ValueError(f"Invalid symmetry: {symmetry}")

    return from_dense(
        hij,
        symmetry=symmetry,
        fermionic=True,
        index_maps=[indexmap] * 4,
        duals=[0, 0, 1, 1],
    )


def fermi_hubbard_local_dense(
    t=1.0,
    U=8.0,
    mu=0.0,
    coordinations=(1, 1),
    like="numpy",
):
    """Construct the dense local 4-tensor for the Fermi-Hubbard model, with
    internal signs precomputed. The indices are ordered as (a, b, a', b').
    The local basis is like (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a
    with up (au) and down (ad) spin respectively and similar for site b.

    Parameters
    ----------
    t : float, optional
        The hopping parameter, by default 1.0.
    U : float, optional
        The on-site interaction parameter, by default 8.0.
    mu : float, optional
        The chemical potential, by default 0.0.
    coordinations : tuple[int, int], optional
        The coordinations of the sites, by default (1, 1). If applying this
        local operator to every edge in a graph, then the single site
        contributions can be properly accounted for if the coordinations are
        provided.
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array
        The local operator in dense form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")
    bu = FermionicOperator("bu")
    bd = FermionicOperator("bd")

    basis_a = ((), (ad.dag,), (au.dag,), (au.dag, ad.dag))
    basis_b = ((), (bd.dag,), (bu.dag,), (bu.dag, bd.dag))
    bases = [basis_a, basis_b]

    terms = [
        (-t, (au.dag, bu)),
        (-t, (bu.dag, au)),
        (-t, (ad.dag, bd)),
        (-t, (bd.dag, ad)),
        # U, mu are single site and will be overcounted without coordinations
        (U / coordinations[0], (au.dag, au, ad.dag, ad)),
        (U / coordinations[1], (bu.dag, bu, bd.dag, bd)),
        (-mu / coordinations[0], (au.dag, au)),
        (-mu / coordinations[0], (ad.dag, ad)),
        (-mu / coordinations[1], (bu.dag, bu)),
        (-mu / coordinations[1], (bd.dag, bd)),
    ]

    return compute_local_fermionic_dense(terms, bases, like=like)


def fermi_hubbard_local_tensor(
    symmetry,
    t=1.0,
    U=8.0,
    mu=0.0,
    coordinations=(1, 1),
    like="numpy",
):
    """Construct the fermionic local tensor for the Fermi-Hubbard model. The
    indices are ordered as (a, b, a', b'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively and similar for site b.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    t : float, optional
        The hopping parameter, by default 1.0.
    U : float, optional
        The interaction parameter, by default 8.0.
    mu : float, optional
        The chemical potential, by default 0.0.
    like : str, optional
        The backend to use, by default "numpy".
    """
    if symmetry == "Z2":
        indexmap = [0, 1, 1, 0]
    elif symmetry == "Z2Z2":
        indexmap = [(0, 0), (0, 1), (1, 0), (1, 1)]
    elif symmetry == "U1":
        indexmap = [0, 1, 1, 2]
    elif symmetry == "U1U1":
        indexmap = [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        raise ValueError(f"Invalid symmetry: {symmetry}")

    hij = fermi_hubbard_local_dense(
        t, U, mu, coordinations=coordinations, like=like
    )

    return from_dense(
        hij,
        symmetry=symmetry,
        fermionic=True,
        index_maps=[indexmap] * 4,
        duals=[0, 0, 1, 1],
    )
