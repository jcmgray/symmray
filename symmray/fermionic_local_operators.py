"""Helper functions for building local fermionic operators, with 'internal'
signs pre-computed.
"""

import itertools
import numbers

import autoray as ar


def labels_lt(labela, labelb):
    try:
        return labela < labelb
    except TypeError:
        # allow certain mixed types to be compared, the logic here is:
        # 1. group all integer labels first
        # 2. then plain strings
        # 3. then tuples by length
        # XXX: some different length tuples can be compared, might revisit
        if isinstance(labela, numbers.Integral):
            la = -2
        elif isinstance(labela, str):
            la = -1
        else:
            la = len(labela)

        if isinstance(labelb, numbers.Integral):
            lb = -2
        elif isinstance(labelb, str):
            lb = -1
        else:
            lb = len(labelb)

        return la < lb


class FermionicOperator:
    """Simple class to represent a fermionic operator with a label and a
    dual flag.
    """

    __slots__ = ("_label", "_dual")

    def __init__(self, label, dual=False):
        if label == ():
            raise ValueError("Label cannot be an empty tuple.")
        self._label = label
        self._dual = dual

    @property
    def label(self):
        return self._label

    @property
    def dual(self):
        return self._dual

    @property
    def dag(self):
        return FermionicOperator(self._label, not self._dual)

    def __eq__(self, other):
        return (self._dual, self._label) == (other._dual, other._label)

    def __lt__(self, other):
        if self.dual:
            if other.dual:
                # dual operator are reflected
                return labels_lt(other.label, self.label)
            else:
                # creation left of annihilation
                return True
        else:
            if other.dual:
                return False
            else:
                return labels_lt(self.label, other.label)

    def __repr__(self):
        return f"{self._label}{'+' if self._dual else '-'}"


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


def build_local_fermionic_elements(terms, bases):
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

        build_local_fermionic_elements(terms, bases)
        # {(0, 1, 1, 0): -1.0, (1, 0, 0, 1): -1.0, (1, 1, 1, 1): -8.0}

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

    entries = {}
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
                    and all(not op.dual for op in group[::2])
                    and all(op.dual for op in group[1::2])
                )
                for group in groups.values()
            )

            if nonvanishing:
                # a non-vanishing element
                index = (*left_indices, *right_indices)
                entries[index] = entries.get(index, 0.0) + phase * coeff

    return entries


def build_local_fermionic_dense(terms, bases, like="numpy"):
    hij = ar.do("zeros", tuple(len(b) for b in bases) * 2, like=like)

    for idx, val in build_local_fermionic_elements(terms, bases).items():
        hij[idx] += val

    return hij


def build_local_fermionic_array(
    terms,
    bases,
    symmetry,
    index_maps,
    like="numpy",
):
    """Compute a local fermionic operator as a `FermionicArray`.

    Parameters
    ----------
    terms : tuple[tuple[float, tuple[FermionicOperator, ...]]]
        The terms in the operator, each a tuple of a coefficient and a tuple
        of FermionicOperator instances.
    bases : tuple[tuple[tuple[FermionicOperator]]]
        The tensor bases to compute the operator elements in. Each basis is a
        sequence of multiple FermionicOperator instances acting on the vacuum.
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2" or "U1U1".
    index_maps : Sequence[Sequence[hashable]]
        For each basis, the sequence mapping linear index to charge sector.

    Returns
    -------
    FermionicArray
        The local operator in fermionic array form.
    """
    from .utils import from_dense

    dense = build_local_fermionic_dense(terms, bases, like=like)
    duals = [False] * len(bases) + [True] * len(bases)

    return from_dense(
        dense,
        duals=duals,
        symmetry=symmetry,
        fermionic=True,
        index_maps=index_maps * 2,
    )


def get_spinless_charge_indexmap(symmetry):
    """Get a mapping of linear index to charge sector for a spinless
    fermion model.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2" or "U1".

    Returns
    -------
    list[hashable]
    """
    if symmetry in ("Z2", "U1"):
        return [0, 1]
    else:
        raise ValueError(f"Invalid symmetry: {symmetry}")


def get_spinful_charge_indexmap(symmetry):
    """Get a mapping of linear index to charge sector for a spinful
    fermion model.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".

    Returns
    -------
    list[hashable]
    """
    if symmetry == "Z2":
        return [0, 1, 1, 0]
    elif symmetry == "U1":
        return [0, 1, 1, 2]
    elif symmetry in ("Z2Z2", "U1U1"):
        return [(0, 0), (0, 1), (1, 0), (1, 1)]
    else:
        raise ValueError(f"Invalid symmetry: {symmetry}")


def fermi_hubbard_spinless_local_array(
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
    mu : float or (float, float), optional
        The chemical potential, by default 0.0. If a tuple, then the chemical
        potential is different for each site.
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
    a, b = map(FermionicOperator, "ab")

    try:
        mua, mub = mu
    except TypeError:
        mua = mub = mu

    terms = (
        # hopping
        (-t, (a.dag, b)),
        (-t, (b.dag, a)),
        # nearest-neighbor interaction
        (V, (a.dag, a, b.dag, b)),
        # chemical potential
        # mu is single site and will be overcounted without coordinations
        (-mua / coordinations[0], (a.dag, a)),
        (-mub / coordinations[1], (b.dag, b)),
    )

    basis_a = ((), (a.dag,))
    basis_b = ((), (b.dag,))
    bases = (basis_a, basis_b)
    indexmap = get_spinless_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap, indexmap],
        like=like,
    )


def fermi_hubbard_local_array(
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
    U : float or (float, float), optional
        The interaction parameter, by default 8.0. If a tuple, then the
        interaction parameter is different for each site.
    mu : float or (float, float), optional
        The chemical potential, by default 0.0. If a tuple, then the chemical
        potential is different for each site.
    coordinations : tuple[int, int], optional
        The coordinations of the sites, by default (1, 1). If applying this
        local operator to every edge in a graph, then the single site
        contributions can be properly accounted for if the coordinations are
        provided.
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")
    bu = FermionicOperator("bu")
    bd = FermionicOperator("bd")

    try:
        Ua, Ub = U
    except TypeError:
        Ua = Ub = U

    try:
        mua, mub = mu
    except TypeError:
        mua = mub = mu

    terms = [
        (-t, (au.dag, bu)),
        (-t, (bu.dag, au)),
        (-t, (ad.dag, bd)),
        (-t, (bd.dag, ad)),
        # U, mu are single site and will be overcounted without coordinations
        (Ua / coordinations[0], (au.dag, au, ad.dag, ad)),
        (Ub / coordinations[1], (bu.dag, bu, bd.dag, bd)),
        (-mua / coordinations[0], (au.dag, au)),
        (-mua / coordinations[0], (ad.dag, ad)),
        (-mub / coordinations[1], (bu.dag, bu)),
        (-mub / coordinations[1], (bd.dag, bd)),
    ]

    basis_a = ((), (ad.dag,), (au.dag,), (au.dag, ad.dag))
    basis_b = ((), (bd.dag,), (bu.dag,), (bu.dag, bd.dag))
    bases = [basis_a, basis_b]
    indexmap = get_spinful_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap, indexmap],
        like=like,
    )


def fermi_number_operator_spinless_local_array(symmetry, like="numpy"):
    """Construct the fermionic number operator for the spinless Fermi-Hubbard
    model. The indices are ordered as (a, a'). The local basis is like
    (|0>, a+|0>) for single site a.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2" or "U1".
    like : str, optional

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    a = FermionicOperator("a")

    terms = [(1, (a.dag, a))]
    bases = [((), (a.dag,))]
    indexmap = get_spinless_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap],
        like=like,
    )


def fermi_number_operator_spinful_local_array(symmetry, like="numpy"):
    """Construct the fermionic number operator for the Fermi-Hubbard model. The
    indices are ordered as (a, a'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively for single site `a`.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")

    # nup + ndown
    terms = [(1, (au.dag, au)), (1, (ad.dag, ad))]
    # |00>, |01>, |10>, |11>
    bases = [((), (ad.dag,), (au.dag,), (au.dag, ad.dag))]
    indexmap = get_spinful_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap],
        like=like,
    )


def fermi_number_up_local_array(symmetry, like="numpy"):
    """Construct the 'up' fermionic number operator for the Fermi-Hubbard
    model. The indices are ordered as (a, a'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively for single site `a`.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")

    # nup
    terms = [(1, (au.dag, au))]
    # |00>, |01>, |10>, |11>
    bases = [((), (ad.dag,), (au.dag,), (au.dag, ad.dag))]
    indexmap = get_spinful_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap],
        like=like,
    )


def fermi_number_down_local_array(symmetry, like="numpy"):
    """Construct the 'down' fermionic number operator for the Fermi-Hubbard
    model. The indices are ordered as (a, a'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively for single site `a`.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")

    # nup
    terms = [(1, (ad.dag, ad))]
    # |00>, |01>, |10>, |11>
    bases = [((), (ad.dag,), (au.dag,), (au.dag, ad.dag))]
    indexmap = get_spinful_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap],
        like=like,
    )


def fermi_spin_operator_local_array(symmetry, like="numpy"):
    """Construct the fermionic spin operator for the Fermi-Hubbard model. The
    indices are ordered as (a, a'), with the local basis like
    (|00>, ad+|00>, au+|00>, au+ad+|00>) for site a with up (au) and down (ad)
    spin respectively for single site `a`.

    Parameters
    ----------
    symmetry : str
        The symmetry of the model. Either "Z2", "U1", "Z2Z2", or "U1U1".
    like : str, optional
        The backend to use, by default "numpy".

    Returns
    -------
    array : FermionicArray
        The local operator in fermionic array form.
    """
    au = FermionicOperator("au")
    ad = FermionicOperator("ad")

    # S^z = 1/2 (nup - ndown)
    terms = [(0.5, (au.dag, au)), (-0.5, (ad.dag, ad))]
    # |00>, |01>, |10>, |11>
    bases = [((), (ad.dag,), (au.dag,), (au.dag, ad.dag))]
    indexmap = get_spinful_charge_indexmap(symmetry)

    return build_local_fermionic_array(
        terms,
        bases,
        symmetry,
        index_maps=[indexmap],
        like=like,
    )
