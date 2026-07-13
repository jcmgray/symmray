import numpy as np
import pytest

import symmray as sr
from symmray.fermionic_local_operators import FermionicOperator

SPINLESS_SYMS = ["Z2", "U1"]
SPINFUL_SYMS = ["Z2", "U1", "Z2Z2", "U1U1"]


def _assert_diagonal(matrix):
    matrix = np.asarray(matrix)
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), atol=1e-12)


def test_fermionic_operator_dag():
    a = FermionicOperator("a")
    assert a.dual is False
    assert a.dag.dual is True
    assert a.dag.dag.dual is False
    assert a.dag.label == a.label


def test_fermionic_operator_empty_label_raises():
    with pytest.raises(ValueError):
        FermionicOperator(())


def test_fermionic_operator_ordering():
    # creation (dual) operators sort left of annihilation operators
    a = FermionicOperator("a")
    assert a.dag < a
    assert not (a < a.dag)


# the worked example from the build_local_fermionic_elements docstring
_EXAMPLE_ELEMENTS = {
    (0, 1, 1, 0): -1.0,
    (1, 0, 0, 1): -1.0,
    (1, 1, 1, 1): -8.0,
}


def _example_terms_bases():
    a, b = map(FermionicOperator, "ab")
    basis_a = [(), (a.dag,)]
    basis_b = [(), (b.dag,)]
    terms = (
        (-1.0, (a.dag, b)),
        (-1.0, (b.dag, a)),
        (8.0, (a.dag, a, b.dag, b)),
    )
    return terms, (basis_a, basis_b)


def test_build_local_fermionic_elements_example():
    terms, bases = _example_terms_bases()
    els = sr.build_local_fermionic_elements(terms, bases)
    assert dict(els) == _EXAMPLE_ELEMENTS


def test_build_local_fermionic_dense_matches_elements():
    terms, bases = _example_terms_bases()
    dense = sr.build_local_fermionic_dense(terms, bases)
    assert dense.shape == (2, 2, 2, 2)
    expected = np.zeros((2, 2, 2, 2))
    for idx, val in _EXAMPLE_ELEMENTS.items():
        expected[idx] = val
    np.testing.assert_allclose(dense, expected)


def test_build_local_fermionic_array_example():
    # the symmetric array round-trips back to the (sign-laden) dense form
    terms, bases = _example_terms_bases()
    dense = sr.build_local_fermionic_dense(terms, bases)
    a = sr.build_local_fermionic_array(
        terms, bases, "U1", index_maps=[[0, 1], [0, 1]]
    )
    a.check()
    # U1 index map [0, 1] is already charge-sorted, so to_dense order matches
    np.testing.assert_allclose(a.to_dense(), dense)


@pytest.mark.parametrize("symmetry", SPINLESS_SYMS)
def test_get_spinless_charge_indexmap(symmetry):
    assert sr.get_spinless_charge_indexmap(symmetry) == [0, 1]


def test_get_spinless_charge_indexmap_invalid():
    with pytest.raises(ValueError):
        sr.get_spinless_charge_indexmap("Z2Z2")


@pytest.mark.parametrize(
    "symmetry,expected",
    [
        ("Z2", [0, 1, 1, 0]),
        ("U1", [0, 1, 1, 2]),
        ("Z2Z2", [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ("U1U1", [(0, 0), (0, 1), (1, 0), (1, 1)]),
    ],
)
def test_get_spinful_charge_indexmap(symmetry, expected):
    assert sr.get_spinful_charge_indexmap(symmetry) == expected


def test_get_spinful_charge_indexmap_invalid():
    with pytest.raises(ValueError):
        sr.get_spinful_charge_indexmap("Z3")


@pytest.mark.parametrize("symmetry", SPINLESS_SYMS)
def test_fermi_number_spinless(symmetry):
    n = sr.fermi_number_operator_spinless_local_array(symmetry)
    n.check()
    matrix = n.to_dense()
    _assert_diagonal(matrix)
    assert sorted(np.diag(matrix).real) == [0.0, 1.0]


@pytest.mark.parametrize("symmetry", SPINFUL_SYMS)
def test_fermi_number_and_spin_operators_spinful(symmetry):
    ntot = sr.fermi_number_operator_spinful_local_array(symmetry)
    nup = sr.fermi_number_up_local_array(symmetry)
    ndn = sr.fermi_number_down_local_array(symmetry)
    sz = sr.fermi_spin_z_local_array(symmetry)
    for op in (ntot, nup, ndn, sz):
        op.check()

    mtot, mup, mdn, msz = (op.to_dense() for op in (ntot, nup, ndn, sz))
    for matrix in (mtot, mup, mdn, msz):
        _assert_diagonal(matrix)

    assert sorted(np.diag(mtot).real) == [0, 1, 1, 2]
    assert sorted(np.diag(mup).real) == [0, 0, 1, 1]
    assert sorted(np.diag(mdn).real) == [0, 0, 1, 1]
    assert sorted(np.round(np.diag(msz).real, 10)) == [-0.5, 0.0, 0.0, 0.5]

    np.testing.assert_allclose(mup + mdn, mtot)
    np.testing.assert_allclose(0.5 * (mup - mdn), msz)


def test_fermi_spin_z_old_name_alias():
    assert sr.fermi_spin_operator_local_array is sr.fermi_spin_z_local_array


@pytest.mark.parametrize("symmetry", ["Z2", "U1"])
def test_fermi_spin_plus_minus_local_array(symmetry):
    splus = sr.fermi_spin_plus_local_array(symmetry)
    sminus = sr.fermi_spin_minus_local_array(symmetry)
    sz = sr.fermi_spin_z_local_array(symmetry)
    for op in (splus, sminus):
        op.check()

    expected_plus = np.zeros((4, 4))
    expected_minus = np.zeros((4, 4))
    if symmetry == "Z2":
        expected_plus[2, 3] = 1.0
        expected_minus[3, 2] = 1.0
    else:
        expected_plus[1, 2] = 1.0
        expected_minus[2, 1] = 1.0

    mplus = splus.to_dense()
    mminus = sminus.to_dense()
    np.testing.assert_allclose(mplus, expected_plus)
    np.testing.assert_allclose(mminus, expected_minus)
    np.testing.assert_allclose(splus.H.to_dense(), mminus)
    np.testing.assert_allclose(
        mplus @ mminus - mminus @ mplus,
        2 * sz.to_dense(),
    )


@pytest.mark.parametrize("symmetry", ["Z2Z2", "U1U1"])
@pytest.mark.parametrize(
    "fn",
    [sr.fermi_spin_plus_local_array, sr.fermi_spin_minus_local_array],
)
def test_fermi_spin_plus_minus_reject_flavor_symmetry(symmetry, fn):
    with pytest.raises(ValueError):
        fn(symmetry)


@pytest.mark.parametrize("symmetry", SPINFUL_SYMS)
def test_fermi_double_occupancy_local_array(symmetry):
    double = sr.fermi_double_occupancy_local_array(symmetry)
    double.check()
    matrix = double.to_dense()

    expected = np.zeros((4, 4))
    expected[1 if symmetry == "Z2" else 3, 1 if symmetry == "Z2" else 3] = 1
    np.testing.assert_allclose(matrix, expected)
    np.testing.assert_allclose(matrix @ matrix, matrix)


@pytest.mark.parametrize("flat", [False, True])
def test_fermi_pairing_onsite_local_array(flat):
    pair = sr.fermi_pairing_onsite_local_array("Z2", flat=flat)
    pair.check()

    expected = np.zeros((4, 4))
    expected[0, 1] = 1.0
    np.testing.assert_allclose(pair.to_dense(), expected)
    np.testing.assert_allclose(pair.H.to_dense(), expected.T)


@pytest.mark.parametrize("symmetry", ["U1", "Z2Z2", "U1U1"])
def test_fermi_pairing_onsite_rejects_non_z2(symmetry):
    with pytest.raises(ValueError):
        sr.fermi_pairing_onsite_local_array(symmetry)


@pytest.mark.parametrize("flat", [False, True])
def test_fermi_pairing_bond_local_array(flat):
    pair = sr.fermi_pairing_bond_local_array("Z2", flat=flat)
    pair.check()

    expected = np.zeros((4, 4, 4, 4))
    expected[0, 0, 2, 3] = -1.0
    expected[0, 0, 3, 2] = 1.0
    expected[0, 2, 2, 1] = -1.0
    expected[0, 3, 3, 1] = -1.0
    expected[2, 0, 1, 2] = -1.0
    expected[2, 3, 1, 1] = -1.0
    expected[3, 0, 1, 3] = -1.0
    expected[3, 2, 1, 1] = 1.0
    np.testing.assert_allclose(pair.to_dense(), expected)
    np.testing.assert_allclose(pair.H.H.to_dense(), expected)


@pytest.mark.parametrize("symmetry", ["U1", "Z2Z2", "U1U1"])
def test_fermi_pairing_bond_rejects_non_z2(symmetry):
    with pytest.raises(ValueError):
        sr.fermi_pairing_bond_local_array(symmetry)


def test_fermi_number_operators_flat_z2():
    # flat backend only supports Z2
    nup = sr.fermi_number_up_local_array("Z2", flat=True)
    nup.check()
    assert type(nup).__name__.endswith("Flat")
    assert sorted(np.diag(nup.to_dense()).real) == [0, 0, 1, 1]

    double = sr.fermi_double_occupancy_local_array("Z2", flat=True)
    double.check()
    assert type(double).__name__.endswith("Flat")


@pytest.mark.parametrize("symmetry", SPINFUL_SYMS)
def test_fermi_hubbard_local_array_builds(symmetry):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = sr.fermi_hubbard_local_array(symmetry, t=1.0, U=8.0, mu=0.5)
    h.check()
    assert h.ndim == 4
    assert h.shape == (4, 4, 4, 4)


@pytest.mark.parametrize("symmetry", SPINLESS_SYMS)
def test_fermi_hubbard_spinless_local_array_builds(symmetry):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # delta=0: pairing only conserves Z2 (see *_pairing_z2)
        h = sr.fermi_hubbard_spinless_local_array(
            symmetry, t=1.0, V=2.0, mu=0.5
        )
    h.check()
    assert h.ndim == 4
    assert h.shape == (2, 2, 2, 2)


def test_fermi_hubbard_spinless_pairing_z2():
    import warnings

    # the superconducting pairing term only conserves Z2 parity (not U1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        h = sr.fermi_hubbard_spinless_local_array(
            "Z2", t=1.0, V=2.0, mu=0.5, delta=0.3
        )
    h.check()
    assert h.shape == (2, 2, 2, 2)


def test_fermi_hubbard_local_array_flat_z2():
    h = sr.fermi_hubbard_local_array("Z2", flat=True)
    h.check()
    assert type(h).__name__.endswith("Flat")


@pytest.mark.parametrize("symmetry", SPINFUL_SYMS)
def test_ham_fermi_hubbard_from_edges(symmetry):
    edges = [(0, 1), (1, 2)]
    terms = sr.ham_fermi_hubbard_from_edges(symmetry, edges, t=1.0, U=8.0)
    assert set(terms) == set(edges)
    for term in terms.values():
        term.check()
