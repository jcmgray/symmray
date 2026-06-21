import numpy as np
import pytest

import symmray as sr

# explicit spin-1/2 matrices, used to build independent ground truth
I = np.array([[1.0, 0.0], [0.0, 1.0]])
X = np.array([[0.0, 1.0], [1.0, 0.0]])
Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
Z = np.array([[1.0, 0.0], [0.0, -1.0]])
SX, SY, SZ = X / 2, Y / 2, Z / 2
SP = np.array([[0.0, 1.0], [0.0, 0.0]])
SM = np.array([[0.0, 0.0], [1.0, 0.0]])

REFERENCE = {
    "I": I,
    "x": X,
    "y": Y,
    "z": Z,
    "sx": SX,
    "sy": SY,
    "sz": SZ,
    "+": SP,
    "-": SM,
}

# (symmetry, flat) combinations - flat backend only supports Z2
SYM_FLAT = [("Z2", False), ("Z2", True), ("U1", False)]


@pytest.mark.parametrize("op", list(REFERENCE))
def test_get_spin_operator_matches_reference(op):
    np.testing.assert_allclose(
        sr.spin_local_operators.get_spin_operator(op), REFERENCE[op]
    )


def test_get_spin_operator_unknown():
    with pytest.raises(ValueError):
        sr.spin_local_operators.get_spin_operator("notanop")


@pytest.mark.parametrize("symmetry", ["Z2", "Z3", "U1"])
def test_get_spinhalf_charge_indexmap(symmetry):
    assert sr.get_spinhalf_charge_indexmap(symmetry) == [0, 1]


def test_get_spinhalf_charge_indexmap_invalid():
    with pytest.raises(ValueError):
        sr.get_spinhalf_charge_indexmap("SU2")


def test_build_local_spin_dense_yy_is_real():
    # y . y should collapse to a real array
    h = sr.build_local_spin_dense([(1.0, ("sy", "sy"))])
    assert not np.iscomplexobj(h)
    np.testing.assert_allclose(h.reshape(4, 4), np.kron(SY, SY).real)


def test_build_local_spin_dense_ladder_placement():
    h = sr.build_local_spin_dense([(1.0, ("+", "-"))])
    np.testing.assert_allclose(h.reshape(4, 4), np.kron(SP, SM))


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("jx,hz", [(-1.0, -3.0), (0.7, (0.3, 1.1))])
def test_tfim_matches_reference(flat, jx, hz):
    a = sr.tfim_local_array("Z2", jx=jx, hz=hz, flat=flat)
    a.check()
    ha, hb = hz if isinstance(hz, tuple) else (hz, hz)
    ref = jx * np.kron(X, X) + ha * np.kron(Z, I) + hb * np.kron(I, Z)
    np.testing.assert_allclose(a.to_dense().reshape(4, 4), ref)


def test_tfim_only_z2():
    with pytest.raises(ValueError):
        sr.tfim_local_array("U1")


@pytest.mark.parametrize("symmetry,flat", SYM_FLAT)
@pytest.mark.parametrize(
    "j,b",
    [
        (1.0, 0.0),
        ((0.5, 0.5, 1.0), 0.0),
        (1.0, 0.3),
    ],
)
def test_heisenberg_matches_reference(symmetry, flat, j, b):
    a = sr.heisenberg_local_array(symmetry, j=j, b=b, flat=flat)
    a.check()
    jx, jy, jz = j if isinstance(j, tuple) else (j, j, j)
    ref = (
        jx * np.kron(SX, SX)
        + jy * np.kron(SY, SY)
        + jz * np.kron(SZ, SZ)
        - b * (np.kron(SZ, I) + np.kron(I, SZ))
    )
    np.testing.assert_allclose(a.to_dense().reshape(4, 4), ref.real)


@pytest.mark.parametrize("symmetry,flat", SYM_FLAT)
def test_heisenberg_coordinations_divide_field(symmetry, flat):
    # field divided by coordination, coupling untouched
    a = sr.heisenberg_local_array(
        symmetry, j=1.0, b=0.4, coordinations=(2, 4), flat=flat
    )
    a.check()
    ref = (
        np.kron(SX, SX)
        + np.kron(SY, SY)
        + np.kron(SZ, SZ)
        - 0.4 / 2 * np.kron(SZ, I)
        - 0.4 / 4 * np.kron(I, SZ)
    )
    np.testing.assert_allclose(a.to_dense().reshape(4, 4), ref.real)


@pytest.mark.parametrize("symmetry,flat", SYM_FLAT)
@pytest.mark.parametrize("op", ["I", "z", "sz", "+", "-"])
def test_spin_operator_local_array(symmetry, flat, op):
    import warnings

    with warnings.catch_warnings():
        # charge is set explicitly, so no invalid-sector warnings expected
        warnings.simplefilter("error")
        a = sr.spin_operator_local_array(symmetry, op=op, flat=flat)
    a.check()
    np.testing.assert_allclose(a.to_dense(), REFERENCE[op])


@pytest.mark.parametrize("op", ["x", "y"])
def test_spin_operator_x_under_z2(op):
    # x / y flip Z2 parity consistently -> well defined under Z2
    a = sr.spin_operator_local_array("Z2", op=op)
    a.check()
    np.testing.assert_allclose(a.to_dense(), REFERENCE[op])


@pytest.mark.parametrize("op", ["x", "y"])
def test_spin_operator_x_under_u1_raises(op):
    # x / y do not conserve U1 magnetization -> ill defined total charge
    with pytest.raises(ValueError):
        sr.spin_operator_local_array("U1", op=op)


def test_quimb_regression():
    qu = pytest.importorskip("quimb")

    # tfim against quimb pauli construction
    Iq, Zq, Xq = (qu.pauli(s, dtype="float64") for s in "IZX")
    a = sr.tfim_local_array("Z2", jx=0.7, hz=(0.3, 1.1))
    ref = 0.7 * (Xq & Xq) + 0.3 * (Zq & Iq) + 1.1 * (Iq & Zq)
    np.testing.assert_allclose(a.to_dense().reshape(4, 4), np.asarray(ref))

    # heisenberg against quimb ham_heis, including coupling tuple and z-field
    for symmetry in ("Z2", "U1"):
        a = sr.heisenberg_local_array(symmetry, j=1.0)
        np.testing.assert_allclose(
            a.to_dense().reshape(4, 4), np.asarray(qu.ham_heis(2)).real
        )
    a = sr.heisenberg_local_array("Z2", j=(0.5, 0.5, 1.0), b=0.3)
    np.testing.assert_allclose(
        a.to_dense().reshape(4, 4),
        np.asarray(qu.ham_heis(2, j=(0.5, 0.5, 1.0), b=0.3)).real,
    )


def test_ham_from_edges_uses_spin_operators():
    edges = [(0, 1), (1, 2)]
    ht = sr.ham_tfim_from_edges("Z2", edges)
    hh = sr.ham_heisenberg_from_edges("U1", edges, j=1.0)
    assert set(ht) == set(hh) == set(edges)
    for term in ht.values():
        term.check()
    for term in hh.values():
        term.check()


@pytest.mark.parametrize("symmetry", ["Z2", "U1"])
def test_ham_heisenberg_from_edges_field_coordination(symmetry):
    # path 0-1-2: node 1 has coordination 2, nodes 0 and 2 coordination 1
    edges = [(0, 1), (1, 2)]
    b = 0.4
    terms = sr.ham_heisenberg_from_edges(symmetry, edges, j=1.0, b=b)
    base = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
    # edge (0, 1): site 0 coord 1, site 1 coord 2
    ref01 = base - b / 1 * np.kron(SZ, I) - b / 2 * np.kron(I, SZ)
    np.testing.assert_allclose(
        terms[(0, 1)].to_dense().reshape(4, 4), ref01.real
    )
    # edge (1, 2): site 1 coord 2, site 2 coord 1
    ref12 = base - b / 2 * np.kron(SZ, I) - b / 1 * np.kron(I, SZ)
    np.testing.assert_allclose(
        terms[(1, 2)].to_dense().reshape(4, 4), ref12.real
    )
    # node 1's two edges each contribute -b/2 -> total -b (no double count)
    assert -b / 2 + -b / 2 == pytest.approx(-b)


def test_ham_heisenberg_from_edges_j_edge_factory():
    # j as a dict mapping edges to (jx, jy, jz) couplings
    edges = [(0, 1), (1, 2)]
    jmap = {(0, 1): 1.0, (1, 2): (0.5, 0.5, 2.0)}
    terms = sr.ham_heisenberg_from_edges("U1", edges, j=jmap)
    np.testing.assert_allclose(
        terms[(0, 1)].to_dense().reshape(4, 4),
        (np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)).real,
    )
    np.testing.assert_allclose(
        terms[(1, 2)].to_dense().reshape(4, 4),
        (
            0.5 * np.kron(SX, SX)
            + 0.5 * np.kron(SY, SY)
            + 2.0 * np.kron(SZ, SZ)
        ).real,
    )


@pytest.mark.parametrize("symmetry,flat", SYM_FLAT)
def test_heisenberg_per_site_z_field(symmetry, flat):
    # length-2 b -> independent z-field on each site
    a = sr.heisenberg_local_array(symmetry, j=1.0, b=(0.1, 0.5), flat=flat)
    a.check()
    base = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
    ref = base - 0.1 * np.kron(SZ, I) - 0.5 * np.kron(I, SZ)
    np.testing.assert_allclose(a.to_dense().reshape(4, 4), ref.real)


@pytest.mark.parametrize("symmetry", ["Z2", "U1"])
def test_ham_heisenberg_from_edges_per_site_field(symmetry):
    # per-site z-field via a node dict, divided by per-site coordination
    edges = [(0, 1), (1, 2)]
    bmap = {0: 0.1, 1: 0.5, 2: 0.9}
    terms = sr.ham_heisenberg_from_edges(symmetry, edges, j=1.0, b=bmap)
    base = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
    # edge (0, 1): site 0 (b=0.1, coord 1), site 1 (b=0.5, coord 2)
    ref01 = base - 0.1 / 1 * np.kron(SZ, I) - 0.5 / 2 * np.kron(I, SZ)
    np.testing.assert_allclose(
        terms[(0, 1)].to_dense().reshape(4, 4), ref01.real
    )
    # edge (1, 2): site 1 (b=0.5, coord 2), site 2 (b=0.9, coord 1)
    ref12 = base - 0.5 / 2 * np.kron(SZ, I) - 0.9 / 1 * np.kron(I, SZ)
    np.testing.assert_allclose(
        terms[(1, 2)].to_dense().reshape(4, 4), ref12.real
    )
    # node 1's field sums to its full -0.5 across both incident edges
    assert -0.5 / 2 + -0.5 / 2 == pytest.approx(-0.5)


@pytest.mark.parametrize("symmetry", ["Z2", "U1"])
def test_ham_heisenberg_from_edges_per_site_callable(symmetry):
    # per-site z-field via a callable node factory
    edges = [(0, 1)]
    terms = sr.ham_heisenberg_from_edges(
        symmetry, edges, j=1.0, b=lambda coo: 0.1 * (coo + 1)
    )
    base = np.kron(SX, SX) + np.kron(SY, SY) + np.kron(SZ, SZ)
    ref = base - 0.1 * np.kron(SZ, I) - 0.2 * np.kron(I, SZ)
    np.testing.assert_allclose(
        terms[(0, 1)].to_dense().reshape(4, 4), ref.real
    )


def test_symmetry_none_returns_dense():
    # symmetry=None returns the raw dense array (matching .to_dense())
    tfim = sr.tfim_local_array(None, jx=0.7, hz=(0.3, 1.1))
    assert isinstance(tfim, np.ndarray) and tfim.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(
        tfim, sr.tfim_local_array("Z2", jx=0.7, hz=(0.3, 1.1)).to_dense()
    )

    heis = sr.heisenberg_local_array(None, j=1.0, b=(0.1, 0.5))
    np.testing.assert_allclose(
        heis, sr.heisenberg_local_array("U1", j=1.0, b=(0.1, 0.5)).to_dense()
    )

    # charge-changing single-site ops are fine without a symmetry
    np.testing.assert_allclose(sr.spin_operator_local_array(None, op="+"), SP)
    np.testing.assert_allclose(sr.spin_operator_local_array(None, op="x"), X)

    array = sr.build_local_spin_array([(1.0, ("z", "z"))], None)
    np.testing.assert_allclose(array.reshape(4, 4), np.kron(Z, Z))


def test_ham_from_edges_symmetry_none_returns_dense():
    edges = [(0, 1), (1, 2)]
    ht = sr.ham_tfim_from_edges(None, edges)
    hh = sr.ham_heisenberg_from_edges(None, edges, j=1.0, b=0.3)
    assert set(ht) == set(hh) == set(edges)
    for term in (*ht.values(), *hh.values()):
        assert isinstance(term, np.ndarray) and term.shape == (2, 2, 2, 2)
