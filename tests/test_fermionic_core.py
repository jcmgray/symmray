import pytest

import symmray as sr


def get_rand_z2fermionicarray(shape, flows=None, charge_total=0):
    ndim = len(shape)

    if flows is None:
        flows = [i < ndim // 2 for i in range(ndim)]

    x = sr.Z2FermionicArray.random(
        indices=[
            sr.BlockIndex(
                {0: d // 2 + d % 2, 1: d // 2},
                flow=f,
            )
            for d, f in zip(shape, flows)
        ],
        charge_total=charge_total,
    )
    x.phase_flip(1, 3, inplace=True)
    return x


def test_fermi_norm():
    x = get_rand_z2fermionicarray((3, 4, 5, 6))
    assert x.phases
    ne = x.norm()
    xc = x.conj()
    assert xc.phases != x.phases
    xx = sr.tensordot(x, xc, axes=4)
    n1 = float(xx) ** 0.5
    assert ne == pytest.approx(n1)
    xx = sr.tensordot(xc, x, axes=4)
    n2 = float(xx) ** 0.5
    assert ne == pytest.approx(n2)
    xd = x.dagger()
    assert xd.phases != x.phases
    assert xd.phases != xc.phases
    xx = sr.tensordot(xd, x, axes=[(3, 2, 1, 0), (0, 1, 2, 3)])
    n3 = float(xx) ** 0.5
    assert ne == pytest.approx(n3)
    xx = sr.tensordot(x, xd, axes=[(0, 1, 2, 3), (3, 2, 1, 0)])
    n4 = float(xx) ** 0.5
    assert ne == pytest.approx(n4)
