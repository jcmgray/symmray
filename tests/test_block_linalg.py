import pytest

import symmray as sr


@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_qr_basics(d0, d1, f0, f1, c):
    x = sr.utils.get_rand_u1array((d0, d1), flows=[f0, f1], charge_total=c)
    x.check()
    q, r = sr.linalg.qr(x)
    q.check()
    r.check()
    assert sr.tensordot(q, r, 1).allclose(x)


@pytest.mark.parametrize("d0", [3, 4])
@pytest.mark.parametrize("d1", [2, 5])
@pytest.mark.parametrize("f0", [False, True])
@pytest.mark.parametrize("f1", [False, True])
@pytest.mark.parametrize("c", [0, 1])
def test_svd_basics(d0, d1, f0, f1, c):
    x = sr.utils.get_rand_u1array((d0, d1), flows=[f0, f1], charge_total=c)
    x.check()
    u, s, vh = sr.linalg.svd(x)
    u.check()
    vh.check()
