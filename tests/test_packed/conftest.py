import pytest


@pytest.fixture(params=["numpy", "numba"])
def kernels(request):
    """Run a test with each packed kernel set."""
    from symmray.packed.kernels import get_kernels_name, set_packed_kernels

    if request.param == "numba":
        pytest.importorskip("numba")
    old = get_kernels_name()
    set_packed_kernels(request.param)
    yield request.param
    set_packed_kernels(old)
