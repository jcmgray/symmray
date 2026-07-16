import autoray as ar
import pytest


@pytest.fixture(autouse=True)
def enable_debug():
    from symmray.utils import set_debug

    set_debug(True)


@pytest.fixture
def require_backend():
    def require(backend):
        if not hasattr(ar, "to"):
            pytest.skip("backend conversion requires autoray>=0.9.0")
        module = pytest.importorskip(backend)
        if backend == "jax":
            module.config.update("jax_enable_x64", True)

    return require


@pytest.fixture
def convert_backend(require_backend):
    def convert(x, backend):
        require_backend(backend)
        return ar.to(x, backend)

    return convert
