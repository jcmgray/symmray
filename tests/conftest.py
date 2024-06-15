import pytest


@pytest.fixture(autouse=True)
def enable_debug():
    from symmray.utils import set_debug

    set_debug(True)
