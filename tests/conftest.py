import pytest

import faultgate_zero as fz

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _patched():
    """Every test starts with the SDK patches installed, as the plugin leaves them."""
    fz.install()
    yield
    fz.install()
