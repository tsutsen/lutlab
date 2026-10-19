import numpy as np
import pytest

from lutlab import testing


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(params=["uniform", "rectilinear"])
def lut3(request):
    if request.param == "uniform":
        return testing.random_lut(11, 3, seed=3)
    axes = [np.linspace(0, 1, s) ** 1.6 for s in (7, 9, 11)]
    return testing.random_lut((7, 9, 11), 3, seed=3, axes=axes)
