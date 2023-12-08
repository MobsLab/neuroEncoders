### KernelDensity Estimation Tests: KernelDensity is very long to fit

import pytest
import pathlib, sys, shutil
import numpy as np
from sklearn.preprocessing import OneHotEncoder

rootFolder = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(rootFolder))
import simpleBayes.butils as butils

# Test till 4D data
# Do the factory pattern for the data
@pytest.fixture(scope='class')
def data_1d(request):
    request.cls.data = np.random.lognormal(size=100)

@pytest.fixture(scope='class')
def data_2d(request):
    request.cls.data = np.random.lognormal(size=(100,2))

@pytest.fixture(scope='class')
def data_3d(request):
    request.cls.data = np.random.lognormal(size=(100,4))

@pytest.fixture(scope='class')
def data_4d(request):
    request.cls.data = np.random.lognormal(size=(100,4))

@pytest.mark.usefixtures("data_1d", "data_2d", "data_3d", "data_4d")
class TestKernels:
    def test_gaussian_kernel_shape(self):
        kernel_est, _ = butils.kdenD(self.data, bandwidth=0.05, kernel='gaussian')
        assert self.data.shape[1] == len(kernel_est.shape)

    def test_epanechnikov_kernel_shape(self):
        kernel_est, _ = butils.kdenD(self.data, bandwidth=0.01, kernel='epanechnikov')
        assert self.data.shape[1] == len(kernel_est.shape)

    def test_gives_right_number_of_bins(self):
        kernel_est, _ = butils.kdenD(self.data, bandwidth=0.05, nbins=10)
        assert np.unique(kernel_est.shape) == 10

    def test_gives_some_non_zero_output(self):
        kernel_est, _ = butils.kdenD(self.data, bandwidth=0.05)
        assert np.any(kernel_est)

    def test_gives_sum_of_1(self):
        kernel_est, _ = butils.kdenD(self.data, bandwidth=0.05)
        assert np.isclose(np.sum(kernel_est), 1)

