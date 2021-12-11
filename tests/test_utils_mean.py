import numpy as np
from pyriemann_qiskit.utils.mean import fro_mean_convex
from pyriemann.utils.mean import mean_euclid


def test_mean_convex_vs_euclid(get_covmats):
    """Test the shape of mean"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = fro_mean_convex(covmats)
    C_euclid = mean_euclid(covmats)
    assert np.allclose(C, C_euclid, atol=0.0001)


def test_mean_convex_shape(get_covmats):
    """Test the shape of mean"""
    n_trials, n_channels = 5, 3
    covmats = get_covmats(n_trials, n_channels)
    C = fro_mean_convex(covmats)
    assert C.shape == (n_channels, n_channels)


def test_mean_convex_all_zeros():
    """Test the shape of mean"""
    n_trials, n_channels = 5, 2
    covmats = np.zeros((n_trials, n_channels, n_channels))
    C = fro_mean_convex(covmats)
    assert np.allclose(covmats[0], C, atol=0.0001)


def test_mean_convex_all_ones():
    """Test the shape of mean"""
    n_trials, n_channels = 5, 2
    covmats = np.ones((n_trials, n_channels, n_channels))
    C = fro_mean_convex(covmats)
    assert np.allclose(covmats[0], C, atol=0.0001)


def test_mean_convex_all_equals():
    """Test the shape of mean"""
    n_trials, n_channels, value = 5, 2, 2.5
    covmats = np.full((n_trials, n_channels, n_channels), value)
    C = fro_mean_convex(covmats)
    assert np.allclose(covmats[0], C, atol=0.0001)


def test_mean_convex_mixed():
    """Test the shape of mean"""
    n_trials, n_channels = 5, 2
    covmats_0 = np.zeros((n_trials, n_channels, n_channels))
    covmats_1 = np.ones((n_trials, n_channels, n_channels))
    expected_mean = np.full((n_channels, n_channels), 0.5)
    C = fro_mean_convex(np.concatenate((covmats_0, covmats_1), axis=0))
    assert np.allclose(expected_mean, C, atol=0.0001)
