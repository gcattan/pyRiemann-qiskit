"""Module for mathematical helpers"""

from pyriemann.utils.covariance import normalize
import numpy as np


def cov_to_corr_matrix(covmat):
    """Convert covariance matrices to correlation matrices.

    Parameters
    ----------
    covmat: ndarray, shape (..., n_channels, n_channels)
        Covariance matrices.

    Returns
    -------
    corrmat : ndarray, shape (..., n_channels, n_channels)
        Correlation matrices.

    Notes
    -----
    .. versionadded:: 0.0.2
    """
    return normalize(covmat, "corr")


def union_of_diff(*arrays):
    """Return the positions for which at least one of the array
    as a different value than the others.

    e.g.:

    A = 0 1 0
    B = 0 1 1
    C = 1 1 0

    return
    A = True False True

    Parameters
    ----------
    arrays: ndarray[], shape (n_samples,)[]
        A list of numpy arrays.

    Returns
    -------
    diff : ndarray, shape (n_samples,)
        A list of boolean.
        True at position i indicates that one of the array
        as a value different from the other ones at this
        position.

    Notes
    -----
    .. versionadded:: 0.2.0
    """
    size = len(arrays[0])
    for array in arrays:
        assert len(array) == size

    diff = [False] * size
    for i in range(size):
        s = set({array[i] for array in arrays})
        if len(s) > 1:
            diff[i] = True

    return np.array(diff)
