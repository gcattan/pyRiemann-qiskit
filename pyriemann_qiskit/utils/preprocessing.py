from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from pyriemann.estimation import Covariances
import numpy as np

class NdRobustScaler(TransformerMixin):
    """Apply one robust scaler by feature.

    RobustScaler of scikit-learn [1]_ is adapted to 3d inputs [2]_.

    References
    ----------
    .. [1] \
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
    .. [2] \
        https://stackoverflow.com/questions/50125844/how-to-standard-scale-a-3d-matrix

    Notes
    -----
    .. versionadded:: 0.2.0
    """

    def __init__(self):
        self._scalers = []

    """Fits one robust scaler on each feature of the training data.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)
        Training matrices.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NdRobustScaler instance
        The NdRobustScaler instance.
    """

    def fit(self, X, _y=None, **kwargs):
        _, n_features, _ = X.shape
        self._scalers = []
        for i in range(n_features):
            scaler = RobustScaler().fit(X[:, i, :])
            self._scalers.append(scaler)
        return self

    """Apply the previously trained robust scalers (on scaler by feature)

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)
        Matrices to scale.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NdRobustScaler instance
        The NdRobustScaler instance.
    """

    def transform(self, X, **kwargs):
        _, n_features, _ = X.shape
        if n_features != len(self._scalers):
            raise ValueError(
                "Input has not the same number of features as the fitted scaler"
            )
        for i in range(n_features):
            X[:, i, :] = self._scalers[i].transform(X[:, i, :])
        return X

class EpochSelectChannel(TransformerMixin):
    """Apply one robust scaler by feature.

    Select channels based on covariance information, 
    keeping only the channel with the maximum covariance.

    Work on signal epochs.

    Notes
    -----
    .. versionadded:: 0.2.0
    """

    def __init__(self, n_components):
        self.n_components = n_components

    """Fits one robust scaler on each feature of the training data.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)
        Training matrices.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : EpochSelectChannel instance
        The EpochSelectChannel instance.
    """

    def fit(self, X, _y=None, **kwargs):
        cov = Covariances()
        covs = cov.fit_transform(X)
        m = np.mean(covs, axis=0)
        features, samples = m.shape
        maxes = []
        for i in range(features):
            for j in range(features):
                if len(maxes) <= self.n_components:
                    maxes.append(m[i, j])
                else:
                    if m[i, j] > max(maxes):
                        maxes[np.argmin(maxes)] = m[i, j]
        indices = []
        for v in maxes:
            w = [w0.tolist() for w0 in np.where(m == v)]
            indices.extend(np.array(w).flatten())
        indices = np.unique(indices)
        import logging
        logging.log(level=logging.CRITICAL, msg=indices)

        self._elec = indices
        return self

    """Apply the previously trained robust scalers (on scaler by feature)

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_features, n_samples)
        Matrices to scale.
    _y : ndarray, shape (n_samples,)
        Unused. Kept for scikit-learn compatibility.

    Returns
    -------
    self : NdRobustScaler instance
        The NdRobustScaler instance.
    """

    def transform(self, X, **kwargs):
        return X[:, self._elec, :]
