from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
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


class Vectorizer(BaseEstimator, TransformerMixin):
    """Vectorization.

    This is an auxiliary transformer that allows one to vectorize data
    structures in a pipeline. For instance, in the case of an X with
    dimensions (n_samples, n_features, n_channels),
    one might be interested in a new data structure with dimensions
    (n_samples, n_features x n_channels)

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.3.0
        Move from filtering to preprocessing module.
        Fix documentation.

    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_features, n_times)
            Training matrices.
        y : ndarray, shape (n_trials,) (default: None)
            Target vector relative to X.
            In practice, never used.

        Returns
        -------
        self : Vectorizer
            The Vectorizer instance.
        """
        return self

    def transform(self, X):
        """Vectorize matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_features, n_times)
            The matrices to vectorize.

        Returns
        -------
        X : ndarray, shape (n_trials, n_features x n_times)
            The vectorized matrices.
        """
        return np.reshape(X, (X.shape[0], -1))


class Devectorizer(TransformerMixin):
    """Transform vector to matrices

    This is an auxiliary transformer that allows one to
    transform vectors of shape (n_feats x n_times)
    into matrices of size (n_feats, n_times)

    Parameters
    ----------
    n_feats : int
        The number of features of the matrices.
    n_times : int
        The number of samples of the matrices.

    Notes
    -----
    .. versionadded:: 0.0.1
    .. versionchanged:: 0.3.0
        Move from filtering to preprocessing module.
        Fix documentation.

    """

    def __init__(self, n_feats, n_times):
        self.n_feats = n_feats
        self.n_times = n_times

    def fit(self, X, y=None):
        """Fit the training data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_features x n_times)
            Training matrices.
        y : ndarray, shape (n_trials,) (default: None)
            Target vector relative to X.
            In practice, never used.

        Returns
        -------
        self : Devectorizer
            The Devectorizer instance.
        """
        return self

    def transform(self, X, y=None):
        """Transform vectors into matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_features x n_times)
            The vectors.

        Returns
        -------
        X : ndarray, shape (n_trials, n_features, n_times)
            The matrices.
        """

        n_trial, _ = X.shape
        return X.reshape((n_trial, self.n_feats, self.n_times))
