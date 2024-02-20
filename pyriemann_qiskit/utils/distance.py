import numpy as np
from docplex.mp.model import Model
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, get_global_optimizer
from pyriemann.classification import MDM
from pyriemann.utils.distance import distance_functions
from pyriemann.utils.base import logm


def logeucl_dist_cpm(X, y, optimizer=ClassicalOptimizer()):
    """Log-Euclidean distance [1]_ by Constraint Programming Model [2]_.

    Constraint Programming Model (CPM) formulation of the Log-Euclidean distance.

    Parameters
    ----------
    X : ndarray, shape (n_classes, n_channels, n_channels)
        Set of SPD matrices.
    y : ndarray, shape (n_channels, n_channels)
        A trial
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    weights : ndarray, shape (n_classes,)
        The weights associated with each class.
        Higher the weight, closer it is to the class prototype.
        Weights are not normalized.

    Notes
    -----
    .. versionadded:: 0.0.4

    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        ‘Convex Class Model on Symmetric Positive Definite Manifolds’, arXiv:1806.05343 [cs], May 2019.
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/mp/_modules/docplex/mp/model.html#Model

    """

    optimizer = get_global_optimizer(optimizer)

    n_classes, _, _ = X.shape
    classes = range(n_classes)

    def log_prod(m1, m2):
        return np.nansum(logm(m1).flatten() * logm(m2).flatten())

    prob = Model()

    # should be part of the optimizer
    w = optimizer.get_weights(prob, classes)

    _2VecLogYD = 2 * prob.sum(w[i] * log_prod(y, X[i]) for i in classes)

    wtDw = prob.sum(
        w[i] * w[j] * log_prod(X[i], X[j]) for i in classes for j in classes
    )

    objectives = wtDw - _2VecLogYD

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob, reshape=False)

    return 1 - result


_mdm_predict_distances_original = MDM._predict_distances


def predict_distances(mdm, X):
    if mdm.metric_dist == "cpm_le":
        centroids = np.array(mdm.covmeans_)
        return np.array([logeucl_dist_cpm(centroids, x) for x in X])
    else:
        return _mdm_predict_distances_original(mdm, X)


MDM._predict_distances = predict_distances

# This is only for validation inside the MDM.
# In fact, we override the _predict_distances method
# inside MDM to directly use logeucl_dist_cpm when the metric is "cpm_le"
# This is due to the fact the the signature of this method is different from
# the usual distance functions.
distance_functions["cpm_le"] = logeucl_dist_cpm
