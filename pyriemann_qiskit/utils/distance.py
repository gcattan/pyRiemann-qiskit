import numpy as np
from docplex.mp.model import Model
from pyriemann_qiskit.utils.docplex import ClassicalOptimizer, get_global_optimizer
from pyriemann.utils.distance import (
    distance_functions,
    distance_logeuclid,
    distance_euclid,
)
from pyriemann.utils.base import logm
from pyriemann.utils.mean import mean_logeuclid
from typing_extensions import deprecated


@deprecated(
    "logeucl_dist_convex is deprecated and will be removed in 0.3.0; "
    "please use distance_logeuclid_cpm."
)
def logeucl_dist_convex():
    pass


def distance_logeuclid_to_convex_hull_cpm(A, B, optimizer=ClassicalOptimizer()):
    """Log-Euclidean distance to a convex hull of SPD matrices.

    Log-Euclidean distance between a SPD matrix B and the convex hull of a set
    of SPD matrices A [1]_, formulated as a Constraint Programming Model (CPM)
    [2]_.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    distance : float
        Log-Euclidean distance between the SPD matrix B and the convex hull of
        the set of SPD matrices A, defined as the distance between B and the
        matrix of the convex hull closest to matrix B.

    Notes
    -----
    .. versionadded:: 0.2.0


    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        ‘Convex Class Model on Symmetric Positive Definite Manifolds’,
        Image and Vision Computing, 2019.
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    weights = weights_logeuclid_to_convex_hull_cpm(A, B, optimizer)

    # compute nearest matrix and distance
    C = mean_logeuclid(A, weights)
    distance = distance_logeuclid(C, B)

    return distance


def weights_logeuclid_to_convex_hull_cpm(A, B, optimizer=ClassicalOptimizer()):
    """Log-Euclidean weights to a convex hull of SPD matrices.

    Log-Euclidean weights between a SPD matrix B and the convex hull of a set
    of SPD matrices A [1]_, formulated as a Constraint Programming Model (CPM)
    [2]_.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        it returns the optimized weights for the set of SPD matrices A.
        Using these weights, the weighted Log-Euclidean mean of set A
        provides the matrix of the convex hull closest to matrix B.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.2.0
        Add linear constraint on weights (sum = 1)

    References
    ----------
    .. [1] \
        K. Zhao, A. Wiliem, S. Chen, and B. C. Lovell,
        ‘Convex Class Model on Symmetric Positive Definite Manifolds’,
        Image and Vision Computing, 2019.
    .. [2] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    n_matrices, _, _ = A.shape
    matrices = range(n_matrices)

    def log_prod(m1, m2):
        return np.nansum(logm(m1).flatten() * logm(m2).flatten())

    prob = Model()
    optimizer = get_global_optimizer(optimizer)
    # should be part of the optimizer
    w = optimizer.get_weights(prob, matrices)

    wtLogAtLogAw = prob.sum(
        w[i] * w[j] * log_prod(A[i], A[j]) for i in matrices for j in matrices
    )
    wLogBLogA = prob.sum(w[i] * log_prod(B, A[i]) for i in matrices)
    objectives = wtLogAtLogAw - 2 * wLogBLogA

    prob.set_objective("min", objectives)
    prob.add_constraint(prob.sum(w) == 1)
    weights = optimizer.solve(prob, reshape=False)

    return weights


def weights_distance_cpm(
    A, B, distance=distance_logeuclid, optimizer=ClassicalOptimizer()
):
    """`distance` weights between a SPD and a set of SPD matrices.

    `distance` weights between a SPD matrix B and each SPD matrix inside A,
    formulated as a Constraint Programming Model (CPM)
    [1]_.
    The higher weight corresponds to the closer SPD matrix inside A,
    which is closer to B.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    distance: Callable[[ndarray, ndarray], float]
        One of the pyRiemann distance
    optimizer: pyQiskitOptimizer
      An instance of pyQiskitOptimizer.

    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        it returns the optimized weights for the set of SPD matrices A.
        The higher weight corresponds to the closer SPD matrix inside A,
        which is closer to B

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    n_matrices, _, _ = A.shape
    matrices = range(n_matrices)

    prob = Model()
    optimizer = get_global_optimizer(optimizer)

    w = optimizer.get_weights(prob, matrices)

    objectif = prob.sum(w[i] * distance(B, A[i]) for i in matrices)

    prob.set_objective("min", objectif)
    prob.add_constraint(prob.sum(w) == 1)
    weights = optimizer.solve(prob, reshape=False)

    return weights


def is_cpm_dist(string):
    """Indicates if the distance is a CPM distance.

    Return True is "string" represents a Constraint Programming Model (CPM) [1]_
    distance available in the library.

    Parameters
    ----------
    string: str
        A string representation of the distance.

    Returns
    -------
    is_cpm_dist : boolean
        True if "string" represents a CPM distance available in the library.

    Notes
    -----
    .. versionadded:: 0.2.0

    References
    ----------
    .. [1] \
        http://ibmdecisionoptimization.github.io/docplex-doc/cp/creating_model.html

    """
    return "_cpm" in string and string in distance_functions


distance_functions["logeuclid_hull_cpm"] = weights_logeuclid_to_convex_hull_cpm
distance_functions["euclid_cpm"] = lambda A, B: weights_distance_cpm(
    A, B, distance_euclid
)
distance_functions["logeuclid_cpm"] = lambda A, B: weights_distance_cpm(
    A, B, distance_logeuclid
)
