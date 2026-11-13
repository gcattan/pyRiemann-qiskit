import numpy as np
from docplex.mp.model import Model
from pyriemann.utils.base import logm
from pyriemann.utils.distance import (
    distance_euclid,
    distance_functions,
    distance_logeuclid,
)
from pyriemann.utils.mean import mean_logeuclid

from .docplex import ClassicalOptimizer, NaiveQAOAOptimizer


def qdistance_logeuclid_to_convex_hull(A, B, optimizer=ClassicalOptimizer()):
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
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of
        :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

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
    weights = weights_logeuclid_to_convex_hull(A, B, optimizer)
    # compute nearest matrix
    C = mean_logeuclid(A, weights)
    distance = distance_logeuclid(C, B)

    return distance


def weights_logeuclid_to_convex_hull(A, B, optimizer=ClassicalOptimizer()):
    """Weights for Log-Euclidean distance to a convex hull of SPD matrices.

    Weights for Log-Euclidean distance between a SPD matrix B
    and the convex hull of a set of SPD matrices A [1]_,
    formulated as a Constraint Programming Model (CPM) [2]_.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of
        :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        Optimized weights for the set of SPD matrices A.
        Using these weights, the weighted Log-Euclidean mean of set A
        provides the matrix of the convex hull closest to matrix B.

    Notes
    -----
    .. versionadded:: 0.0.4
    .. versionchanged:: 0.2.0
        Rename from `logeucl_dist_convex` to `weights_logeuclid_to_convex_hull`.
        Add linear constraint on weights (sum = 1).

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

    def trace_prod_log(m1, m2):
        return np.trace(logm(m1) @ logm(m2))

    prob = Model()
    raw = optimizer.get_weights(prob, matrices)
    # Integer encodings (NaiveQAOA) represent the weights on a grid of step
    # 1 / upper_bound: the integers sum to upper_bound and the objective uses
    # them divided by upper_bound. With "sum == 1" on integers, the only
    # feasible points would be the vertices of the hull.
    bound = optimizer.upper_bound if isinstance(optimizer, NaiveQAOAOptimizer) else None
    w = raw if bound is None else [raw[i] / bound for i in matrices]

    wtLogAtLogAw = prob.sum(
        w[i] * w[j] * trace_prod_log(A[i], A[j]) for i in matrices for j in matrices
    )
    wLogBLogA = prob.sum(w[i] * trace_prod_log(B, A[i]) for i in matrices)
    objective = wtLogAtLogAw - 2 * wLogBLogA

    prob.set_objective("min", objective)
    if bound is None:
        prob.add_constraint(prob.sum(w) == 1)
    else:
        prob.add_constraint(prob.sum(raw) == bound)

    weights = optimizer.solve(prob, reshape=False)
    if bound is not None:
        weights = np.asarray(weights, dtype=float) / bound
        if weights.sum() <= 0:  # infeasible all-zero solution: no information
            weights = np.full(n_matrices, 1.0 / n_matrices)

    return weights


def _weights_distance(
    A, B, distance=distance_logeuclid, optimizer=ClassicalOptimizer()
):
    """`distance` weights between a SPD and a set of SPD matrices.

    `distance` weights between a SPD matrix B and each SPD matrix inside A,
    formulated as a Constraint Programming Model (CPM) [1]_.
    The higher weight corresponds to the closer SPD matrix inside A,
    which is closer to B.

    Parameters
    ----------
    A : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    B : ndarray, shape (n_channels, n_channels)
        SPD matrix.
    distance : Callable[[ndarray, ndarray], float]
        One of the pyRiemann distance.
    optimizer : pyQiskitOptimizer, default=ClassicalOptimizer()
        An instance of :class:`pyriemann_qiskit.utils.docplex.pyQiskitOptimizer`.

    Returns
    -------
    weights : ndarray, shape (n_matrices,)
        Optimized weights for the set of SPD matrices A.
        The higher weight corresponds to the closer SPD matrix inside A,
        which is closer to B.

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

    w = optimizer.get_weights(prob, matrices)

    objectif = prob.sum(w[i] * distance(B, A[i]) for i in matrices)

    prob.set_objective("min", objectif)
    prob.add_constraint(prob.sum(w) == 1)
    weights = optimizer.solve(prob, reshape=False)

    return weights


distance_functions["qlogeuclid_hull"] = weights_logeuclid_to_convex_hull
distance_functions["qeuclid"] = lambda A, B, optimizer: _weights_distance(
    A, B, distance_euclid, optimizer
)
distance_functions["qlogeuclid"] = lambda A, B, optimizer: _weights_distance(
    A, B, distance_logeuclid, optimizer
)
