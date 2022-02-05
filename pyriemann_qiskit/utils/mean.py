import numpy as np
from docplex.mp.model import Model
from qiskit.optimization import QuadraticProgram
from docplex.mp.vartype import ContinuousVarType
from qiskit.optimization.algorithms import CobylaOptimizer
from pyriemann.utils.mean import mean_methods


def fro_mean_convex(covmats, sample_weight=None):
    n_trials, n_channels, _ = covmats.shape
    channels = range(n_channels)
    trials = range(n_trials)

    prob = Model()

    ContinuousVarType.one_letter_symbol = lambda _: 'C'
    X_mean = prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                        name='fro_mean', lb=-prob.infinity)

    def _fro_dist(A, B):
        return prob.sum_squares(A[r, c] - B[r, c]
                                for r in channels
                                for c in channels)

    objectives = prob.sum(_fro_dist(covmats[i], X_mean) for i in trials)

    prob.set_objective("min", objectives)

    qp = QuadraticProgram()
    qp.from_docplex(prob)

    result = CobylaOptimizer(rhobeg=0.01, rhoend=0.0001).solve(qp)

    return np.reshape(result.x, (n_channels, n_channels))


mean_methods["convex"] = fro_mean_convex
