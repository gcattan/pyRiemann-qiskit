import cvxpy as cvx
from docplex.mp.model import Model
from docplex.cp.modeler import constant
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import GroverOptimizer
from qiskit import Aer
from docplex.mp.vartype import ContinuousVarType

from qiskit.aqua.components.optimizers import COBYLA, SLSQP
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer,  CobylaOptimizer, SlsqpOptimizer
from qiskit.optimization.converters import QuadraticProgramToQubo
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
import numpy as np


def fro_mean_convex(covmats, sample_weight=None):
    n_trials, n_channels, _ = covmats.shape
    channels=range(n_channels)
    trials=range(n_trials)

    prob = Model()

    ContinuousVarType.one_letter_symbol = lambda _: 'C'
    X_mean = prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                        name='fro_mean', lb=-prob.infinity) 

    def _fro_dist(A, B):
        return prob.sum_squares(A[r, c] - B[r,c] for r in channels for c in channels)

    objectives = prob.sum(_fro_dist(covmats[i], X_mean) for i in trials)
    
    prob.set_objective("min", objectives)

    qp = QuadraticProgram()
    qp.from_docplex(prob)
    
    result = CobylaOptimizer().solve(qp)

    return np.reshape(result.x, (n_channels, n_channels))

