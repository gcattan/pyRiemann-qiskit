import numpy as np
from docplex.mp.model import Model
from docplex.mp.vartype import ContinuousVarType, IntegerVarType
from pyriemann.utils.mean import mean_methods
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer, CobylaOptimizer
from qiskit.optimization.converters import IntegerToBinary


class pyQiskitOptimizer():
    def convert_covmat(self, covmat, precision=None):
        return covmat

    def covmat_var(self, prob, channels, name):
        raise NotImplementedError()

    def _solve_qp(self, qp):
        raise NotImplementedError()

    def solve(self, prob):
        qp = QuadraticProgram()
        qp.from_docplex(prob)
        return self._solve_qp(qp)

class ClassicalOptimizer(pyQiskitOptimizer):
    def covmat_var(self, prob, channels, name):
        ContinuousVarType.one_letter_symbol = lambda _: 'C'
        return prob.continuous_var_matrix(keys1=channels, keys2=channels,
                                        name=name, lb=-prob.infinity)
    
    def _solve_qp(self, qp):
        return CobylaOptimizer(rhobeg=0.01, rhoend=0.0001).solve(qp) 

class NaiveQuantumOptimizer(pyQiskitOptimizer):
    def convert_covmat(self, covmat, precision=10**4):
        return np.round(covmat * precision, 0)

    def covmat_var(self, prob, channels, name):
        IntegerVarType.one_letter_symbol = lambda _: 'I'
        return prob.integer_var_matrix(keys1=channels, keys2=channels,
                                        name=name, lb=-prob.infinity)
    
    def _solve_qp(self, qp):
        conv = IntegerToBinary()
        qubo = conv.convert(qp)
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                   seed_simulator=aqua_globals.random_seed,
                                   seed_transpiler=aqua_globals.random_seed)
        qaoa_mes = QAOA(quantum_instance=quantum_instance, initial_point=[0., 0.])
        qaoa = MinimumEigenOptimizer(qaoa_mes) 
        return qaoa.solve(qubo)

# class CVQAOAOptimizer(pyQiskitOptimizer):
#     def covmat_var(self):
#         return super().covmat_var()
    
#     def solve(self):
#       return super().solve()


def fro_mean_convex(covmats, sample_weight=None, optimizer=ClassicalOptimizer()):
    n_trials, n_channels, _ = covmats.shape
    channels = range(n_channels)
    trials = range(n_trials)

    prob = Model()

    X_mean = optimizer.covmat_var(prob, channels, 'fro_mean')

    def _fro_dist(A, B):
        return prob.sum_squares(A[r, c] - B[r, c]
                                for r in channels
                                for c in channels)

    objectives = prob.sum(_fro_dist(optimizer.convert_covmat(covmats[i]), X_mean) for i in trials)

    prob.set_objective("min", objectives)

    result = optimizer.solve(prob)

    return np.reshape(result.x, (n_channels, n_channels))


mean_methods["convex"] = fro_mean_convex
