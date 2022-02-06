import numpy as np
from docplex.mp.vartype import ContinuousVarType, IntegerVarType
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import (MinimumEigenOptimizer,
                                            CobylaOptimizer)
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


class NaiveQAOAOptimizer(pyQiskitOptimizer):
    def convert_covmat(self, covmat, precision=10**4):
        return np.round(covmat * precision, 0)

    def covmat_var(self, prob, channels, name):
        IntegerVarType.one_letter_symbol = lambda _: 'I'
        return prob.integer_var_matrix(keys1=channels, keys2=channels,
                                       name=name, lb=-prob.infinity)

    def _solve_qp(self, qp):
        conv = IntegerToBinary()
        qubo = conv.convert(qp)
        backend = BasicAer.get_backend('statevector_simulator')
        quantum_instance = QuantumInstance(backend)
        qaoa_mes = QAOA(quantum_instance=quantum_instance,
                        initial_point=[0., 0.])
        qaoa = MinimumEigenOptimizer(qaoa_mes)
        return qaoa.solve(qubo)


class CVQAOAOptimizer(pyQiskitOptimizer):
    def covmat_var(self, prob, channels, name):
        return super().covmat_var(prob, channels, name)

    def _solve_qp(self, qp):
        return super()._solve_qp(qp)
