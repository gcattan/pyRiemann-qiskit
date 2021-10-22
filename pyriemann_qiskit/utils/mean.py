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

from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
import numpy as np

def fro_mean_convex(covmats):
    k, _, n = covmats.shape
    prob = Model()
    keys=range(n)
    # workaround for one_letter_symbol not in this version of the docplex
    # C stands for continuous variable
    ContinuousVarType.one_letter_symbol = lambda self: 'C'
    X_mean = prob.continuous_var_matrix(keys1=keys, keys2=keys, name='fro_mean') #put lower bound to infinity
    # X_mean = cvx.Variable((n, n), symmetric=True)
    # X_mean = [[prob.continuous_var() for i in keys] for j in keys]

    # covset = [[[prob.continuous_var(ub=covmats[l][j][i]+0.001, lb=covmats[l][j][i]-0.001) for i in keys] for j in keys] for l in range(k)]
    
    # with open('out3.txt', 'w') as f:
    #     f.write(str(covset))

    def dist(i):
        return prob.sum_squares(covmats[i][r][c] - X_mean[r,c] for r in keys for c in keys)
        # return cvx.norm(X_mean - covmats[i], "fro")

    # expression = sum(dist(i) for i in range(k))
    # for i in range(k):
    #     prob.set_objective("min", dist(i))
    expression = prob.sum(dist(i) for i in range(k))

    # for i in range(k):  
    #     for j in keys:
    #         for l in keys:
    #             prob.add_constraint(covset[i][j][l] == covmats[i][j][l])
    # prob.add_constraints(X_mean >= 0)
    # prob.add_constraints(X_mean[r, c] >= 0 for r in keys for c in keys)
    # prob.add_constraints(-X_mean[r, c] > 0 for r in keys for c in keys)
    # constraints = [X_mean >> 0]

    
    prob.set_objective("min", expression)
    # prob = cvx.Problem(cvx.Minimize(expression), constraints)
    # solution = prob.solve()
    print(prob.lp_string)
    qp = QuadraticProgram()
    qp.from_docplex(prob)

    # prob.solve(verbose=True)
    # with open('out.txt', 'w') as f:
    #     f.write(str(expression))
    
    quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                      seed_simulator=42,
                                      seed_transpiler=42)


    admm_params = ADMMParameters(
                            rho_initial=1001,
                            beta=1000,
                            factor_c=900,
                            maxiter=100,
                            three_block=True, tol=1.e-6
                        )
    # initialize ADMM with classical QUBO and convex optimizer
    admm = ADMMOptimizer(params=admm_params,
                        continuous_optimizer=CobylaOptimizer(rhobeg=0.01))
    result = admm.solve(qp)
    with open('out2.txt', 'w') as f:
        f.write(str(result))
    return np.reshape(result.x, (n, n))
    # return X_mean.value
