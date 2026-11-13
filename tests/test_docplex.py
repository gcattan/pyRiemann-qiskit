import pytest
from docplex.mp.model import Model
from docplex.mp.vartype import BinaryVarType, ContinuousVarType, IntegerVarType

from pyriemann_qiskit.utils.docplex import (
    ClassicalOptimizer,
    NaiveQAOAOptimizer,
    QAOACVOptimizer,
    square_bin_mat_var,
    square_cont_mat_var,
    square_int_mat_var,
)


@pytest.mark.parametrize(
    "square_mat_var",
    [
        (square_cont_mat_var, ContinuousVarType),
        (square_int_mat_var, IntegerVarType),
        (square_bin_mat_var, BinaryVarType),
    ],
)
def test_get_square_cont_var(square_mat_var):
    """
    Tests that:
      - square_cont_mat_var is returning a variable of type ContinuousVarType
      - square_cont_mat_var is returning a variable of type IntegerVarType
      - square_bin_mat_var is returning a variable of type BinaryVarType
      - all returned variables are square matrices of size 'n' where 'n' is
      the (given) number of channels.
    """
    n = 4
    channels = range(n)
    prob = Model()
    handle = square_mat_var[0]
    expected_result_type = square_mat_var[1]
    mat = handle(prob, channels, name="test")
    first_element = mat[(0, 0)]
    assert len(mat) == n * n
    assert type(first_element.vartype) is expected_result_type


@pytest.mark.parametrize(
    "optimizer", [ClassicalOptimizer, NaiveQAOAOptimizer, QAOACVOptimizer]
)
def test_optimizer_creation(optimizer):
    assert optimizer()


def test_naive_qaoa_ramp_initial_point():
    """The ramp orders betas then gammas: beta falls, gamma rises."""
    opt = NaiveQAOAOptimizer(n_reps=2, initial_points="ramp")
    assert opt._initial_point() == [0.75, 0.25, 0.25, 0.75]


def test_naive_qaoa_pair_initial_point_is_tiled():
    """A single (beta, gamma) pair starts every layer."""
    opt = NaiveQAOAOptimizer(n_reps=3, initial_points=[0.1, 0.2])
    assert opt._initial_point() == [0.1, 0.1, 0.1, 0.2, 0.2, 0.2]


def test_naive_qaoa_initial_point_wrong_length():
    opt = NaiveQAOAOptimizer(n_reps=2, initial_points=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        opt._initial_point()


def test_naive_qaoa_hull_weights_on_simplex_grid():
    """Integer hull weights are a convex combination on a 1/upper_bound grid.

    With upper_bound = 3, two hull points use 2 qubits each (4 qubits).
    """
    import numpy as np
    from qiskit_algorithms.optimizers import COBYLA

    from pyriemann_qiskit.utils.distance import weights_logeuclid_to_convex_hull

    A = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 1.0])])
    B = np.diag([2.0, 1.5])
    opt = NaiveQAOAOptimizer(
        upper_bound=3, n_reps=2, initial_points="ramp", optimizer=COBYLA(maxiter=20)
    )
    weights = weights_logeuclid_to_convex_hull(A, B, opt)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights * 3, np.round(weights * 3))
    assert np.all(weights >= 0)


def test_naive_qaoa_hull_constraint_is_on_the_grid():
    """The integer program sums the weights to upper_bound, and the integer
    solution is returned as a convex combination (divided by upper_bound)."""
    import numpy as np

    from pyriemann_qiskit.utils.distance import weights_logeuclid_to_convex_hull

    class Capture(NaiveQAOAOptimizer):
        def _solve_qp(self, qp, reshape=True):
            self.qp = qp
            return np.array([1.0, 2.0])

    A = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 1.0])])
    B = np.diag([2.0, 1.5])
    capture = Capture(upper_bound=3)
    weights = weights_logeuclid_to_convex_hull(A, B, capture)
    assert capture.qp.linear_constraints[0].rhs == 3
    assert np.allclose(weights, [1 / 3, 2 / 3])


def test_qaoa_statevector_state_matches_qiskit():
    """The numpy QAOA state equals the statevector of qiskit's QAOA circuit."""
    import numpy as np
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from qiskit_optimization import QuadraticProgram

    from pyriemann_qiskit.utils.qaoa_statevector import _ising_diagonal, qaoa_state

    rng = np.random.default_rng(0)
    n = 5
    problem = QuadraticProgram()
    for i in range(n):
        problem.binary_var(f"x{i}")
    problem.minimize(linear=rng.normal(size=n), quadratic=rng.normal(size=(n, n)))
    operator, _ = problem.to_ising()
    assert isinstance(operator, SparsePauliOp)
    ansatz = QAOAAnsatz(operator, reps=2).decompose()
    betas, gammas = rng.uniform(-np.pi, np.pi, 2), rng.uniform(-np.pi, np.pi, 2)
    # qiskit orders the parameters as the betas, then the gammas
    reference = Statevector(ansatz.assign_parameters(list(betas) + list(gammas))).data
    psi = qaoa_state(_ising_diagonal(operator, n), n, betas, gammas)
    overlap = np.vdot(reference, psi)
    assert np.isclose(abs(overlap), 1.0)
    assert np.allclose(psi, overlap / abs(overlap) * reference)


def test_qaoa_statevector_solves_small_qubo():
    """The numpy QAOA loop returns the minimizer of a small QUBO."""
    import itertools

    import numpy as np
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_optimization import QuadraticProgram

    from pyriemann_qiskit.utils.qaoa_statevector import solve_qubo_qaoa_statevector

    rng = np.random.default_rng(1)
    n = 4
    linear, quadratic = rng.normal(size=n), rng.normal(size=(n, n))
    problem = QuadraticProgram()
    for i in range(n):
        problem.binary_var(f"x{i}")
    problem.minimize(linear=linear, quadratic=quadratic)
    values = []
    evaluations = []
    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)
        values.append((x @ linear + x @ quadratic @ x, bits))
    best = np.array(min(values)[1], dtype=float)
    x = solve_qubo_qaoa_statevector(
        problem,
        COBYLA(maxiter=30),
        [0.75, 0.25, 0.25, 0.75],
        reps=2,
        shots=256,
        seed=0,
        callback=lambda *args: evaluations.append(args),
    )
    assert np.array_equal(x, best)
    assert len(evaluations) > 1


def test_naive_qaoa_statevector_simulator_on_simplex_grid():
    """simulator="statevector" returns hull weights on the simplex grid."""
    import numpy as np
    from qiskit_algorithms.optimizers import COBYLA

    from pyriemann_qiskit.utils.distance import weights_logeuclid_to_convex_hull

    A = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 1.0])])
    B = np.diag([2.0, 1.5])
    opt = NaiveQAOAOptimizer(
        upper_bound=3,
        n_reps=2,
        initial_points="ramp",
        optimizer=COBYLA(maxiter=20),
        simulator="statevector",
    )
    weights = weights_logeuclid_to_convex_hull(A, B, opt)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights * 3, np.round(weights * 3))
    assert len(opt.evaluated_values_) > 1


def test_naive_qaoa_unknown_simulator():
    import numpy as np

    from pyriemann_qiskit.utils.distance import weights_logeuclid_to_convex_hull

    A = np.array([np.diag([1.0, 2.0]), np.diag([3.0, 1.0])])
    B = np.diag([2.0, 1.5])
    with pytest.raises(ValueError):
        weights_logeuclid_to_convex_hull(A, B, NaiveQAOAOptimizer(simulator="gpu"))
