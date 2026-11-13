"""QAOA on a QUBO, simulated exactly with numpy.

This module reproduces, for a quadratic unconstrained binary problem, what
``MinimumEigenOptimizer(QAOA(sampler, optimizer, reps, initial_point))`` does
with a noiseless statevector sampler, but without building circuits:

- the QAOA state is computed exactly. The cost layer exp(-i gamma H) is a
  phase on each basis state, because the Ising Hamiltonian H of a QUBO is
  diagonal. The mixer layer exp(-i beta sum_k X_k) is applied as a product of
  one-qubit rotations;
- at each evaluation, ``shots`` measurements are drawn from the exact
  probabilities, with a generator seeded with ``seed`` at every evaluation (as
  a seeded simulator does), and the energy is the mean of H over the drawn
  bitstrings;
- the classical optimizer minimizes this energy from the initial point, the
  angles being ordered as in qiskit: the betas, then the gammas;
- the solution is the drawn bitstring with the lowest objective value, among
  ``shots`` measurements of the optimized state.

The circuit is the same as the one built by qiskit's QAOA, so the state is the
same; only the random generator used to draw the measurements differs.
"""

import math

import numpy as np

_BLOCK = 6  # qubits per dense block in the mixer layer


def _ising_diagonal(operator, n_qubits):
    """Value of a diagonal Pauli operator (Z and I terms) on every basis state.

    Qubit k is bit k of the basis-state index (qiskit's ordering).
    """
    if operator.paulis.x.any():
        raise ValueError("The cost operator must be diagonal.")
    index = np.arange(2**n_qubits, dtype=np.int64)
    signs = (1 - 2 * ((index[:, None] >> np.arange(n_qubits)) & 1)).astype(np.float64)
    diagonal = np.zeros(2**n_qubits)
    coeffs = np.real(operator.coeffs)
    for term in range(len(coeffs)):
        qubits = np.flatnonzero(operator.paulis.z[term])
        if len(qubits) == 0:
            diagonal += coeffs[term]
        elif len(qubits) == 1:
            diagonal += coeffs[term] * signs[:, qubits[0]]
        else:
            diagonal += coeffs[term] * np.prod(signs[:, qubits], axis=1)
    return diagonal


def _rx(angle):
    """exp(-i angle X)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -1j * s], [-1j * s, c]])


def _apply_mixer(psi, n_qubits, beta):
    """Apply exp(-i beta sum_k X_k), by dense blocks of _BLOCK qubits."""
    one_qubit = _rx(beta)
    for k0 in range(0, n_qubits, _BLOCK):
        size = min(_BLOCK, n_qubits - k0)
        block = np.array([[1.0 + 0j]])
        for _ in range(size):
            block = np.kron(block, one_qubit)
        high, low = 2 ** (n_qubits - k0 - size), 2**k0
        view = psi.reshape(high, 2**size, low)
        if low == 1:
            psi = view.reshape(high, 2**size) @ block.T
        elif high == 1:
            psi = block @ view.reshape(2**size, low)
        else:
            psi = np.matmul(block, view)
        psi = np.ascontiguousarray(psi).reshape(-1)
    return psi


def qaoa_state(cost_diagonal, n_qubits, betas, gammas):
    """Exact QAOA state for the given angles, starting from |+>^n."""
    psi = np.full(2**n_qubits, 1 / math.sqrt(2**n_qubits), dtype=complex)
    for beta, gamma in zip(betas, gammas):
        psi = psi * np.exp(-1j * gamma * cost_diagonal)
        psi = _apply_mixer(psi, n_qubits, beta)
    return psi


def _draw(psi, shots, seed):
    """Draw `shots` measurements of psi. Returns the basis states and their counts."""
    probabilities = np.abs(psi) ** 2
    cumulative = np.cumsum(probabilities / probabilities.sum())
    uniform = np.random.default_rng(seed).random(shots)
    outcomes = np.searchsorted(cumulative, uniform, side="right")
    outcomes = np.minimum(outcomes, len(cumulative) - 1)
    return np.unique(outcomes, return_counts=True)


def solve_qubo_qaoa_statevector(
    problem, optimizer, initial_point, reps, shots=1024, seed=None, callback=None
):
    """Minimize a QUBO with QAOA, simulated exactly with numpy.

    Parameters
    ----------
    problem : QuadraticProgram
        A problem with binary variables only and no constraints.
    optimizer : qiskit_algorithms.optimizers.Optimizer
        Classical optimizer of the QAOA angles.
    initial_point : list[float]
        Initial angles, the ``reps`` betas then the ``reps`` gammas.
    reps : int
        Number of QAOA layers.
    shots : int, default=1024
        Number of measurements per evaluation and for the final state.
    seed : int | None, default=None
        Seed of the measurements, reused at every evaluation.
    callback : callable | None, default=None
        Called as ``callback(eval_count, parameters, energy, metadata)`` after
        each evaluation, as in qiskit's QAOA.

    Returns
    -------
    x : ndarray, shape (n_variables,)
        Binary values of the variables, in the order of ``problem.variables``.
    """
    if problem.linear_constraints or problem.quadratic_constraints:
        raise ValueError("The problem must not have constraints.")
    operator, _ = problem.to_ising()
    n_qubits = operator.num_qubits
    cost_diagonal = _ising_diagonal(operator, n_qubits)
    eval_count = 0

    def measure(parameters):
        betas, gammas = parameters[:reps], parameters[reps:]
        return _draw(qaoa_state(cost_diagonal, n_qubits, betas, gammas), shots, seed)

    def energy(parameters):
        nonlocal eval_count
        states, counts = measure(np.asarray(parameters, dtype=float))
        value = float(np.sum(counts / shots * cost_diagonal[states]))
        eval_count += 1
        if callback is not None:
            callback(eval_count, list(parameters), value, {"shots": shots})
        return value

    result = optimizer.minimize(
        fun=energy,
        x0=np.asarray(initial_point, dtype=float),
        bounds=[(None, None)] * 2 * reps,
    )
    states, _ = measure(np.asarray(result.x, dtype=float))
    # Variable i is qubit i, i.e. bit i of the basis-state index.
    x = ((states[:, None] >> np.arange(n_qubits)) & 1).astype(float)
    linear = problem.objective.linear.to_array()
    quadratic = problem.objective.quadratic.to_array()
    values = x @ linear + np.einsum("si,ij,sj->s", x, quadratic, x)
    return x[np.argsort(values, kind="stable")[0]]
