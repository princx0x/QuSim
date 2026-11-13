"""State-vector helpers for QuSim.

An ``n``-qubit state is stored as a complex NumPy array of length ``2 ** n``.
Reshaping that array to shape ``[2] * n`` gives one axis per qubit, so the
magnitude at index ``k`` belongs to computational basis state ``k``.
"""

import numpy as np

# Tolerance used when checking that a state vector has magnitude 1.
atol = 1e-9


def apply_gate(gate, state):
    """Apply a gate matrix to a state and return the new state.

    The gate must be a square matrix whose dimension matches the length of
    ``state``, so the same helper works for one- and two-qubit gates.
    """
    state = np.asarray(state, dtype=complex)
    gate = np.asarray(gate, dtype=complex)
    norm = np.vdot(state, state).real
    if not np.isclose(norm, 1.0, atol):
        raise ValueError(f'Expected a state of magnitude 1, got magnitude = {norm}')
    elif state.ndim != 1 or gate.shape != (state.size, state.size):
        raise ValueError(
            f'Expected a {state.size}x{state.size} gate for a length-{state.size} '
            f'state, got {gate.shape} and {state.shape}'
        )
    else:
        return gate @ state


def zero_state(n=1):
    """Return |00...0> as a complex array of length 2**n."""
    out_array = np.array([1] + [0 for _ in range(2**n - 1)], dtype=complex)
    return out_array


def basis_state(bits):
    """Return the basis state for ``bits``, for example '01' -> |01>."""
    out_lst = [0] * (2**len(bits))
    index = int(bits, 2)
    out_lst[index] = 1
    return np.array(out_lst, dtype=complex)


def normalize(state):
    """Return a unit-norm copy of ``state``; raise ValueError for the zero vector."""
    out = np.array(state, dtype=complex)
    norm = np.linalg.norm(out)
    if np.isclose(norm, 0):
        raise ValueError('Cannot normalize the zero state')
    return out / norm


# The four Bell states, each as a normalized two-qubit state vector.
def bell_phi_plus():
    """Return the Bell state (|00> + |11>) / sqrt(2)."""
    return normalize([1, 0, 0, 1])


def bell_phi_minus():
    """Return the Bell state (|00> - |11>) / sqrt(2)."""
    return normalize([1, 0, 0, -1])


def bell_psi_plus():
    """Return the Bell state (|01> + |10>) / sqrt(2)."""
    return normalize([0, 1, 1, 0])


def bell_psi_minus():
    """Return the Bell state (|01> - |10>) / sqrt(2)."""
    return normalize([0, 1, -1, 0])


def probabilities(state):
    """Return the measurement probability of each basis state."""
    out_lst = []
    for amplitude in state:
        out_lst.append(np.abs(amplitude) ** 2)
    return np.array(out_lst)
