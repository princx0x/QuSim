import numpy as np
import pytest
from qusim import gates, state

ALL_GATES = [gates.i, gates.x, gates.y, gates.z, gates.h, gates.s, gates.t]
SELF_INVERSE = [gates.i, gates.x, gates.y, gates.z, gates.h]

I2 = np.eye(2)

@pytest.mark.parametrize('gate', ALL_GATES)
def test_gate_is_unitary(gate):
    assert np.allclose(gate.conj().T @ gate, I2)

@pytest.mark.parametrize('gate', SELF_INVERSE)
def test_gate_is_its_own_inverse(gate):
    assert np.allclose(gate @ gate, I2)

def test_x_and_z_anticommute():
    assert np.allclose(gates.x @ gates.z, -(gates.z @ gates.x))

def test_s_squared_is_z():
    assert np.allclose(gates.s @ gates.s, gates.z)

def test_t_squared_is_s():
    assert np.allclose(gates.t @ gates.t, gates.s)

def test_x_flips_zero_to_one():
    result = gates.x @ state.zero_state(1)
    assert np.allclose(result, state.basis_state('1'))

def test_h_gives_equal_probabilities():
    result = gates.h @ state.zero_state(1)
    assert np.allclose(state.probabilities(result), [0.5, 0.5])

def test_h_twice_returns_zero_state():
    result = gates.h @ gates.h @ state.zero_state(1)
    assert np.allclose(result, state.zero_state(1))

def test_h_of_zero_state_gives_equal_probabilities():
    result = state.probabilities(gates.H(state.zero_state()))
    assert np.allclose(result, [0.5, 0.5])