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

ANGLES = [0, 0.5, 1, np.pi / 2, np.pi, 2 * np.pi, -1.3]
ROTATIONS = [gates.rx, gates.ry, gates.rz]

@pytest.mark.parametrize('rot', ROTATIONS)
@pytest.mark.parametrize('theta', ANGLES)
def test_rotation_is_unitary(rot, theta):
    m = rot(theta)
    assert np.allclose(m.conj().T @ m, np.eye(2))

@pytest.mark.parametrize('rot', ROTATIONS)
def test_rotation_zero_is_identity(rot):
    assert np.allclose(rot(0), np.eye(2))

@pytest.mark.parametrize('rot', ROTATIONS)
def test_rotation_has_complex_dtype(rot):
    assert rot(0.7).dtype == complex

def test_rx_pi_is_x_up_to_phase():
    assert np.allclose(gates.rx(np.pi), -1j * gates.x)

def test_ry_pi_is_y_up_to_phase():
    assert np.allclose(gates.ry(np.pi), -1j * gates.y)

def test_rz_pi_is_z_up_to_phase():
    assert np.allclose(gates.rz(np.pi), -1j * gates.z)

@pytest.mark.parametrize('rot', ROTATIONS)
def test_full_turn_is_minus_identity(rot):
    assert np.allclose(rot(2 * np.pi), -np.eye(2))

@pytest.mark.parametrize('rot', ROTATIONS)
def test_rotations_add_up(rot):
    a, b = 0.4, 1.1
    assert np.allclose(rot(a) @ rot(b), rot(a + b))

def test_rx_half_pi_gives_equal_probabilities():
    psi = gates.rx(np.pi / 2) @ state.zero_state(1)
    assert np.allclose(state.probabilities(psi), [0.5, 0.5])

def test_ry_on_zero_gives_cos_sin():
    theta = 0.9
    psi = gates.ry(theta) @ state.zero_state(1)
    assert np.allclose(psi, [np.cos(theta / 2), np.sin(theta / 2)])

def test_rz_does_not_change_probabilities():
    psi = gates.h @ state.zero_state(1)
    rotated = gates.rz(1.234) @ psi
    assert np.allclose(state.probabilities(rotated), state.probabilities(psi))

@pytest.mark.parametrize('angle, expected', [
    (np.pi, 'z'),
    (np.pi / 2, 's'),
    (np.pi / 4, 't'),
])
def test_phase_matches_named_gates(angle, expected):
    assert np.allclose(gates.phase(angle), getattr(gates, expected))

def test_phase_is_unitary_and_complex():
    m = gates.phase(0.8)
    assert m.dtype == complex
    assert np.allclose(m.conj().T @ m, np.eye(2))

def test_rz_equals_phase_up_to_global_phase():
    phi = 0.8
    assert np.allclose(gates.rz(phi), np.exp(-1j * phi / 2) * gates.phase(phi))