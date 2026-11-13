"""Tests for the single-qubit gate matrices and rotation helpers."""

import numpy as np
import pytest

from qusim import gates, state

# Named gate matrices, and the subset that are their own inverse.
ALL_GATES = [gates.i, gates.x, gates.y, gates.z, gates.h, gates.s, gates.t]
SELF_INVERSE = [gates.i, gates.x, gates.y, gates.z, gates.h]

I2 = np.eye(2)


# Basic matrix properties.
@pytest.mark.parametrize('gate', ALL_GATES)
def test_gate_is_unitary(gate):
    """Every named gate satisfies U-dagger U = I."""
    assert np.allclose(gate.conj().T @ gate, I2)


@pytest.mark.parametrize('gate', SELF_INVERSE)
def test_gate_is_its_own_inverse(gate):
    """Self-inverse gates square to the identity."""
    assert np.allclose(gate @ gate, I2)


def test_x_and_z_anticommute():
    """Pauli-X and Pauli-Z anticommute."""
    assert np.allclose(gates.x @ gates.z, -(gates.z @ gates.x))


def test_s_squared_is_z():
    """Applying S twice is equivalent to Z."""
    assert np.allclose(gates.s @ gates.s, gates.z)


def test_t_squared_is_s():
    """Applying T twice is equivalent to S."""
    assert np.allclose(gates.t @ gates.t, gates.s)


# Gate action on states.
def test_x_flips_zero_to_one():
    """X maps |0> to |1>."""
    result = gates.x @ state.zero_state(1)
    assert np.allclose(result, state.basis_state('1'))


def test_h_gives_equal_probabilities():
    """H|0> has a 50/50 chance of measuring 0 or 1."""
    result = gates.h @ state.zero_state(1)
    assert np.allclose(state.probabilities(result), [0.5, 0.5])


def test_h_twice_returns_zero_state():
    """Applying H twice restores the original state."""
    result = gates.h @ gates.h @ state.zero_state(1)
    assert np.allclose(result, state.zero_state(1))


def test_h_of_zero_state_gives_equal_probabilities():
    """The H helper also produces an equal superposition."""
    result = state.probabilities(gates.H(state.zero_state()))
    assert np.allclose(result, [0.5, 0.5])


# Rotation gates.
ANGLES = [0, 0.5, 1, np.pi / 2, np.pi, 2 * np.pi, -1.3]
ROTATIONS = [gates.rx, gates.ry, gates.rz]


@pytest.mark.parametrize('rot', ROTATIONS)
@pytest.mark.parametrize('theta', ANGLES)
def test_rotation_is_unitary(rot, theta):
    """Every rotation matrix is unitary."""
    m = rot(theta)
    assert np.allclose(m.conj().T @ m, np.eye(2))


@pytest.mark.parametrize('rot', ROTATIONS)
def test_rotation_zero_is_identity(rot):
    """A rotation by zero is the identity."""
    assert np.allclose(rot(0), np.eye(2))


@pytest.mark.parametrize('rot', ROTATIONS)
def test_rotation_has_complex_dtype(rot):
    """Rotation matrices always use the complex dtype."""
    assert rot(0.7).dtype == complex


def test_rx_pi_is_x_up_to_phase():
    """Rx(pi) equals X up to a global phase."""
    assert np.allclose(gates.rx(np.pi), -1j * gates.x)


def test_ry_pi_is_y_up_to_phase():
    """Ry(pi) equals Y up to a global phase."""
    assert np.allclose(gates.ry(np.pi), -1j * gates.y)


def test_rz_pi_is_z_up_to_phase():
    """Rz(pi) equals Z up to a global phase."""
    assert np.allclose(gates.rz(np.pi), -1j * gates.z)


@pytest.mark.parametrize('rot', ROTATIONS)
def test_full_turn_is_minus_identity(rot):
    """A 2*pi rotation returns minus the identity."""
    assert np.allclose(rot(2 * np.pi), -np.eye(2))


@pytest.mark.parametrize('rot', ROTATIONS)
def test_rotations_add_up(rot):
    """Two rotations compose into a rotation by the sum of the angles."""
    a, b = 0.4, 1.1
    assert np.allclose(rot(a) @ rot(b), rot(a + b))


def test_rx_half_pi_gives_equal_probabilities():
    """Rx(pi/2)|0> is an equal superposition."""
    psi = gates.rx(np.pi / 2) @ state.zero_state(1)
    assert np.allclose(state.probabilities(psi), [0.5, 0.5])


def test_ry_on_zero_gives_cos_sin():
    """Ry(theta)|0> has amplitudes cos(theta/2) and sin(theta/2)."""
    theta = 0.9
    psi = gates.ry(theta) @ state.zero_state(1)
    assert np.allclose(psi, [np.cos(theta / 2), np.sin(theta / 2)])


def test_rz_does_not_change_probabilities():
    """A Z rotation only changes phases, not measurement probabilities."""
    psi = gates.h @ state.zero_state(1)
    rotated = gates.rz(1.234) @ psi
    assert np.allclose(state.probabilities(rotated), state.probabilities(psi))


# Phase gate.
@pytest.mark.parametrize(
    'angle, expected',
    [
        (np.pi, 'z'),
        (np.pi / 2, 's'),
        (np.pi / 4, 't'),
    ],
)
def test_phase_matches_named_gates(angle, expected):
    """The phase gate reproduces Z, S and T at the usual angles."""
    assert np.allclose(gates.phase(angle), getattr(gates, expected))


def test_phase_is_unitary_and_complex():
    """The phase gate is unitary and uses the complex dtype."""
    m = gates.phase(0.8)
    assert m.dtype == complex
    assert np.allclose(m.conj().T @ m, np.eye(2))


def test_rz_equals_phase_up_to_global_phase():
    """Rz and the phase gate differ only by a global phase."""
    phi = 0.8
    assert np.allclose(gates.rz(phi), np.exp(-1j * phi / 2) * gates.phase(phi))
