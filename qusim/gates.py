"""Single-qubit gate matrices and helpers for QuSim.

The lowercase names are the complex matrices; the uppercase helpers apply a
matrix to a state. Rotation helpers return matrices rather than applying them.
"""

import numpy as np

from .state import apply_gate

# Named gate matrices.
i = np.array([[1, 0], [0, 1]], dtype=complex)
x = np.array([[0, 1], [1, 0]], dtype=complex)
y = np.array([[0, -1j], [1j, 0]], dtype=complex)
z = np.array([[1, 0], [0, -1]], dtype=complex)
h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
s = np.array([[1, 0], [0, 1j]], dtype=complex)
t = np.array([[1, 0], [0, (1 + 1j) / np.sqrt(2)]], dtype=complex)


# Gate helpers: apply the matching matrix to a state.
def X(state):
    """Apply the Pauli-X (bit-flip) gate."""
    return apply_gate(x, state)


def Y(state):
    """Apply the Pauli-Y gate."""
    return apply_gate(y, state)


def Z(state):
    """Apply the Pauli-Z (phase-flip) gate."""
    return apply_gate(z, state)


def H(state):
    """Apply the Hadamard gate."""
    return apply_gate(h, state)


def S(state):
    """Apply the S (phase) gate."""
    return apply_gate(s, state)


def T(state):
    """Apply the T (pi/8) gate."""
    return apply_gate(t, state)


# Rotation and phase helpers: return a 2x2 matrix for the given angle.
def rx(theta):
    """Return a 2x2 matrix for a rotation about the X axis by ``theta`` radians."""
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    matrix = np.array([[c, s], [s, c]], dtype=complex)
    return matrix


def ry(theta):
    """Return a 2x2 matrix for a rotation about the Y axis by ``theta`` radians."""
    c = np.cos(theta / 2)
    s = -np.sin(theta / 2)
    matrix = np.array([[c, s], [-s, c]], dtype=complex)
    return matrix


def rz(theta):
    """Return a 2x2 matrix for a rotation about the Z axis by ``theta`` radians."""
    e = np.exp(-1j * theta / 2)
    matrix = np.array([[e, 0], [0, 1 / e]], dtype=complex)
    return matrix


def phase(phi):
    """Return diag(1, e^(i*phi)), the general phase gate."""
    matrix = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    return matrix
