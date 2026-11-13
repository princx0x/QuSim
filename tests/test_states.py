"""Tests for state construction, normalization and probabilities."""

import numpy as np
import pytest

from qusim import state

BELL_STATES = [
    state.bell_phi_plus,
    state.bell_phi_minus,
    state.bell_psi_plus,
    state.bell_psi_minus,
]


def test_normalize_real():
    """Normalizing real amplitudes scales them to unit length."""
    assert np.allclose(state.normalize([3, 4]), [0.6, 0.8])


def test_normalize_complex():
    """Normalizing complex amplitudes gives probabilities that sum to 1."""
    result = state.normalize([1, 1j])
    assert np.isclose(np.sum(np.abs(result) ** 2), 1.0)


def test_normalize_leaves_input_unchanged():
    """Normalize returns a copy and does not modify its input."""
    original = np.array([3, 4], dtype=complex)
    state.normalize(original)
    assert np.allclose(original, [3, 4])


def test_normalize_zero_raises():
    """Normalizing the zero vector raises ValueError."""
    with pytest.raises(ValueError):
        state.normalize([0, 0])


def test_probabilities_plus_state():
    """The |+> state has equal measurement probabilities."""
    plus = state.normalize([1, 1])
    assert np.allclose(state.probabilities(plus), [0.5, 0.5])


def test_probabilities_basis_state():
    """A basis state has probability 1 at its index and 0 elsewhere."""
    assert np.allclose(state.probabilities(state.basis_state('10')), [0, 0, 1, 0])


def test_probabilities_sum_to_one():
    """Probabilities always sum to 1 for a normalized state."""
    psi = state.normalize([1, 2j, 3, -1])
    assert np.isclose(state.probabilities(psi).sum(), 1.0)


@pytest.mark.parametrize(
    'bell, expected',
    [
        (state.bell_phi_plus, [1, 0, 0, 1]),
        (state.bell_phi_minus, [1, 0, 0, -1]),
        (state.bell_psi_plus, [0, 1, 1, 0]),
        (state.bell_psi_minus, [0, 1, -1, 0]),
    ],
)
def test_bell_states_match_their_definitions(bell, expected):
    """Each Bell-state helper returns the expected normalized vector."""
    assert np.allclose(bell(), state.normalize(expected))


@pytest.mark.parametrize('bell', BELL_STATES)
def test_bell_states_are_normalized(bell):
    """Every Bell state is a valid two-qubit state."""
    psi = bell()
    assert len(psi) == 4
    assert np.isclose(state.probabilities(psi).sum(), 1.0)


def test_bell_states_are_orthonormal():
    """The four Bell states form an orthonormal basis."""
    vectors = [bell() for bell in BELL_STATES]
    for i, first in enumerate(vectors):
        for j, second in enumerate(vectors):
            expected = 1.0 if i == j else 0.0
            assert np.isclose(np.vdot(first, second).real, expected)
