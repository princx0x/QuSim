import numpy as np
import pytest
from qusim import state

def test_normalize_real():
    assert np.allclose(state.normalize([3, 4]), [0.6, 0.8])

def test_normalize_complex():
    result = state.normalize([1, 1j])
    assert np.isclose(np.sum(np.abs(result) ** 2), 1.0)

def test_normalize_leaves_input_unchanged():
    original = np.array([3, 4], dtype=complex)
    state.normalize(original)
    assert np.allclose(original, [3, 4])

def test_normalize_zero_raises():
    with pytest.raises(ValueError):
        state.normalize([0, 0])

def test_probabilities_plus_state():
    plus = state.normalize([1, 1])
    assert np.allclose(state.probabilities(plus), [0.5, 0.5])

def test_probabilities_basis_state():
    assert np.allclose(state.probabilities(state.basis_state('10')), [0, 0, 1, 0])

def test_probabilities_sum_to_one():
    psi = state.normalize([1, 2j, 3, -1])
    assert np.isclose(state.probabilities(psi).sum(), 1.0)