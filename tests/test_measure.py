"""Tests for sampling and single-shot measurement."""

import numpy as np

from qusim import measure, state


def test_sample_on_basic_states():
    """Sampling a definite basis state always gives the same bitstring."""
    assert measure.sample([0, 1, 0, 0], 50) == {'01': 50}


def test_sample_bell_state_keys():
    """A Bell state only produces the 00 and 11 outcomes."""
    counts = measure.sample(state.normalize([1, 0, 0, 1]), shots=1000)
    assert counts['00'] + counts['11'] == 1000


def test_sample_counts_sum_to_shots():
    """The counts returned by sample add up to the number of shots."""
    assert sum(measure.sample(state.normalize([1, 1]), shots=777).values()) == 777


def test_sample_same_seed_same_counts():
    """Seeded sampling is reproducible."""
    psi = state.normalize([1, 1])
    a = measure.sample(psi, 500, np.random.default_rng(5))
    b = measure.sample(psi, 500, np.random.default_rng(5))
    assert a == b


def test_measure_basis_state_is_deterministic():
    """Measuring a basis state returns that state and leaves it unchanged."""
    bits, collapsed = measure.measure(state.basis_state('10'))
    assert bits == '10'
    assert np.allclose(collapsed, state.basis_state('10'))


def test_measure_collapsed_state_is_basis_state():
    """A collapsed state is a normalized basis state."""
    psi = state.normalize([1, 1, 1, 1])
    _, collapsed = measure.measure(psi, np.random.default_rng(0))
    assert np.isclose(np.sum(np.abs(collapsed) ** 2), 1.0)
    assert np.count_nonzero(collapsed) == 1


def test_measuring_twice_gives_same_outcome():
    """Re-measuring a collapsed state reproduces the first outcome."""
    psi = state.normalize([1, 1])
    for seed in range(20):
        first, collapsed = measure.measure(psi, np.random.default_rng(seed))
        second, _ = measure.measure(collapsed, np.random.default_rng(seed + 100))
        assert first == second


def test_measure_same_seed_same_outcome():
    """Seeded single-shot measurement is reproducible."""
    psi = state.normalize([1, 1])
    a, _ = measure.measure(psi, np.random.default_rng(5))
    b, _ = measure.measure(psi, np.random.default_rng(5))
    assert a == b
