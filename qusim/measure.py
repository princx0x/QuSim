"""Measurement helpers: sampling, single-shot measurement and state collapse."""

import numpy as np

from .state import basis_state, probabilities


def _test_valid_state(state):
    """Raise ValueError unless ``state`` has length 2**n and magnitude 1."""
    if np.log2(len(state)) % 1 != 0:
        raise ValueError(f'Expected a state with 2^n possibilities, got {len(state)}')
    norm = np.sqrt(np.vdot(state, state).real)
    if not np.isclose(norm, 1):
        raise ValueError(f'Expected a state of magnitude 1, got {norm:.2f}')


def _pad_bitstring(bitstr, n):
    """Left-pad ``bitstr`` with zeros so that it has length ``n``."""
    return '0' * (n - len(bitstr)) + bitstr


def sample(state, shots=1000, rng=None):
    """Return measurement counts, for example {'0': 512, '1': 488}."""
    _test_valid_state(state)
    if rng is None:
        rng = np.random.default_rng()
    n = int(np.log2(len(state)))
    probs = probabilities(state)
    probs = probs / probs.sum()

    outcomes = rng.choice(len(probs), size=shots, p=probs)
    counts = np.bincount(outcomes, minlength=len(probs))
    out_dict = {}
    for i, count in enumerate(counts):
        key = _pad_bitstring(bin(i)[2:], n)
        count = int(count)
        if count > 0:
            out_dict[key] = count
    return out_dict


def measure(state, rng=None):
    """Pick one outcome and return (outcome_bitstring, collapsed_state)."""
    _test_valid_state(state)
    if rng is None:
        rng = np.random.default_rng()

    probs = probabilities(state)
    probs = probs / probs.sum()
    n = int(np.log2(len(probs)))
    state_value = rng.choice(len(probs), p=probs)
    bitstring = _pad_bitstring(bin(state_value)[2:], n)
    return (bitstring, basis_state(bitstring))
