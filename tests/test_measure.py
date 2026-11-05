import pytest
from qusim import measure, state
import numpy as np

def test_sample_on_basic_states():
    assert measure.sample([0,1,0,0],50) == {'01':50}

def test_sample_bell_state_keys():
    counts = measure.sample(state.normalize([1, 0, 0, 1]), shots=1000)
    assert counts['00'] + counts['11'] == 1000

def test_sample_counts_sum_to_shots():
    assert sum(measure.sample(state.normalize([1, 1]), shots=777).values()) == 777

def test_sample_same_seed_same_counts():
    psi = state.normalize([1, 1])
    a = measure.sample(psi, 500, np.random.default_rng(5))
    b = measure.sample(psi, 500, np.random.default_rng(5))
    assert a == b

