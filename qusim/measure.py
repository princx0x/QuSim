import numpy as np
from .state import probabilities, basis_state

def _pad_bitstring(bitstr, n):
    return "0"*(n-len(bitstr))+bitstr

def sample(state, shots=1000, rng=None):
    """Return counts like {'0': 512, '1': 488}."""
    if rng is None:
        rng = np.random.default_rng()
    n = int(np.log2(len(state)))
    probs = probabilities(state)
    probs = probs/probs.sum()

    outcomes_lst = rng.choice(len(probs), size = shots, p=probs)
    counts = np.bincount(outcomes_lst, minlength=len(probs))
    out_Dict = {}
    for i in range(len(counts)):
        key = _pad_bitstring(bin(i)[2:],n)
        cnt = int(counts[i])
        if cnt > 0:
            out_Dict[key] = cnt
    return out_Dict



def measure(state, rng=None):
    """Pick one outcome. Return (outcome_bitstring, collapsed_state)."""
    if rng is None:
        rng = np.random.default_rng()

    probs = probabilities(state)
    probs = probs/probs.sum()
    n = int(np.log2(len(probs)))
    state_Value = rng.choice(len(probs), p=probs)
    bitstring = _pad_bitstring(bin(state_Value)[2:],n)
    return (bitstring,basis_state(bitstring))
