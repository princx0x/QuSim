import numpy as np
from .state import probabilities

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
        key = bin(i)[2:]
        key = "0"*(n-len(key)) + key
        cnt = int(counts[i])
        if cnt > 0:
            out_Dict[key] = cnt
    return out_Dict



def measure(state, rng=None):
    """Pick one outcome. Return (outcome_bitstring, collapsed_state)."""
    #Currently working on it