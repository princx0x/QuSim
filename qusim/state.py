import numpy as np

#CONSTANTS FOR VARIOUS PURPOSES
atol = 1e-9 #tolerance for magnitude of the state


def apply_gate(gate,state):
    state = np.asarray(state, dtype=complex)
    norm = np.vdot(state, state).real
    if not (np.isclose(norm,1.0,atol)):
        raise ValueError(f'Expected a state of magnitude 1, got magnitude = {norm}')
    elif gate.shape != (2, 2) or state.shape != (2,):
        raise ValueError(f'Expected a 2x2 gate and a length-2 state, got {gate.shape} and {state.shape}')
    else:
        return (gate@state)



def zero_state(n=1):
    """Return |00...0> as a complex array of length 2**n."""
    out_array = np.array([1] + [0 for i in range(2**n - 1)], dtype=complex)
    return out_array

def basis_state(bits):
    """basis_state('01') returns |01>, a length-4 array with a 1 at index 1."""
    out_lst = [0]*(2**len(bits))
    index = int(bits,2)
    out_lst[index] = 1
    return np.array(out_lst,dtype=complex)
    

def normalize(state):
    """Divide by the norm so the probabilities sum to 1."""
    out = np.array(state, dtype=complex)
    norm = np.linalg.norm(out)
    if np.isclose(norm, 0):
        raise ValueError("Cannot Normalize ZERO state")
    return out/norm

def probabilities(state):
    """Return |amplitude|**2 for each basis state."""
    out_lst = []
    for i in state:
        out_lst.append((np.abs(i)**2))
    return np.array(out_lst)
