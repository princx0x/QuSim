import numpy as np

#CONSTANTS FOR VARIOUS PURPOSES
atol = 1e-9 #tolerance for magnitude of the state

#MATRICES FOR GATES
i = np.array([[1,0],[0,1]],dtype=complex)
x = np.array([[0,1],[1,0]],dtype=complex)
y = np.array([[0,-1j],[1j,0]],dtype=complex)
z = np.array([[1,0],[0,-1]],dtype=complex)
h = np.array([[1,1],[1,-1]]) / (np.sqrt(2))
h = np.array(h,dtype=complex)
s = np.array([[1,0],[0,1j]],dtype=complex)
t = np.array([[1,0],[0,(1+1j)/np.sqrt(2)]],dtype=complex)

def apply_gate(gate,state):
    state = np.asarray(state, dtype=complex)
    norm = np.vdot(state, state).real
    if not (np.isclose(norm,1.0,atol)):
        raise ValueError(f'Expected a state of magnitude 1, got magnitude = {norm}')
    elif gate.shape != (2, 2) or state.shape != (2,):
        raise ValueError(f'Expected a 2x2 gate and a length-2 state, got {gate.shape} and {state.shape}')
    else:
        return (gate@state)

def X(state):
    return apply_gate(x,state)

def Y(state):
    return apply_gate(y,state)

def Z(state):
    return apply_gate(z,state)

def H(state):
    return apply_gate(h,state)

def S(state):
    return apply_gate(s,state)

def T(state):
    return apply_gate(t,state)