import numpy as np
from .state import apply_gate



#MATRICES FOR GATES
i = np.array([[1,0],[0,1]],dtype=complex)
x = np.array([[0,1],[1,0]],dtype=complex)
y = np.array([[0,-1j],[1j,0]],dtype=complex)
z = np.array([[1,0],[0,-1]],dtype=complex)
h = np.array([[1,1],[1,-1]],dtype=complex) / (np.sqrt(2))
s = np.array([[1,0],[0,1j]],dtype=complex)
t = np.array([[1,0],[0,(1+1j)/np.sqrt(2)]],dtype=complex)


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