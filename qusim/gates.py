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

def rx(theta):
    """Rotation about the X axis by angle theta. Returns a 2x2 complex matrix."""
    c = np.cos(theta/2)
    s =  -1*np.sin(theta/2)*1j
    matrix = np.array([[c,s],
                       [s,c]],dtype = complex)
    return matrix

def ry(theta):
    """Rotation about the Y axis by angle theta. Returns a 2x2 complex matrix."""
    c = np.cos(theta/2)
    s =  -1*np.sin(theta/2)
    matrix = np.array([[c,s],
                    [-1*s,c]],dtype = complex)
    return matrix

def rz(theta):
    """Rotation about the Z axis by angle theta. Returns a 2x2 complex matrix."""
    E = np.exp(1j*(theta/2)*(-1))
    matrix = np.array([[E,0],
                       [0,1/E]],dtype = complex)
    return matrix

def phase(phi):
    """Phase gate diag(1, e^(i*phi)). Returns a 2x2 complex matrix."""
    matrix = np.array([[1,0],
                       [0,np.exp(1j*phi)]],dtype=complex)
    return matrix