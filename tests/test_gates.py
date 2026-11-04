import numpy as np
import pytest
from qusim.gates import i,x,y,z,h,s,t

ALL_GATES = [i,x,y,z,h,s,t]
@pytest.mark.parametrize('Gate', ALL_GATES)
def test_gate_is_unitary(Gate):
    conjT_matrix = Gate.conj().T
    assert np.allclose(conjT_matrix@Gate, i)

def test_HadamardGate_is_Own_Inverse():
    assert np.allclose(h@h,i)

def test_X_and_Z_anticommute():
    assert np.allclose(x@z, z@x*(-1))
