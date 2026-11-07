import numpy as np
import matplotlib
matplotlib.use('Agg')
import pytest
from qusim import gates, state, viz

S2 = 1 / np.sqrt(2)

@pytest.mark.parametrize('psi, expected', [
    ([1, 0], (0, 0, 1)),
    ([0, 1], (0, 0, -1)),
    ([S2, S2], (1, 0, 0)),
    ([S2, -S2], (-1, 0, 0)),
    ([S2, 1j * S2], (0, 1, 0)),
    ([S2, -1j * S2], (0, -1, 0)),
])
def test_bloch_cardinal_states(psi, expected):
    assert np.allclose(viz.bloch_coordinates(np.array(psi, dtype=complex)), expected)

def test_bloch_ignores_global_phase():
    psi = gates.h @ state.zero_state(1)
    assert np.allclose(viz.bloch_coordinates(np.exp(0.7j) * psi), viz.bloch_coordinates(psi))

def test_bloch_vector_has_unit_length():
    psi = state.normalize([0.3 + 0.1j, 0.2 - 0.5j])
    assert np.isclose(sum(c * c for c in viz.bloch_coordinates(psi)), 1.0)

def test_rx_half_pi_points_to_minus_y():
    psi = gates.rx(np.pi / 2) @ state.zero_state(1)
    assert np.allclose(viz.bloch_coordinates(psi), (0, -1, 0))

def test_plots_run_and_draw_bars():
    ax = viz.plot_histogram({'00': 3, '11': 5})
    assert len(ax.patches) == 2
    viz.plot_bloch(state.zero_state(1))