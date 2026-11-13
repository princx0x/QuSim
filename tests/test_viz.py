"""Tests for Bloch coordinates, the Bloch plot and the histogram plot."""

import matplotlib

# Use a non-interactive backend so tests do not open plot windows.
matplotlib.use('Agg')

import numpy as np
import pytest

from qusim import gates, state, viz

S2 = 1 / np.sqrt(2)


@pytest.mark.parametrize(
    'psi, expected',
    [
        ([1, 0], (0, 0, 1)),
        ([0, 1], (0, 0, -1)),
        ([S2, S2], (1, 0, 0)),
        ([S2, -S2], (-1, 0, 0)),
        ([S2, 1j * S2], (0, 1, 0)),
        ([S2, -1j * S2], (0, -1, 0)),
    ],
)
def test_bloch_cardinal_states(psi, expected):
    """The six cardinal states point along the coordinate axes."""
    assert np.allclose(viz.bloch_coordinates(np.array(psi, dtype=complex)), expected)


def test_bloch_ignores_global_phase():
    """A global phase does not move the Bloch vector."""
    psi = gates.h @ state.zero_state(1)
    assert np.allclose(
        viz.bloch_coordinates(np.exp(0.7j) * psi),
        viz.bloch_coordinates(psi),
    )


def test_bloch_vector_has_unit_length():
    """Pure states have a Bloch vector of length 1."""
    psi = state.normalize([0.3 + 0.1j, 0.2 - 0.5j])
    assert np.isclose(sum(c * c for c in viz.bloch_coordinates(psi)), 1.0)


def test_rx_half_pi_points_to_minus_y():
    """Rx(pi/2)|0> points along the -Y axis."""
    psi = gates.rx(np.pi / 2) @ state.zero_state(1)
    assert np.allclose(viz.bloch_coordinates(psi), (0, -1, 0))


def test_plots_run_and_draw_bars():
    """The plotting helpers run and draw one bar per outcome."""
    ax = viz.plot_histogram({'00': 3, '11': 5})
    assert len(ax.patches) == 2
    viz.plot_bloch(state.zero_state(1))
