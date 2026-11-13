"""Plotting helpers for measurement counts and single-qubit Bloch spheres."""

import matplotlib.pyplot as plt
import numpy as np

from .measure import _test_valid_state


def plot_histogram(counts, ax=None):
    """Draw a bar chart of measurement counts and return the axis."""
    if ax is None:
        _, ax = plt.subplots()
    keys = sorted(counts)
    values = [counts[key] for key in keys]
    bars = ax.bar(keys, values)
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Counts')
    ax.bar_label(bars)
    return ax


def bloch_coordinates(state):
    """Return the (x, y, z) Bloch vector for a single-qubit state."""
    _test_valid_state(state)
    alpha = state[0]
    beta = state[1]
    product = np.conj(alpha) * beta
    x = round(float(product.real * 2), 8) + 0.0
    y = round(float(product.imag * 2), 8) + 0.0
    z = round(float(np.abs(alpha) ** 2 - np.abs(beta) ** 2), 8)
    return (x, y, z)


def plot_bloch(state, ax=None):
    """Draw the Bloch sphere and an arrow for ``state``; return the 3D axis."""
    _test_valid_state(state)
    if ax is None:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

    # Wireframe sphere built from spherical coordinates.
    u = np.linspace(0, 2 * np.pi)
    v = np.linspace(0, np.pi)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(X, Y, Z, color='gray', alpha=0.3, linewidth=0.5)

    # Coordinate axes.
    ax.plot([-1, 1], [0, 0], [0, 0], color='red', label='X-axis')
    ax.plot([0, 0], [-1, 1], [0, 0], color='green', label='Y-axis')
    ax.plot([0, 0], [0, 0], [-1, 1], color='blue', label='Z-axis')

    # State vector drawn as an arrow from the origin.
    x, y, z = bloch_coordinates(state)
    ax.quiver(0, 0, 0, x, y, z)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Bloch Sphere')
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    return ax
