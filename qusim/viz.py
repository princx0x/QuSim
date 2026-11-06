import numpy as np
import matplotlib.pyplot as plt
from .measure import _test_valid_state

def plot_histogram(counts):
    """Bar chart of measurement counts (a measurement histogram)."""
    keys = sorted(counts)
    values = [counts[i] for i in keys]
    bars = plt.bar(keys, values)
    plt.xlabel('Outcome')
    plt.ylabel('Counts')
    plt.bar_label(bars)

def bloch_coordinates(state):
    """Return (x, y, z) for a single-qubit state."""
    _test_valid_state(state)
    alpha = state[0]
    beta = state[1]
    prod = np.conj(alpha)*beta
    x = round(float(prod.real*2),8)+0.0
    y = round(float(prod.imag*2),8)+0.0
    z = round(float(np.abs(alpha)**2 - np.abs(beta)**2),8)
    return (x,y,z)

    

def plot_bloch(state, ax=None):
    """Draw the Bloch sphere with an arrow for the state."""
    _test_valid_state(state)
    if ax is None:
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111,projection='3d')
    u = np.linspace(0, 2*np.pi)
    v = np.linspace(0, np.pi)
    X = np.outer(np.cos(u), np.sin(v))     
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(X,Y,Z,
        color='gray',
        alpha=0.3,
        linewidth=0.5)
    ax.plot([-1, 1], [0, 0], [0, 0], color='red', label='X-axis')
    ax.plot([0, 0], [-1, 1], [0, 0], color='green', label='Y-axis')
    ax.plot([0, 0], [0, 0], [-1, 1], color='blue', label='Z-axis')
    x,y,z, = bloch_coordinates(state)
    ax.quiver(
        0,0,0,
        x,y,z
    )
     # Set axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_title('Bloch Sphere')

    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    plt.legend()

    return ax