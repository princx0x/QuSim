import matplotlib.pyplot as plt

def plot_histogram(counts):
    """Bar chart of measurement counts (a measurement histogram)."""
    keys = sorted(counts)
    values = [counts[i] for i in keys]
    bars = plt.bar(keys, values)
    plt.xlabel('Outcome')
    plt.ylabel('Counts')
    plt.bar_label(bars)