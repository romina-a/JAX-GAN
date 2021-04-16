import numpy as np
import matplotlib.pyplot as plt


def visualize_sample_heatmap(samples):
    X = samples[:, 0]
    Y = samples[:, 1]
    heatmap, x_edges, y_edges = np.histogram2d(X, Y, bins=[50, 50])
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def plot_samples_scatter(samples):
    X = samples[:, 0]
    Y = samples[:, 1]

    plt.scatter(X, Y)
    plt.show()