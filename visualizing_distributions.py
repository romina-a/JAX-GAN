import numpy as np
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp


def plot_sample_heatmap(samples):
    X = samples[:, 0]
    Y = samples[:, 1]
    heatmap, x_edges, y_edges = np.histogram2d(X, Y, bins=[100, 100])
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def plot_samples_scatter(samples, samples2=None, samples_ratings=None, save_adr=None, show=True, cmap=None):
    X = samples[:, 0],
    Y = samples[:, 1]
    if samples_ratings is not None:
        cmap = cmap or 'cividis'

    plt.scatter(X, Y, c=samples_ratings, alpha=0.2, cmap=cmap)

    if samples2 is not None:
        X2 = samples2[:, 0]
        Y2 = samples2[:, 1]

        plt.scatter(X2, Y2, color='red', alpha=0.2)

    if samples_ratings is not None:
        plt.colorbar()
    if save_adr is not None:
        plt.savefig(save_adr)
    if show:
        plt.show()
    else:
        plt.clf()