import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from Models import GAN
from ToyData import GaussianMixture, Circular
import argparse


# ---------------------------- Visualization -------------------------------

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
    if samples_ratings is not None:
        plt.colorbar()

    if samples2 is not None:
        X2 = samples2[:, 0]
        Y2 = samples2[:, 1]

        plt.scatter(X2, Y2, color='red', alpha=0.2)

    if save_adr is not None:
        plt.savefig(save_adr)
    if show:
        plt.show()
    else:
        plt.clf()


def visualize_gan_state(gan, d_state, g_state, ls_start=-6, ls_stop=6, ls_num=100, apply_sigmoid=True, plot_gen=True):
    z = jax.random.normal(jax.random.PRNGKey(0), (1000, 2))
    x, y = jnp.meshgrid(jnp.linspace(ls_start, ls_stop, ls_num), jnp.linspace(ls_start, ls_stop, ls_num))
    grid = jnp.concatenate((x.reshape((x.size, 1)), y.reshape((y.size, 1))), axis=1)
    ratings = gan.rate_samples(grid, d_state)
    if apply_sigmoid:
        ratings = sigmoid(ratings)
    if plot_gen:
        plot_samples_scatter(samples=grid, samples2=gan.generate_samples(z, g_state),
                             samples_ratings=ratings)
    else:
        plot_samples_scatter(samples=grid, samples_ratings=ratings)


# ---------------------------- Gaussian Mixture Evaluation -------------------------------
def _get_z(num, seed):
    prng = jax.random.PRNGKey(seed)
    z = jax.random.normal(prng, (num, 2))
    return z


def evaluate_gan_gaussian(gan, d_state, g_state, num_modes, var, seed=0):
    z = _get_z(10000, seed)
    fake_samples = np.array(gan.generate_samples(z, g_state))
    modes = GaussianMixture.create_2d_mean_matrix(num_modes)
    sd = np.sqrt(var)

    mode_inds, dists = _get_nearest_modes(fake_samples, modes)
    recovered_modes = np.unique(mode_inds[dists < 4 * sd])
    high_quality_samples = mode_inds[dists < 4 * sd]
    print(f"num of recovered modes:{len(recovered_modes)}")
    print(f"high guality samples:{len(high_quality_samples) / len(dists) * 100} %")
    print(f"samples within sd: "
          f"[0,1):{len(dists[dists < sd]) / len(dists) * 100}, "
          f"[1,2):{len(dists[(dists >= sd) & (dists < sd * 2)]) / len(dists) * 100}, "
          f"[2,3):{len(dists[(dists >= sd * 2) & (dists < sd * 3)]) / len(dists) * 100}, "
          f"[3,4):{len(dists[(dists >= sd * 3) & (dists < sd * 4)]) / len(dists) * 100}, "
          f"[4,inf):{len(dists[(dists >= sd * 4)]) / len(dists) * 100}")

    mode, counts = np.unique(mode_inds[dists <= 0.2], return_counts=True)
    plt.bar(mode, counts)
    plt.savefig("./distribution-of-high-quality-samples-per-mode")
    plt.show()

    return mode_inds, dists


def _get_nearest_modes(samples, modes):
    mode_inds = [-1 for _ in range(len(samples))]
    dists = [-1 for _ in range(len(samples))]
    for i, sample in enumerate(samples):
        mode_inds[i], dists[i] = _get_nearest_mode(sample, modes)
    return np.array(mode_inds), np.array(dists)


def _get_nearest_mode(sample, modes):
    dist = np.sqrt((sample[0] - modes[0][0]) ** 2 + (sample[1] - modes[0][1]) ** 2)
    mode_ind = 0
    for i, mode in enumerate(modes):
        d = np.sqrt((sample[0] - mode[0]) ** 2 + (sample[1] - mode[1]) ** 2)
        if d < dist:
            dist = d
            mode_ind = i
    return mode_ind, dist


# ---------------------------- Circles Evaluation -----------------------------------------

def evaluate_gan_circular(gan, d_state, g_state, num_circles, var, seed, ):
    z = _get_z(10000, seed)
    fake_samples = np.array(gan.generate_samples(z, g_state))
    sd = np.sqrt(var)
    data_rads = Circular.create_radius_array(num_circles)
    sample_rads = np.sqrt(fake_samples[:, 0] ** 2 + fake_samples[:, 1] ** 2)

    temp = np.array([np.abs(r - data_rads) for r in sample_rads])
    mode_ids = np.argmin(temp, 1)
    dists = np.array([np.abs(r - data_rads[mode_ids[i]]) for i, r in enumerate(sample_rads)])

    samples_degs = np.arctan2(fake_samples[:, 1], fake_samples[:, 0])

    print(f"high guality samples:{len(dists[dists < 4 * sd]) / len(dists) * 100} %")

    m, c = np.unique(mode_ids[dists < 4 * sd], return_counts=True)
    print(f"count ratio per circle:{[count / np.min(c) for count in c]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_gan", required=True, type=str, help="path to the GAN pickle file")
    parser.add_argument("--num_components", required=True, type=int, help="number of modes of the data the GAN is "
                                                                          "trained on")
    parser.add_argument("--variance", required=True, type=float, help="the variance of the data the GAN is trained on")
    parser.add_argument("--seed", required=False, default=0, type=int, help="seed for generating test data")
    parser.add_argument("--data", required=True, type=str, choices={'gaussian_mixture', 'circle'},
                        help="the training data to evaluate based on")
    args = vars(parser.parse_args())
    gan, d_state, g_state = GAN.load_gan_from_file(args['path_to_gan'])
    if args['data'] == 'gaussian_mixture':
        _ = evaluate_gan_gaussian(gan, d_state, g_state,
                                  num_modes=args['num_components'],
                                  var=args['variance'],
                                  seed=args['seed'])
    if args['data'] == 'circle':
        evaluate_gan_circular(gan, d_state, g_state,
                              num_circles=args['num_components'],
                              var=args['variance'],
                              seed=args['seed'])
