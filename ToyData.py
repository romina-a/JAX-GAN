import jax.random as random
import jax.numpy as jnp
import numpy as np
import os

SEED_DEFAULT = 100


class DataLoader:
    def get_next_batch(self):
        pass


class GaussianMixture(DataLoader):
    @staticmethod
    def create_2d_mean_matrix(num_components):
        a = int(np.sqrt(num_components))
        while a < num_components:
            if num_components % a == 0:
                break
            a += 1
        b = num_components // a
        return np.array([[i, j] for i in range(-a // 2 + 1, a // 2 + 1, 1) for j in range(-b // 2 + 1, b // 2 + 1, 1)])

    @staticmethod
    def create_2d_covariance_matrix(variance, num_components):
        return np.array([np.identity(2) * variance for _ in range(num_components)])

    def __init__(self, prng, batch_size, num_modes=None, variance=None, means=None, covariances=None):
        self.prng = prng
        self.batch_size = batch_size
        if means is not None:
            self.means = means
            self.num_modes = len(means)
        else:
            self.means = self.create_2d_mean_matrix(num_modes)
            self.num_modes = num_modes
        if covariances is not None:
            self.covariances = covariances
        else:
            self.covariances = self.create_2d_covariance_matrix(variance, self.num_modes)
        assert self.means.shape[0] == self.covariances.shape[0], "means and covariances must have equal length"
        assert self.means.shape[1] == self.covariances.shape[1], "means and covariances must have corresponding " \
                                                                 "dimensionality "

    def get_next_batch(self):
        self.prng, counts_key, shuffle_key, *keys = random.split(self.prng, self.num_modes + 3)
        numbs, counts = np.unique(random.randint(counts_key, (self.batch_size,), 0, self.num_modes), return_counts=True)

        batch = []
        for i, comp_ind in enumerate(numbs):
            samples = random.multivariate_normal(keys[i], self.means[comp_ind], self.covariances[comp_ind],
                                                 (counts[i],), jnp.float32)
            batch.extend(samples)
        batch = np.array(batch)
        batch = batch[random.permutation(shuffle_key, len(batch)), ]
        return batch

    def get_iteration_samples(self, num_iter):
        self.prng, counts_key, shuffle_key, *keys = random.split(self.prng, self.num_modes + 3)
        numbs, counts = np.unique(random.randint(counts_key, (self.batch_size * num_iter,), 0, self.num_modes),
                                  return_counts=True)
        batches = []
        for i, comp_ind in enumerate(numbs):
            print(f"creating {i}th components: {self.means[comp_ind]}->{counts[i]}")
            samples = random.multivariate_normal(keys[i], self.means[comp_ind], self.covariances[comp_ind],
                                                 (counts[i],), jnp.float32)
            batches.extend(samples)
        batches = np.array(batches)
        print(f"shuffling")
        batches = batches[random.permutation(shuffle_key, len(batches)), ]
        batches = batches.reshape((num_iter, self.batch_size, self.means[1].size))
        return batches


def get_gaussian_mixture(batch_size, num_iters, components, variance, seed=SEED_DEFAULT,
                         save=True, from_file=True):
    prng = random.PRNGKey(seed)
    path = f"./ToyData/GaussianMixture-{components}-{variance}-{batch_size}-{num_iters}.npy"
    if os.path.exists(path) and from_file:
        print("file exists")
        data = np.load(path)
    else:
        print("file didn't exist, creating")
        dl = GaussianMixture(prng, batch_size, components, variance)
        data = dl.get_iteration_samples(num_iters)
        if save:
            np.save(path, data)
    return data
