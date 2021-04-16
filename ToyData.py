import jax.random as random
import jax.numpy as jnp
import numpy as np


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
