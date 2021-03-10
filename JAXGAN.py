import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random
from functools import partial
from jax.scipy.special import logsumexp

from torch.utils import data
from torchvision.datasets import MNIST

import time

from matplotlib import pyplot as plt


# ~~~~~~~~~~~~~~~~~~~~~~~~~ LOADING MNIST WITH PYTORCH ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class NumpyLoader(data.DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ START OF GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


dist_dim = 100
disc_layer_sizes = [784, 512, 256, 2]
gen_layer_sizes = [dist_dim, 256, 512, 784]
# param_scale = 0.1
step_size = 0.01
num_epochs = 1
batch_size = 128
n_targets = 10


def relu(x):
    return jnp.maximum(0, x)


# @partial(vmap, in_axes=(None, 0), out_axes=0) TODO: why doesn't work?
def disc_predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)
    # return jnp.exp(logits)/sum(jnp.exp(logits))


batched_disc_predict = vmap(disc_predict, in_axes=(None, 0), out_axes=0)


def gen_generate(params, noise):
    # per-example predictions
    activations = noise
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return jnp.tanh(logits)


batched_gen_generate = vmap(gen_generate, in_axes=(None, 0), out_axes=0)


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_disc_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def disc_loss(d_params, images, targets):
    preds = batched_disc_predict(d_params, images)
    # return -jnp.mean(preds * targets)
    return jnp.mean(preds * targets)


# @jit
def update_disc(d_params, images, labels):
    grads = grad(disc_loss)(d_params, images, labels)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(d_params, grads)], grads


def gen_loss(g_params, d_params, g_noise):
    ims = batched_gen_generate(g_params, g_noise)
    # plt.imshow(jnp.reshape(ims[0], (28, 28)))
    # plt.show()
    preds = batched_disc_predict(d_params, ims)
    return -jnp.mean(preds * one_hot(jnp.ones(len(preds)), 1))
    # return -jnp.mean(preds * one_hot(jnp.zeros(len(preds)), 1))


# @jit
def update_gen(g_params, d_params, gen_noise):
    grads = grad(gen_loss)(g_params, d_params, gen_noise)

    return [(w + step_size * dw, b + step_size * db)
            for (w, b), (dw, db) in zip(g_params, grads)], grads


mnist_dataset = MNIST('./tmp/mnist/', download=True, transform=FlattenAndCast())
# load training with the generator (makes batch easier I think)
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)
train_images = np.array(mnist_dataset.train_data).reshape(len(mnist_dataset.train_data), -1)
train_labels = one_hot(np.array(mnist_dataset.train_labels), n_targets)
# Get full test dataset
# mnist_dataset_test = MNIST('./tmp/mnist/', download=True, train=False)
# test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1),
#                         dtype=jnp.float32)
# test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)

disc_params = init_network_params(disc_layer_sizes, random.PRNGKey(0))
gen_params = init_network_params(gen_layer_sizes, random.PRNGKey(0))
for epoch in range(num_epochs):
    start_time = time.time()
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    for real_images, _ in training_generator:
        noise = random.normal(key, (batch_size, dist_dim), dtype=jnp.float32)
        key, subkey = random.split(subkey)
        fake_images = batched_gen_generate(gen_params, noise)
        disc_t_imgs = jnp.concatenate([real_images, fake_images])
        disc_t_lbls = jnp.concatenate([one_hot(jnp.ones(len(real_images)), 2),
                                       one_hot(jnp.zeros(len(fake_images)), 2)])
        # disc_t_lbls = disc_t_lbls + 0.05 * random.uniform(key, disc_t_lbls.shape)
        # key, subkey = random.split(subkey)
        disc_t_lbls = 0.9 * disc_t_lbls
        # print(f'disc weight before update:{gen_params[0][0][0:3,0:1]}')
        disc_params, gradd = update_disc(disc_params, disc_t_imgs, disc_t_lbls)
        # print(f'disc weight after update:{gen_params[0][0][0:3,0:1]}')
        print(f'disc grad after update:{gradd[0][0][0:3, 0:1]}')
        noise = random.normal(key, (batch_size, dist_dim), dtype=jnp.float32)
        key, subkey = random.split(subkey)
        # print(f'gen weight before update:{gen_params[0][0][0:3,0:1]}')
        gen_params, gradg = update_gen(gen_params, disc_params, noise)
        # print(f'gen weight after update:{gen_params[0][0][0:3,0:1]}')
        print(f'gen grad after update:{gradg[0][0][0:3,0:1]}')
    epoch_time = time.time() - start_time

    train_acc = accuracy(disc_params, train_images, one_hot(np.ones(len(train_images)), 2))
    # test_acc = accuracy(disc_params, test_images, test_labels)
    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    # print("Test set accuracy {}".format(test_acc))
    noise = random.normal(key, (1, dist_dim), dtype=jnp.float32)
    key, subkey = random.split(subkey)
    fake_image = batched_gen_generate(gen_params, noise)
    plt.imshow(jnp.reshape(fake_image, (28, 28)))
    plt.show()