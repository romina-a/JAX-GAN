import jax.numpy as jnp
import numpy as np
from jax import jit, vmap, value_and_grad
from jax import random
from functools import partial
from jax.experimental import stax
from jax.experimental import optimizers

from torch.utils import data
from torchvision.datasets import MNIST

import time
import _pickle as pickle

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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ Helper Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def one_hot(x, k, dtype=jnp.float32):
#     """Create a one-hot encoding of x of size k."""
#     return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = batched_disc_predict(params, images)
    return jnp.mean(predicted_class == target_class)


def save_obj(obj, filename, adr='./'):
    with open(adr + filename + '.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output)


def load_obj(filename, adr='./'):
    with open(adr + filename + '.pkl', 'rb') as input:
        obj = pickle.load(input)
    return obj


def save_params(params, filename, adr='./'):
    save_obj(params, filename, adr)


def load_params(filename, adr='./'):
    params = []
    temp_params = load_obj(filename, adr)
    for (w, b) in temp_params:
        params.append((jnp.array(w), jnp.array(b)))
    return params


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ START OF GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scale_default = 2e-2
dist_dim = 100
d_layer_sizes = [784, 512, 256, 1]
g_layer_sizes = [dist_dim, 256, 512, 784]
# param_scale = 0.1
d_step_size = 0.0002
g_step_size = 0.0002
num_epochs = 1000
batch_size = 128
# n_targets = 10
digit = 0
minimize = True


def random_layer_params(m, n, key, scale=scale_default):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def relu(x):
    return jnp.maximum(0, x)


def sigmoid(x):
    return jnp.exp(x) / (1. + jnp.exp(x))


def BCELoss(predictions, targets):
    return -jnp.mean(jnp.log(predictions) * targets + jnp.log(1 - predictions) * (1 - targets))


# @partial(vmap, in_axes=(None, 0), out_axes=0) TODO: why doesn't work?
def disc_predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logit = jnp.dot(final_w, activations) + final_b
    return sigmoid(logit)
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


def disc_loss(d_params, images, targets):
    preds = batched_disc_predict(d_params, images)
    return BCELoss(preds, targets)


@jit
def update_disc(d_params, images, labels):
    """

    :param g_params:
    :param d_params:
    :param g_noise:
    :return: params, loss value, grads
    """
    dics_loss_value, d_grads = value_and_grad(disc_loss)(d_params, images, labels)
    return [(w - d_step_size * dw, b - d_step_size * db)
            for (w, b), (dw, db) in zip(d_params, d_grads)], dics_loss_value, d_grads


def gen_loss(g_params, d_params, g_noise):
    fake_imgs = batched_gen_generate(g_params, g_noise)
    # plt.imshow(jnp.reshape(fake_imgs[0], (28, 28)))
    # plt.show()
    return disc_loss(d_params, fake_imgs, jnp.ones(len(fake_imgs)))
    # return -jnp.mean(preds * one_hot(jnp.zeros(len(preds)), 1))


@jit
def update_gen(g_params, d_params, g_noise):
    """

    :param g_params:
    :param d_params:
    :param g_noise:
    :return: params, loss value, grads
    """
    g_loss_value, g_grads = value_and_grad(gen_loss)(g_params, d_params, g_noise)

    return [(w - g_step_size * dw, b - g_step_size * db)
            for (w, b), (dw, db) in zip(g_params, g_grads)], g_loss_value, g_grads


# ~~~~~~~~~~~~~~~~~~~~~~~ Data Loader, I don't understand exactly ~~~~~~~~~~~~~~~~~~~~
data_adr = ""
mnist_dataset = MNIST('./tmp/mnist/', download=True, transform=FlattenAndCast())
# load training with the generator (makes batch easier I think)
training_generator = NumpyLoader(mnist_dataset, batch_size=batch_size, num_workers=0)


# Get full test dataset
# mnist_dataset_test = MNIST('./tmp/mnist/', download=True, train=False)
# test_images = jnp.array(mnist_dataset_test.test_data.numpy().reshape(len(mnist_dataset_test.test_data), -1),
#                         dtype=jnp.float32)
# test_labels = one_hot(np.array(mnist_dataset_test.test_labels), n_targets)


# ~~~~~~~~~~~~~~~~~~~~~~~~ Train GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_gan():
    key = random.PRNGKey(0)
    key, dkey, gkey = random.split(key, 3)
    d_params = init_network_params(d_layer_sizes, dkey)
    g_params = init_network_params(g_layer_sizes, gkey)
    g_loss_history = []
    d_loss_history = []
    g_loss_epoch = []
    d_loss_epoch = []
    for epoch in range(num_epochs):
        start_time = time.time()
        for real_images, _ in training_generator:
            real_images = (real_images-127.5)/127.5
            key, subkey = random.split(key)
            noise = random.normal(subkey, (batch_size, dist_dim), dtype=jnp.float32)
            fake_images = batched_gen_generate(g_params, noise)

            d_t_imgs = jnp.concatenate([real_images, fake_images])
            d_t_lbls = jnp.concatenate([0.9 * jnp.ones(len(real_images)), jnp.zeros(len(fake_images))])

            # d_t_lbls = 0.9 * d_t_lbls
            d_params, d_loss, d_grad = update_disc(d_params, d_t_imgs, d_t_lbls)
            # print(f'disc grad after update:{d_grad[0][0][0:3, 0:1]}')
            # print(f'disc loss:{d_loss}')

            # TODO: to create new noise or not to create new noise?
            #  I believe in all codes I have seen and in the paper, new noise is created, but why?
            key, subkey = random.split(key)
            noise = random.normal(subkey, (batch_size, dist_dim), dtype=jnp.float32)

            g_params, g_loss, g_grad = update_gen(g_params, d_params, noise)

            d_loss_epoch.append(d_loss)
            g_loss_epoch.append(g_loss)
            # print(f'gen grad after update:{g_grad[0][0][0:3,0:1]}')
            # print(f'gen loss:{g_loss}')

        epoch_time = time.time() - start_time
        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        g_loss_history.append(np.mean(g_loss_epoch))
        d_loss_history.append(np.mean(d_loss_epoch))
        g_loss_epoch = []
        d_loss_epoch = []

        if epoch % 1 == 0:
            key = random.PRNGKey(0)
            noise = random.normal(key, (1, dist_dim), dtype=jnp.float32)

            fake_image = batched_gen_generate(g_params, noise)*127.5+127.5
            plt.imshow(jnp.reshape(fake_image, (28, 28)))
            plt.show()

    print('ended')
    save_params(g_params, 'g_params')
    save_params(d_params, 'd_params')
    save_obj(g_loss_history, 'g_loss_history')
    save_obj(d_loss_history, 'd_loss_history')


def load_generator_and_generate_im(n=1):
    g_params = load_params('g_params')
    key = random.PRNGKey(0)
    noise = random.normal(key, (n, dist_dim), dtype=jnp.float32)
    ims = batched_gen_generate(g_params, noise)
    ims = ims * 127.5 + 127.5
    ims = ims.reshape(-1, 28, 28)
    return ims


def load_and_plot_history():
    d_loss_history = load_obj('d_loss_history')
    g_loss_history = load_obj('g_loss_history')
    plt.plot(d_loss_history, label='discriminator_objective')
    plt.plot(g_loss_history, label='generator_objective')
    plt.show()


if __name__ == '__main__':
    train_gan()
