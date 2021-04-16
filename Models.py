from jax.nn.initializers import normal
from jax.nn import leaky_relu
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, ConvTranspose, Dense,
                                   Tanh, Relu, Flatten)
import jax.numpy as jnp
import jax.random as random

from jax import value_and_grad, jit
from functools import partial


# ~~~~~~~~~~~~ helper functions ~~~~~~~~~~~~~~~~~~~~~~~~
def print_param_dims(params):
    for a in params:
        if len(a) == 0:
            print("()", end='')
        for b in a:
            print(b.shape, end=',')
        print()


# ~~~~~~~~~~~~ losses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def BCE_from_logits(logits, desired_labels):
    return jnp.mean(
        jnp.log(1 + jnp.exp(-logits)) * desired_labels +
        jnp.log(1 + jnp.exp(logits)) * (1 - desired_labels)
    )


def MSE(logits, desired_values):
    return jnp.mean((logits - desired_values) ** 2)


# ---------------------------- layers with stax convention --------------------------
def Reshape(output_shape):
    def init_fun(rng, input_shape):
        size_in = 1
        for a in input_shape[1:]: size_in = size_in * a
        size_out = 1
        for a in output_shape: size_out = size_out * a
        assert size_out == size_in, "input and output sizes must match"
        return (input_shape[0], *output_shape[:]), ()

    def apply_fun(params, inputs, **kwargs):
        return jnp.reshape(inputs, (inputs.shape[0], *output_shape))

    return init_fun, apply_fun


def LeakyRelu(negative_slope):
    return stax.elementwise(leaky_relu, negative_slope=negative_slope)


# -----------------------------------   Models   ------------------------------------
def conv_generator_mnist():
    model = stax.serial(
        Dense(1024 * 7 * 7),
        Reshape((7, 7, 1024)),
        ConvTranspose(out_chan=512, filter_shape=(5, 5), strides=(1, 1),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Relu, BatchNorm(),
        ConvTranspose(out_chan=256, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Relu, BatchNorm(),
        ConvTranspose(out_chan=128, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Relu, BatchNorm(),
        ConvTranspose(out_chan=1, filter_shape=(5, 5), strides=(1, 1),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Tanh,
    )
    return model


def conv_generator_cifar10():
    model = stax.serial(
        Dense(1024 * 2 * 2),
        Reshape((2, 2, 1024)),
        ConvTranspose(out_chan=512, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Relu, BatchNorm(),
        ConvTranspose(out_chan=256, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Relu, BatchNorm(),
        ConvTranspose(out_chan=128, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Relu, BatchNorm(),
        ConvTranspose(out_chan=3, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=None, b_init=normal(1e-6)),
        Tanh,
    )
    return model


def conv_discriminator():
    model = stax.serial(
        Conv(out_chan=64, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=None, b_init=normal(1e-6)),
        LeakyRelu(negative_slope=0.2),
        Conv(out_chan=128, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=None, b_init=normal(1e-6)),
        LeakyRelu(negative_slope=0.2), BatchNorm(),
        Conv(out_chan=256, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=None, b_init=normal(1e-6)),
        LeakyRelu(negative_slope=0.2), BatchNorm(),
        Conv(out_chan=512, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=None, b_init=normal(1e-6)),
        LeakyRelu(negative_slope=0.2), BatchNorm(), Flatten,
        Dense(1)
    )
    return model


def mlp_discriminator():
    model = stax.serial(
        Dense(out_dim=256), LeakyRelu(negative_slope=0.2),
        BatchNorm(axis=(1,)),
        Dense(out_dim=256), LeakyRelu(negative_slope=0.2),
        BatchNorm(axis=(1,)),
        Dense(out_dim=256), LeakyRelu(negative_slope=0.2),
        BatchNorm(axis=(1,)),
        Dense(out_dim=256), LeakyRelu(negative_slope=0.2),
        BatchNorm(axis=(1,)),
        Dense(1)
    )
    return model


def mlp_generator_2d():
    model = stax.serial(
        Dense(out_dim=256), Relu,
        BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        BatchNorm(axis=(1,)),
        Dense(2)
    )
    return model


# ----------------------------------- GAN --------------------------------------------
class GAN:
    r"""
    GAN implementation using jax.experimental
    generator and discriminator are jax.experimental.stax models: (init_func, apply_func) pairs
    optimizers are jax.experimental.optimizers optimizers: (init, update, get_params) triplets
    """
    def __init__(self, d, g, d_opt, g_opt, loss_function):
        """

        :param d: (dictionary) keys: 'init' and 'apply' representing a jax.experimental.stax model
        :param g: (dictionary) keys: 'init' and 'apply' representing a jax.experimental.stax model
        :param d_opt: (dictionary) keys: 'init', 'update', and 'get_params' representing a jax.experimental.optimizer
        :param g_opt: (dictionary) keys: 'init', 'update', and 'get_params' representing a jax.experimental.optimizer
        :param loss_function: (function) to calculate loss from discriminator outputs:
                              (discriminator-outputs, real-labels)-> loss
        """
        self.d = {'init': d['init'], 'apply': d['apply']}
        self.g = {'init': g['init'], 'apply': g['apply']}
        self.d_opt = {'init': d_opt['init'], 'update': d_opt['update'], 'get_params': d_opt['get_params']}
        self.g_opt = {'init': g_opt['init'], 'update': g_opt['update'], 'get_params': g_opt['get_params']}
        self.loss_function = loss_function
        self.d_output_shape = None
        self.g_output_shape = None
        self.d_input_shape = None
        self.g_input_shape = None

    def init(self, prng_d, prng_g, d_input_shape, g_input_shape):
        """

        :param prng_d: (jax.PRNGKey) for discriminator initialization
        :param prng_g: (jax.PRNGKey) for generator initialization
        :param d_input_shape: (tuple) shape of the discriminator input excluding batch size
        :param g_input_shape: (tuple) shape of the generator input excluding batch size
        :return: discriminator and generator states (needed for train_step and generate_samples)
        """
        self.g_input_shape = g_input_shape
        self.d_input_shape = d_input_shape
        self.d_output_shape, d_params = self.d['init'](prng_d, (1, *d_input_shape))
        self.g_output_shape, g_params = self.g['init'](prng_g, (1, *g_input_shape))
        d_state = self.d_opt['init'](d_params)
        g_state = self.g_opt['init'](g_params)
        return d_state, g_state

    @partial(jit, static_argnums=(0, 4,))
    def _d_loss(self, d_params, g_params, prng_key, batch_size, real_samples):
        z = random.normal(prng_key, (batch_size, *self.g_input_shape))
        fake_ims = self.g['apply'](g_params, z)

        fake_predictions = self.d['apply'](d_params, fake_ims)
        real_predictions = self.d['apply'](d_params, real_samples)

        fake_loss = self.loss_function(fake_predictions, jnp.zeros(batch_size))
        real_loss = self.loss_function(real_predictions, jnp.ones(batch_size))

        return fake_loss + real_loss

    @partial(jit, static_argnums=(0, 4,))
    def _g_loss(self, g_params, d_params, prng_key, batch_size):
        z = random.normal(prng_key, (batch_size, *self.g_input_shape))
        fake_ims = self.g['apply'](g_params, z)

        fake_predictions = self.d['apply'](d_params, fake_ims)

        loss = self.loss_function(fake_predictions, jnp.ones(batch_size))

        return loss

    @partial(jit, static_argnums=(0, 6,))
    def train_step(self, i, prng_key, d_state, g_state, real_samples, batch_size):
        """
        !: call init function before train_step

        :param i: (int) step number
        :param prng_key: (jax.random.PRNGKey) used to create random samples from the generator
        :param d_state: previous discriminator state
        :param g_state: previous generator state
        :param real_samples: (np/jnp array) samples form the training set
        :param batch_size: (int)
        :return: updated discriminator and generator states and discriminator and generator loss values
        """
        prng1, prng2 = random.split(prng_key, 2)
        d_params = self.d_opt['get_params'](d_state)
        g_params = self.g_opt['get_params'](g_state)

        d_loss_value, d_grads = value_and_grad(self._d_loss)(d_params, g_params, prng1, batch_size, real_samples)
        d_state = self.d_opt['update'](i, d_grads, d_state)

        g_loss_value, g_grads = value_and_grad(self._g_loss)(g_params, d_params, prng2, batch_size)
        g_state = self.g_opt['update'](i, g_grads, g_state)

        return d_state, g_state, d_loss_value, g_loss_value

    @partial(jit, static_argnums=(0,))
    def generate_samples(self, z, g_state):
        """

        :param z: (np/jnp array) shape: (n, generator_input_dims)
        :param g_state: generator state
        :return: (np/jnp array) shape: (n, generator_output_dims) n generated samples
        """
        fakes = self.g['apply'](self.g_opt['get_params'](g_state), z)
        return fakes

