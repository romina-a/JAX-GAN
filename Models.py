from jax.nn.initializers import normal
from jax.nn import leaky_relu, sigmoid
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, ConvTranspose, Dense,
                                   Tanh, Relu, Flatten, Sigmoid)
from jax.experimental.optimizers import pack_optimizer_state, unpack_optimizer_state
import jax.numpy as jnp
import jax.random as random

from jax.lax import sort

from jax import value_and_grad, jit
from functools import partial
import pickle
import os

EPSILON = 1e-10


# ~~~~~~~~~~~~ helper functions ~~~~~~~~~~~~~~~~~~~~~~~~
def print_param_dims(params):
    for a in params:
        if len(a) == 0:
            print("()", end='')
        for b in a:
            print(b.shape, end=',')
        print()


def save_state(state, file_adr, file_name):
    pickle.dump(unpack_optimizer_state(state), open(os.path.join(file_adr, file_name), "wb"))


def load_state(file_adr, file_name):
    return pack_optimizer_state(pickle.load(open(os.path.join(file_adr, file_name), "rb")))


# ~~~~~~~~~~~~ losses ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def BCE_from_logits(logits, targets):
    p = sigmoid(logits)
    loss_array = -jnp.log(jnp.where(p == 0, EPSILON, p)) * targets\
                 - jnp.log(jnp.where(1-p == 0, EPSILON, 1-p)) * (1 - targets)
    return jnp.mean(loss_array)


def BCE(predictions, targets):
    loss_array = -jnp.log(jnp.where(predictions == 0, EPSILON, predictions)) * targets\
                 - jnp.log(jnp.where(1-predictions == 0, EPSILON, 1-predictions)) * (1 - targets)
    return jnp.mean(loss_array)


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


# -----------------------------------   Network Models   ------------------------------------
def conv_generator_mnist():
    model = stax.serial(
        Dense(1024 * 7 * 7),
        Reshape((7, 7, 1024)),
        ConvTranspose(out_chan=512, filter_shape=(5, 5), strides=(1, 1),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        BatchNorm(), Relu,
        ConvTranspose(out_chan=256, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        BatchNorm(), Relu,
        ConvTranspose(out_chan=128, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        BatchNorm(), Relu,
        ConvTranspose(out_chan=1, filter_shape=(5, 5), strides=(1, 1),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        Tanh,
    )
    return model


def conv_generator_cifar10():
    model = stax.serial(
        Dense(1024 * 2 * 2),
        Reshape((2, 2, 1024)),
        ConvTranspose(out_chan=512, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        BatchNorm(), Relu,
        ConvTranspose(out_chan=256, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        BatchNorm(), Relu,
        ConvTranspose(out_chan=128, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        BatchNorm(), Relu,
        ConvTranspose(out_chan=3, filter_shape=(5, 5), strides=(2, 2),
                      padding='SAME', W_init=normal(2e-2), b_init=normal(2e-2)),
        Tanh,
    )
    return model


def conv_discriminator():
    model = stax.serial(
        Conv(out_chan=64, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=normal(2e-2), b_init=normal(1e-6)),
        LeakyRelu(negative_slope=0.2),
        Conv(out_chan=128, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=normal(2e-2), b_init=normal(1e-6)),
        BatchNorm(), LeakyRelu(negative_slope=0.2),
        Conv(out_chan=256, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=normal(2e-2), b_init=normal(1e-6)),
        BatchNorm(), LeakyRelu(negative_slope=0.2),
        Conv(out_chan=512, filter_shape=(5, 5), strides=(2, 2),
             padding='SAME', W_init=normal(2e-2), b_init=normal(1e-6)),
        BatchNorm(), LeakyRelu(negative_slope=0.2), Flatten,
        Dense(1), Sigmoid
    )
    return model


def mlp_discriminator():
    model = stax.serial(
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(1), Sigmoid
    )
    return model


def mlp_generator_2d():
    model = stax.serial(
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
        Dense(out_dim=256), Relu,
        # BatchNorm(axis=(1,)),
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

    @staticmethod
    def save_gan_to_file(gan, d_state, g_state, save_adr):
        params = {'d_creator': gan.d_creator,
                  'g_creator': gan.g_creator,
                  'd_opt_creator': gan.d_opt_creator,
                  'g_opt_creator': gan.g_opt_creator,
                  'loss_function': gan.loss_function,
                  'batch_size': gan.batch_size,
                  'd_input_shape': gan.d_input_shape,
                  'g_input_shape': gan.g_input_shape,
                  'd_output_shape': gan.d_output_shape,
                  'g_output_shape': gan.g_output_shape,
                  'g_state': unpack_optimizer_state(g_state),
                  'd_state': unpack_optimizer_state(d_state)
                  }
        with open(save_adr, 'wb') as f:
            pickle.dump(params, f)

    @staticmethod
    def load_gan_from_file(load_adr):
        with open(load_adr, 'rb') as f:
            params = pickle.load(f)
        if params['d_state'] is not None:
            params['d_state'] = pack_optimizer_state(params['d_state'])
        if params['g_state'] is not None:
            params['g_state'] = pack_optimizer_state(params['g_state'])
        gan = GAN(params['d_creator'], params['g_creator'], params['d_opt_creator'], params['g_opt_creator'],
                  params['loss_function'])
        gan.d_output_shape = params['d_output_shape']
        gan.g_output_shape = params['g_output_shape']
        gan.d_input_shape = params['d_input_shape']
        gan.g_input_shape = params['g_input_shape']
        gan.batch_size = params['batch_size']
        return gan, params['d_state'], params['g_state']

    def __init__(self, d_creator, g_creator, d_opt_creator, g_opt_creator, loss_function):
        """

        :param d_creator: (callable) with no input returns discriminator stax model: (init_func, apply_func)
        :param g_creator: (callable) with no input returns generator stax model: (init_func, apply_func)
        :param d_opt_creator: (callable) with no input returns discriminator optimizer: (init, update, get_params)
        :param g_opt_creator: (callable) with no input returns generator optimizer: (init, update, get_params)
        :param loss_function: (function) to calculate loss from discriminator outputs:
                      (discriminator-outputs, real-labels)-> loss
        """
        d_init, d_apply = d_creator()
        g_init, g_apply = g_creator()
        (d_opt_init, d_opt_update, d_opt_get_params) = d_opt_creator()
        (g_opt_init, g_opt_update, g_opt_get_params) = g_opt_creator()

        # self.creators = {'d_creator': d_creator,
        #                  'g_creator': g_creator,
        #                  'd_opt_creator': d_opt_creator,
        #                  'g_opt_creator': g_opt_creator
        #                  }
        self.d_creator = d_creator
        self.g_creator = g_creator
        self.d_opt_creator = d_opt_creator
        self.g_opt_creator = g_opt_creator
        self.d = {'init': d_init, 'apply': d_apply}
        self.g = {'init': g_init, 'apply': g_apply}
        self.d_opt = {'init': d_opt_init, 'update': d_opt_update, 'get_params': d_opt_get_params}
        self.g_opt = {'init': g_opt_init, 'update': g_opt_update, 'get_params': g_opt_get_params}
        self.loss_function = loss_function
        self.d_output_shape = None
        self.g_output_shape = None
        self.d_input_shape = None
        self.g_input_shape = None
        self.batch_size = None

    def init(self, prng_d, prng_g, d_input_shape, g_input_shape, batch_size):
        """

        :param prng_d: (jax.PRNGKey) for discriminator initialization
        :param prng_g: (jax.PRNGKey) for generator initialization
        :param d_input_shape: (tuple) shape of the discriminator input excluding batch size
        :param g_input_shape: (tuple) shape of the generator input excluding batch size
        :param batch_size: (int) used for initialization and training
        :return: discriminator and generator states (needed for train_step and generate_samples)
        """
        self.g_input_shape = g_input_shape
        self.d_input_shape = d_input_shape
        self.d_output_shape, d_params = self.d['init'](prng_d, (batch_size, *d_input_shape))
        self.g_output_shape, g_params = self.g['init'](prng_g, (batch_size, *g_input_shape))
        self.batch_size = batch_size
        d_state = self.d_opt['init'](d_params)
        g_state = self.g_opt['init'](g_params)
        return d_state, g_state

    @partial(jit, static_argnums=(0,))
    def _d_loss(self, d_params, g_params, z, real_samples):
        fake_ims = self.g['apply'](g_params, z)

        fake_predictions = self.d['apply'](d_params, fake_ims)
        real_predictions = self.d['apply'](d_params, real_samples)
        fake_loss = self.loss_function(fake_predictions, jnp.zeros(len(fake_predictions)))
        real_loss = self.loss_function(real_predictions, jnp.ones(len(real_predictions)))

        return fake_loss + real_loss

    @partial(jit, static_argnums=(0, 4))
    def _g_loss(self, g_params, d_params, z, k):
        fake_ims = self.g['apply'](g_params, z)

        fake_predictions = self.d['apply'](d_params, fake_ims)
        fake_predictions = sort(fake_predictions, 0)
        fake_predictions = jnp.flip(fake_predictions, 0)
        fake_predictions = fake_predictions[:k]

        loss = self.loss_function(fake_predictions, jnp.ones(len(fake_predictions)))

        return loss

    @partial(jit, static_argnums=(0, 6))
    def train_step(self, i, prng_key, d_state, g_state, real_samples, k):
        """
        !: call init function before train_step

        :param i: (int) step number
        :param prng_key: (jax.random.PRNGKey) used to create random samples from the generator
        :param d_state: previous discriminator state
        :param g_state: previous generator state
        :param real_samples: (np/jnp array) samples form the training set
        :param k: (int) to choose top k for training generator, if None all elements are chosen
        :return: updated discriminator and generator states and discriminator and generator loss values
        """
        k = k or self.batch_size
        prng1, prng2 = random.split(prng_key, 2)
        d_params = self.d_opt['get_params'](d_state)
        g_params = self.g_opt['get_params'](g_state)

        z = random.normal(prng1, (self.batch_size, *self.g_input_shape))
        d_loss_value, d_grads = value_and_grad(self._d_loss)(d_params, g_params, z, real_samples)
        d_state = self.d_opt['update'](i, d_grads, d_state)

        z = random.normal(prng2, (self.batch_size, *self.g_input_shape))
        g_loss_value, g_grads = value_and_grad(self._g_loss)(g_params, d_params, z, k)
        g_state = self.g_opt['update'](i, g_grads, g_state)

        return d_state, g_state, d_loss_value, g_loss_value

    @partial(jit, static_argnums=(0,))
    def generate_samples(self, z, g_state):
        """

        :param z: (np/jnp array) shape: (n, generator_input_dims)
        :param g_state: generator state
        :return: (jnp array) shape: (n, generator_output_dims) n generated samples
        """
        fakes = self.g['apply'](self.g_opt['get_params'](g_state), z)
        return fakes

    @partial(jit, static_argnums=(0,))
    def rate_samples(self, samples, d_state):
        """

        :return: (jnp array) shape: (n, 1) discriminator ratings for the samples
        """
        rates = self.d['apply'](self.d_opt['get_params'](d_state), samples)
        return rates
