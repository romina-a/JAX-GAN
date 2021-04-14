from jax.nn.initializers import glorot_normal, normal, ones, zeros
from jax.nn import leaky_relu
from jax.experimental import stax
from jax.experimental.stax import (BatchNorm, Conv, ConvTranspose, Dense,
                                   Tanh, Relu, Flatten)
import jax.numpy as jnp


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
