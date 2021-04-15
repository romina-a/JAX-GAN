import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import value_and_grad, jit
import matplotlib.pyplot as plt
import argparse
import time
from functools import partial

from Models import conv_generator_mnist, conv_generator_cifar10, conv_discriminator
from dataset_loader import get_NumpyLoader_mnist as mnist_dataset
from dataset_loader import get_NumpyLoader_cifar10 as cifar10_dataset


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


# ~~~~~~~~~~~ Stax GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset_default = 'mnist'

d_lr_default = 0.0002
d_momentum_default = 0.5
d_momentum2_default = 0.5
g_lr_default = 0.0002
g_momentum_default = 0.5
g_momentum2_default = 0.5
loss_function_default = BCE_from_logits

batch_size_default = 128
num_iter_default = 2000
digit_default = 0

dataset_loaders = {'mnist': mnist_dataset, 'cifar10': cifar10_dataset}
generators = {'mnist': conv_generator_mnist, 'cifar10': conv_generator_cifar10}


dataset_loader = dataset_loaders[dataset_default]
d_init, d_apply = conv_discriminator()
g_init, g_apply = generators[dataset_default]()
(d_opt_init_fun, d_opt_update_fun, d_opt_get_params) = adam(d_lr_default, d_momentum_default, d_momentum2_default)
(g_opt_init_fun, g_opt_update_fun, g_opt_get_params) = adam(g_lr_default, g_momentum_default, g_momentum2_default)
d_opt = {'init': d_opt_init_fun, 'update': d_opt_update_fun, 'get_params': d_opt_get_params}
g_opt = {'init': g_opt_init_fun, 'update': g_opt_update_fun, 'get_params': g_opt_get_params}


def initialize(dataset,
               d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
               g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,
               ):
    globals()['dataset_loader'] = dataset_loaders[dataset]

    globals()['d_init'], globals()['d_apply'] = conv_discriminator()
    globals()['g_init'], globals()['g_apply'] = generators[dataset]()

    (d_opt_init_fun, d_opt_update_fun, d_opt_get_params) = adam(d_lr, d_momentum, d_momentum2)
    (g_opt_init_fun, g_opt_update_fun, g_opt_get_params) = adam(g_lr, g_momentum, g_momentum2)
    globals()['d_opt'] = {'init': d_opt_init_fun, 'update': d_opt_update_fun, 'get_params': d_opt_get_params}
    globals()['g_opt'] = {'init': g_opt_init_fun, 'update': g_opt_update_fun, 'get_params': g_opt_get_params}


def plot_sample_output(g_state, n=1):
    z = jax.random.normal(jax.random.PRNGKey(0), (n, 100))
    fakes = g_apply(g_opt["get_params"](g_state), z)
    for im in fakes:
        if im.shape[2] == 1:
            im = im.reshape(im.shape[:2])
        plt.imshow((im + 1.0) / 2.0)
        plt.show()


@partial(jit, static_argnums=(3,))
def d_loss(d_params, g_params, prng_key, batch_size, real_ims):
    z = jax.random.normal(prng_key, (batch_size, 100))
    fake_ims = g_apply(g_params, z)

    fake_predictions = d_apply(d_params, fake_ims)
    real_predictions = d_apply(d_params, real_ims)

    fake_loss = loss_function_default(fake_predictions, jnp.zeros(batch_size))
    real_loss = loss_function_default(real_predictions, jnp.ones(batch_size))

    return fake_loss + real_loss


@partial(jit, static_argnums=(3,))
def g_loss(g_params, d_params, prng_key, batch_size):
    z = jax.random.normal(prng_key, (batch_size, 100))
    fake_ims = g_apply(g_params, z)

    fake_predictions = d_apply(d_params, fake_ims)

    loss = loss_function_default(fake_predictions, jnp.ones(batch_size))

    return loss


@partial(jit, static_argnums=(5,))
def train_step(i, prng_key, d_state, g_state, real_ims, batch_size):
    prng1, prng2 = jax.random.split(prng_key, 2)
    d_params = d_opt['get_params'](d_state)
    g_params = g_opt['get_params'](g_state)

    d_loss_value, d_grads = value_and_grad(d_loss)(d_params, g_params, prng1, batch_size, real_ims)
    d_state = d_opt['update'](i, d_grads, d_state)

    g_loss_value, g_grads = value_and_grad(g_loss)(g_params, d_params, prng2, batch_size)
    g_state = g_opt['update'](i, g_grads, g_state)

    return d_state, g_state, d_loss_value, g_loss_value


def train(batch_size, num_iter, digit,
          dataset=dataset_default,
          d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
          g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,
          ):
    initialize(
        dataset=dataset,
        d_lr=d_lr, d_momentum=d_momentum, d_momentum2=d_momentum2,
        g_lr=g_lr, g_momentum=g_momentum, g_momentum2=g_momentum2,
    )
    real_data = dataset_loader(batch_size, digit=digit)
    samples, sample_labels = next(iter(real_data))
    im_shape = samples[0].shape

    prng_key = jax.random.PRNGKey(0)
    prng1, prng2, prng = jax.random.split(prng_key, 3)
    d_output_shape, d_params = d_init(prng1, (batch_size, *im_shape))
    g_output_shape, g_params = g_init(prng2, (batch_size, 100))
    d_state = d_opt['init'](d_params)
    g_state = g_opt['init'](g_params)

    d_losses = []
    g_losses = []

    start_time = time.time()
    prev_time = time.time()
    i = 0

    while i < num_iter:
        epoch_start_time = time.time()
        for real_ims, _ in real_data:
            if i >= num_iter:
                break
            if i % 100 == 0:
                print(f"{i}/{num_iter} took {time.time() - prev_time}")
                prev_time = time.time()
                plot_sample_output(g_state)

            prng, prng_to_use = jax.random.split(prng, 2)
            d_state, g_state, d_loss_value, g_loss_value = train_step(i, prng_to_use, d_state, g_state, real_ims,
                                                                      batch_size)
            d_losses.append(d_loss_value)
            g_losses.append(g_loss_value)
            i = i + 1
        print(f'epoch finished in {time.time() - epoch_start_time}')
    print(f'finished, took{time.time() - start_time}')

    return d_losses, g_losses, d_state, g_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=batch_size_default, type=int,
                        help="training batch size")
    parser.add_argument("--num_iter", required=False, default=num_iter_default, type=int,
                        help="number of iterations")
    parser.add_argument("--digit", required=False, default=digit_default, type=int,
                        help="digit")
    parser.add_argument("--dataset", required=False, default=dataset_default, type=str,
                        choices={'mnist','cifar10'}, help="the dataset for training")
    parser.add_argument("--d_lr", required=False, default=d_lr_default, type=jnp.float32,
                        help="discriminator learning rate")
    parser.add_argument("--d_momentum", required=False, default=d_momentum_default, type=jnp.float32,
                        help="discriminator momentum")
    parser.add_argument("--d_momentum2", required=False, default=d_momentum2_default, type=jnp.float32,
                        help="discriminator second momentum")
    parser.add_argument("--g_lr", required=False, default=g_lr_default, type=jnp.float32,
                        help="generator learning rate")
    parser.add_argument("--g_momentum", required=False, default=g_momentum_default, type=jnp.float32,
                        help="generator momentum")
    parser.add_argument("--g_momentum2", required=False, default=g_momentum2_default, type=jnp.float32,
                        help="generator second momentum")

    args = vars(parser.parse_args())
    train(
        batch_size=args['batch_size'],
        num_iter=args['num_iter'],
        digit=args['digit'],
        dataset=args['dataset'],
        d_lr=args['d_lr'], d_momentum=args['d_momentum'], d_momentum2=args['d_momentum2'],
        g_lr=args['g_lr'], g_momentum=args['g_momentum'], g_momentum2=args['g_momentum2'],
    )
