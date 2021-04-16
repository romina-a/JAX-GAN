import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import value_and_grad, jit
import matplotlib.pyplot as plt
import argparse
import time
from functools import partial

from Models import mlp_generator_2d, mlp_discriminator
from Models import BCE_from_logits
from ToyData import GaussianMixture
from visualizing_distributions import plot_samples_scatter
from visualizing_distributions import visualize_sample_heatmap


# ~~~~~~~~~~~ Stax GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset_default = 'gaussian_mixture'
num_components_default = 25
prior_dim_default = 2

d_lr_default = 0.0001
d_momentum_default = 0.9
d_momentum2_default = 0.99
g_lr_default = 0.0001
g_momentum_default = 0.9
g_momentum2_default = 0.99
loss_function_default = BCE_from_logits

batch_size_default = 256
batch_size_min_default = 192
decay_rate_default = 0.99

num_iter_default = 100000


def get_dataset(prng_key, batch_size, num_components):
    return GaussianMixture(prng_key, batch_size, num_modes=num_components, variance=0.05)


d_init, d_apply = mlp_discriminator()
g_init, g_apply = mlp_generator_2d()
(d_opt_init_fun, d_opt_update_fun, d_opt_get_params) = adam(d_lr_default, d_momentum_default, d_momentum2_default)
(g_opt_init_fun, g_opt_update_fun, g_opt_get_params) = adam(g_lr_default, g_momentum_default, g_momentum2_default)
d_opt = {'init': d_opt_init_fun, 'update': d_opt_update_fun, 'get_params': d_opt_get_params}
g_opt = {'init': g_opt_init_fun, 'update': g_opt_update_fun, 'get_params': g_opt_get_params}


def initialize(d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
               g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,):

    (d_opt_init_fun, d_opt_update_fun, d_opt_get_params) = adam(d_lr, d_momentum, d_momentum2)
    (g_opt_init_fun, g_opt_update_fun, g_opt_get_params) = adam(g_lr, g_momentum, g_momentum2)
    globals()['d_opt'] = {'init': d_opt_init_fun, 'update': d_opt_update_fun, 'get_params': d_opt_get_params}
    globals()['g_opt'] = {'init': g_opt_init_fun, 'update': g_opt_update_fun, 'get_params': g_opt_get_params}


def plot_sample_output(g_state, n=1):
    z = jax.random.normal(jax.random.PRNGKey(0), (n, prior_dim_default))
    fakes = g_apply(g_opt["get_params"](g_state), z)
    plot_samples_scatter(fakes)


@partial(jit, static_argnums=(3,))
def d_loss(d_params, g_params, prng_key, batch_size, real_ims):
    z = jax.random.normal(prng_key, (batch_size, prior_dim_default))
    fake_ims = g_apply(g_params, z)

    fake_predictions = d_apply(d_params, fake_ims)
    real_predictions = d_apply(d_params, real_ims)

    fake_loss = loss_function_default(fake_predictions, jnp.zeros(batch_size))
    real_loss = loss_function_default(real_predictions, jnp.ones(batch_size))

    return fake_loss + real_loss


@partial(jit, static_argnums=(3,))
def g_loss(g_params, d_params, prng_key, batch_size):
    z = jax.random.normal(prng_key, (batch_size, prior_dim_default))
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


def train(batch_size, num_iter, num_components=num_components_default,
          d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
          g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,
          ):
    initialize(
        d_lr=d_lr, d_momentum=d_momentum, d_momentum2=d_momentum2,
        g_lr=g_lr, g_momentum=g_momentum, g_momentum2=g_momentum2,
    )

    prng_key = jax.random.PRNGKey(0)
    prng_data, prng1, prng2, prng = jax.random.split(prng_key, 4)
    real_data = get_dataset(prng_data, batch_size, num_components)
    d_output_shape, d_params = d_init(prng1, (batch_size, 2))
    g_output_shape, g_params = g_init(prng2, (batch_size, prior_dim_default))
    d_state = d_opt['init'](d_params)
    g_state = g_opt['init'](g_params)

    d_losses = []
    g_losses = []

    start_time = time.time()
    prev_time = time.time()
    i = 0

    while i < num_iter:
        if i >= num_iter:
            break
        if i % 1000 == 0:
            print(f"{i}/{num_iter} took {time.time() - prev_time}")
            prev_time = time.time()
            plot_sample_output(g_state, 1000)

        real_ims = real_data.get_next_batch()

        prng, prng_to_use = jax.random.split(prng, 2)
        d_state, g_state, d_loss_value, g_loss_value = train_step(i, prng_to_use, d_state, g_state, real_ims,
                                                                  batch_size)
        d_losses.append(d_loss_value)
        g_losses.append(g_loss_value)
        i = i + 1

    print(f'finished, took{time.time() - start_time}')

    return d_losses, g_losses, d_state, g_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=batch_size_default, type=int,
                        help="training batch size")
    parser.add_argument("--num_iter", required=False, default=num_iter_default, type=int,
                        help="number of iterations")
    parser.add_argument("--dataset", required=False, default=dataset_default, type=str,
                        choices={'gaussian_mixture'}, help="the dataset for training")
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
        d_lr=args['d_lr'], d_momentum=args['d_momentum'], d_momentum2=args['d_momentum2'],
        g_lr=args['g_lr'], g_momentum=args['g_momentum'], g_momentum2=args['g_momentum2'],
    )
