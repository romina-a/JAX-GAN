import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

import argparse
import time

from Models import mlp_generator_2d, mlp_discriminator, GAN
from Models import BCE_from_logits
from ToyData import GaussianMixture
from visualizing_distributions import plot_samples_scatter

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
    return GaussianMixture(prng_key, batch_size, num_modes=num_components, variance=0.005)


def create_and_initialize_gan(prng, d_lr, d_momentum, d_momentum2, g_lr, g_momentum, g_momentum2, loss_function,
                              d_input_shape, g_input_shape):
    temp_d_init, temp_d_apply = mlp_discriminator()
    temp_g_init, temp_g_apply = mlp_generator_2d()
    temp_d = {'init': temp_d_init, 'apply': temp_d_apply}
    temp_g = {'init': temp_g_init, 'apply': temp_g_apply}
    (temp_d_opt_init_fun, temp_d_opt_update_fun, temp_d_opt_get_params) = adam(d_lr, d_momentum, d_momentum2)
    (temp_g_opt_init_fun, temp_g_opt_update_fun, temp_g_opt_get_params) = adam(g_lr, g_momentum, g_momentum2)
    temp_d_opt = {'init': temp_d_opt_init_fun, 'update': temp_d_opt_update_fun, 'get_params': temp_d_opt_get_params}
    temp_g_opt = {'init': temp_g_opt_init_fun, 'update': temp_g_opt_update_fun, 'get_params': temp_g_opt_get_params}

    gan = GAN(temp_d, temp_g, temp_d_opt, temp_g_opt, loss_function)

    prng1, prng2 = jax.random.split(prng, 2)
    d_state, g_state = gan.init(prng1, prng2, d_input_shape, g_input_shape)
    return gan, d_state, g_state


def train(batch_size, num_iter, num_components, dataset=dataset_default,
          loss_function=loss_function_default,
          prior_dim=prior_dim_default,
          d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
          g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,
          ):
    prng = jax.random.PRNGKey(0)
    prng_to_use, prng = jax.random.split(prng)
    dataset_loader = get_dataset(prng_to_use, batch_size, num_components)
    im_shape = (2,)

    prng_to_use, prng = jax.random.split(prng, 2)
    gan, d_state, g_state = create_and_initialize_gan(prng_to_use,
                                                      d_lr, d_momentum, d_momentum2,
                                                      g_lr, g_momentum, g_momentum2,
                                                      loss_function, im_shape, (prior_dim,))

    d_losses = []
    g_losses = []

    start_time = time.time()
    prev_time = time.time()
    i = 0

    prng_images, prng = jax.random.split(prng, 2)
    z = jax.random.normal(prng_images, (batch_size, prior_dim_default))

    while i < num_iter:
        real_ims = dataset_loader.get_next_batch()
        if i >= num_iter:
            break
        if i % 1000 == 0:
            print(f"{i}/{num_iter} took {time.time() - prev_time}")
            prev_time = time.time()
            plot_samples_scatter(gan.generate_samples(z, g_state))
            plot_samples_scatter(real_ims)

        prng, prng_to_use = jax.random.split(prng, 2)
        d_state, g_state, d_loss_value, g_loss_value = gan.train_step(i, prng_to_use, d_state, g_state, real_ims,
                                                                      batch_size)
        d_losses.append(d_loss_value)
        g_losses.append(g_loss_value)
        i = i + 1
    print(f'finished, took{time.time() - start_time}')

    return d_losses, g_losses, d_state, g_state, gan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=batch_size_default, type=int,
                        help="training batch size")
    parser.add_argument("--num_iter", required=False, default=num_iter_default, type=int,
                        help="number of iterations")
    parser.add_argument("--dataset", required=False, default=dataset_default, type=str,
                        choices={'gaussian_mixture'}, help="the dataset for training")
    parser.add_argument("--num_components", required=False, default=num_components_default, type=int,
                        help="number of gaussian components")
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
        num_components=args['num_components'],
        dataset=args['dataset'],
        d_lr=args['d_lr'], d_momentum=args['d_momentum'], d_momentum2=args['d_momentum2'],
        g_lr=args['g_lr'], g_momentum=args['g_momentum'], g_momentum2=args['g_momentum2'],
    )