import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

import argparse
import time

from Models import mlp_generator_2d, mlp_discriminator, GAN
from Models import BCE_from_logits
from ToyData import get_gaussian_mixture
from visualizing_distributions import plot_samples_scatter
from functools import partial

# ~~~~~~~~~~~ Stax GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset_default = 'gaussian_mixture'
num_components_default = 25
gaussian_variance_default = 0.0025
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


def create_and_initialize_gan(prng, d_lr, d_momentum, d_momentum2, g_lr, g_momentum, g_momentum2, loss_function,
                              d_input_shape, g_input_shape, batch_size):
    d_creator = mlp_discriminator
    g_creator = mlp_generator_2d
    d_opt_creator = partial(adam, d_lr, d_momentum, d_momentum2)
    g_opt_creator = partial(adam, g_lr, g_momentum, g_momentum2)

    gan = GAN(d_creator, g_creator, d_opt_creator, g_opt_creator, loss_function)

    prng1, prng2 = jax.random.split(prng, 2)
    d_state, g_state = gan.init(prng1, prng2, d_input_shape, g_input_shape, batch_size)
    return gan, d_state, g_state


def train(num_components, batch_size=batch_size_default, num_iter=num_iter_default,
          dataset=dataset_default, loss_function=loss_function_default,
          prior_dim=prior_dim_default, d_lr=d_lr_default, d_momentum=d_momentum_default,
          d_momentum2=d_momentum2_default, g_lr=g_lr_default, g_momentum=g_momentum_default,
          g_momentum2=g_momentum2_default, top_k=1):
    prng = jax.random.PRNGKey(10)

    im_shape = (2,)
    prng_to_use, prng = jax.random.split(prng, 2)
    gan, d_state, g_state = create_and_initialize_gan(prng_to_use,
                                                      d_lr, d_momentum, d_momentum2,
                                                      g_lr, g_momentum, g_momentum2,
                                                      loss_function, im_shape, (prior_dim,), batch_size)

    data = get_gaussian_mixture(batch_size, num_iter, num_components, gaussian_variance_default)

    d_losses = []
    g_losses = []

    prng_images, prng = jax.random.split(prng, 2)
    z = jax.random.normal(prng_images, (10000, prior_dim_default))

    start_time = time.time()
    prev_time = time.time()
    k = batch_size
    for i, real_ims in enumerate(data):
        if i % 1000 == 0:
            print(f"{i}/{num_iter} took {time.time() - prev_time}")
            prev_time = time.time()
            fakes = gan.generate_samples(z, g_state)
            plot_samples_scatter(fakes, real_ims,
                                 save_adr=f"./output_ims/experiment2-seed=10/{num_components}-{top_k}-{i//1000}.jpg",
                                 samples_ratings=gan.rate_samples(fakes, d_state))
            # plot_samples_scatter(gan.generate_samples(z, g_state))
        if top_k == 1 and i % 2000 == 1999:
            k = int(k * decay_rate_default)
            k = max(batch_size_min_default, k)
            print(f"iter:{i}/{num_iter}, updated k: {k}")

        prng, prng_to_use = jax.random.split(prng, 2)
        d_state, g_state, d_loss_value, g_loss_value = gan.train_step(i, prng_to_use, d_state, g_state, real_ims, k)
        d_losses.append(d_loss_value)
        g_losses.append(g_loss_value)
    print(f'finished, took{time.time() - start_time}')

    return d_losses, g_losses, d_state, g_state, gan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", required=False, default=1, type=int,
                        choices={1, 0}, help="1: use top-k, 0: no top-k")
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
    d_losses, g_losses, d_state, g_state, gan = train(num_components=args['num_components'],
                                                      batch_size=args['batch_size'], num_iter=args['num_iter'],
                                                      dataset=args['dataset'], d_lr=args['d_lr'],
                                                      d_momentum=args['d_momentum'], d_momentum2=args['d_momentum2'],
                                                      g_lr=args['g_lr'], g_momentum=args['g_momentum'],
                                                      g_momentum2=args['g_momentum2'], top_k=args['top_k'])
    gan.save_gan_to_file(gan, d_state, g_state, "./Models/gan-25-wo-no-norm.pkl")
