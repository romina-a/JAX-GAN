import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam

import matplotlib.pyplot as plt
import argparse
import time
from functools import partial

from Models import conv_generator_mnist, conv_generator_cifar10, conv_discriminator, GAN
from Models import BCE_from_logits
from dataset_loader import get_NumpyLoader_mnist as mnist_dataset
from dataset_loader import get_NumpyLoader_cifar10 as cifar10_dataset

# ~~~~~~~~~~~ Stax GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset_default = 'mnist'
digit_default = 0

d_lr_default = 0.0002
d_momentum_default = 0.5
d_momentum2_default = 0.5
g_lr_default = 0.0002
g_momentum_default = 0.5
g_momentum2_default = 0.5
loss_function_default = BCE_from_logits

batch_size_default = 128
batch_size_min_default = 64
decay_rate_default = 0.99

num_iter_default = 10000

dataset_loaders = {'mnist': mnist_dataset, 'cifar10': cifar10_dataset}
generators = {'mnist': conv_generator_mnist, 'cifar10': conv_generator_cifar10}


def plot_samples(ims, grid_dim=1, save_adr=None):
    dim1 = grid_dim
    dim2 = len(ims)//dim1

    for i in range(len(ims)):
        im = ims[i]
        if im.shape[2] == 1:
            im = im.reshape(im.shape[:2])
        plt.subplot(dim1, dim2, i + 1)
        plt.imshow((im + 1.0) / 2.0)
        plt.axis('off')
    if save_adr is not None:
        plt.savefig(save_adr)
    plt.show()


def create_and_initialize_gan(prng, d_lr, d_momentum, d_momentum2, g_lr, g_momentum, g_momentum2, loss_function,
                              d_input_shape, g_input_shape, batch_size, dataset):
    d_creator = conv_discriminator
    g_creator = generators[dataset]
    d_opt_creator = partial(adam, d_lr, d_momentum, d_momentum2)
    g_opt_creator = partial(adam, g_lr, g_momentum, g_momentum2)

    gan = GAN(d_creator, g_creator, d_opt_creator, g_opt_creator, loss_function)

    prng1, prng2 = jax.random.split(prng, 2)
    d_state, g_state = gan.init(prng1, prng2, d_input_shape, g_input_shape, batch_size)
    return gan, d_state, g_state


def train(batch_size=batch_size_default, num_iter=num_iter_default, digit=digit_default, dataset=dataset_default, loss_function=loss_function_default,
          d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
          g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,
          top_k=1):

    dataset_loader = dataset_loaders[dataset]
    real_data = dataset_loader(batch_size, digit=digit)
    samples, sample_labels = next(iter(real_data))
    im_shape = samples[0].shape


    prng = jax.random.PRNGKey(0)
    prng_to_use, prng = jax.random.split(prng, 2)
    gan, d_state, g_state = create_and_initialize_gan(prng_to_use,
                                                      d_lr, d_momentum, d_momentum2,
                                                      g_lr, g_momentum, g_momentum2,
                                                      loss_function, im_shape, (100,), batch_size, dataset)

    d_losses = []
    g_losses = []

    start_time = time.time()
    prev_time = time.time()
    i = 0

    prng_images, prng = jax.random.split(prng, 2)
    z = jax.random.normal(prng_images, (9, 100))

    k = batch_size
    while i < num_iter:
        epoch_start_time = time.time()
        for real_ims, _ in real_data:
            if i >= num_iter:
                break
            if i % 1 == 0:
                print(f"{i}/{num_iter} took {time.time() - prev_time}")
                prev_time = time.time()
                plot_samples(gan.generate_samples(z, g_state), 3)

            prng, prng_to_use = jax.random.split(prng, 2)
            d_state, g_state, d_loss_value, g_loss_value = gan.train_step(i, prng_to_use, d_state, g_state, real_ims, k)
            d_losses.append(d_loss_value)
            g_losses.append(g_loss_value)
            i = i + 1
        print(f'epoch finished in {time.time() - epoch_start_time}')
        if top_k == 1:
            k = int(k * decay_rate_default)
            k = max(batch_size_min_default, k)
            print(f"iter:{i}/{num_iter}, updated k: {k}")
    print(f'finished, took{time.time() - start_time}')

    return d_losses, g_losses, d_state, g_state, gan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=batch_size_default, type=int,
                        help="training batch size")
    parser.add_argument("--num_iter", required=False, default=num_iter_default, type=int,
                        help="number of iterations")
    parser.add_argument("--digit", required=False, default=digit_default, type=int,
                        help="digit")
    parser.add_argument("--dataset", required=False, default=dataset_default, type=str,
                        choices={'mnist', 'cifar10'}, help="the dataset for training")
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
