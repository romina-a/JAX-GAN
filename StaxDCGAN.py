import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import value_and_grad, jit
import matplotlib.pyplot as plt
import argparse
import time
from functools import partial

from Models import conv_generator_mnist, conv_generator_cifar10, conv_discriminator
from Models import BCE_from_logits
from dataset_loader import get_NumpyLoader_mnist as mnist_dataset
from dataset_loader import get_NumpyLoader_cifar10 as cifar10_dataset

# ~~~~~~~~~~~ Stax GAN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset_default = 'mnist'

batch_size_default = 128
num_iter_default = 2000
digit_default = 0

d_lr_default = 0.0002
d_momentum_default = 0.5
d_momentum2_default = 0.5
g_lr_default = 0.0002
g_momentum_default = 0.5
g_momentum2_default = 0.5
loss_function_default = BCE_from_logits

dataset_loaders = {'mnist': mnist_dataset, 'cifar10': cifar10_dataset}
generators = {'mnist': conv_generator_mnist, 'cifar10': conv_generator_cifar10}


class GAN:
    def __init__(self, d, g, d_opt, g_opt, loss_function):
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
        self.g_input_shape = g_input_shape
        self.d_input_shape = d_input_shape
        self.d_output_shape, d_params = self.d['init'](prng_d, (1, *d_input_shape))
        self.g_output_shape, g_params = self.g['init'](prng_g, (1, *g_input_shape))
        d_state = self.d_opt['init'](d_params)
        g_state = self.g_opt['init'](g_params)
        return d_state, g_state

    @partial(jit, static_argnums=(0, 4,))
    def d_loss(self, d_params, g_params, prng_key, batch_size, real_ims):
        z = jax.random.normal(prng_key, (batch_size, *self.g_input_shape))
        fake_ims = self.g['apply'](g_params, z)

        fake_predictions = self.d['apply'](d_params, fake_ims)
        real_predictions = self.d['apply'](d_params, real_ims)

        fake_loss = self.loss_function(fake_predictions, jnp.zeros(batch_size))
        real_loss = self.loss_function(real_predictions, jnp.ones(batch_size))

        return fake_loss + real_loss

    @partial(jit, static_argnums=(0, 4,))
    def g_loss(self, g_params, d_params, prng_key, batch_size):
        z = jax.random.normal(prng_key, (batch_size, *self.g_input_shape))
        fake_ims = self.g['apply'](g_params, z)

        fake_predictions = self.d['apply'](d_params, fake_ims)

        loss = self.loss_function(fake_predictions, jnp.ones(batch_size))

        return loss

    @partial(jit, static_argnums=(0, 6,))
    def train_step(self, i, prng_key, d_state, g_state, real_ims, batch_size):
        prng1, prng2 = jax.random.split(prng_key, 2)
        d_params = self.d_opt['get_params'](d_state)
        g_params = self.g_opt['get_params'](g_state)

        d_loss_value, d_grads = value_and_grad(self.d_loss)(d_params, g_params, prng1, batch_size, real_ims)
        d_state = self.d_opt['update'](i, d_grads, d_state)

        g_loss_value, g_grads = value_and_grad(self.g_loss)(g_params, d_params, prng2, batch_size)
        g_state = self.g_opt['update'](i, g_grads, g_state)

        return d_state, g_state, d_loss_value, g_loss_value

    @partial(jit, static_argnums=(0,))
    def get_images(self, z, g_state):
        fakes = self.g['apply'](self.g_opt['get_params'](g_state), z)
        return fakes


def plot_samples(ims):
    for im in ims:
        if im.shape[2] == 1:
            im = im.reshape(im.shape[:2])
        plt.imshow((im + 1.0) / 2.0)
        plt.show()


def create_and_initialize_gan(prng, dataset,
                              d_lr, d_momentum, d_momentum2, g_lr, g_momentum, g_momentum2, loss_function,
                              d_input_shape, g_input_shape):
    temp_d_init, temp_d_apply = conv_discriminator()
    temp_g_init, temp_g_apply = generators[dataset]()
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


def train(batch_size, num_iter, digit, dataset=dataset_default, loss_function=loss_function_default,
          d_lr=d_lr_default, d_momentum=d_momentum_default, d_momentum2=d_momentum2_default,
          g_lr=g_lr_default, g_momentum=g_momentum_default, g_momentum2=g_momentum2_default,
          ):
    prng = jax.random.PRNGKey(0)
    dataset_loader = dataset_loaders[dataset]
    real_data = dataset_loader(batch_size, digit=digit)
    samples, sample_labels = next(iter(real_data))
    im_shape = samples[0].shape

    prng_to_use, prng = jax.random.split(prng, 2)
    gan, d_state, g_state = create_and_initialize_gan(prng_to_use, dataset,
                                                      d_lr, d_momentum, d_momentum2,
                                                      g_lr, g_momentum, g_momentum2,
                                                      loss_function, im_shape, (100,))

    d_losses = []
    g_losses = []

    start_time = time.time()
    prev_time = time.time()
    i = 0

    prng_images, prng = jax.random.split(prng, 2)
    z = jax.random.normal(prng_images, (1, 100))

    while i < num_iter:
        epoch_start_time = time.time()
        for real_ims, _ in real_data:
            if i >= num_iter:
                break
            if i % 100 == 0:
                print(f"{i}/{num_iter} took {time.time() - prev_time}")
                prev_time = time.time()
                plot_samples(gan.get_images(z, g_state))

            prng, prng_to_use = jax.random.split(prng, 2)
            d_state, g_state, d_loss_value, g_loss_value = gan.train_step(i, prng_to_use, d_state, g_state, real_ims,
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
