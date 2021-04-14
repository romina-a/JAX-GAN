import jax
import jax.numpy as jnp
from jax.experimental.optimizers import adam
from jax import value_and_grad, jit
import matplotlib.pyplot as plt
import argparse
import time
from functools import partial

from Models import conv_generator_mnist, conv_discriminator
from dataset_loader import get_NumpyLoader_mnist as mnist_dataset


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


d_init, d_apply = conv_discriminator()
g_init, g_apply = conv_generator_mnist()

(d_opt_init_fun, d_opt_update_fun, d_opt_get_params) = adam(d_lr_default, d_momentum_default, d_momentum2_default)
(g_opt_init_fun, g_opt_update_fun, g_opt_get_params) = adam(g_lr_default, g_momentum_default, g_momentum2_default)
d_opt = {'init': d_opt_init_fun, 'update': d_opt_update_fun, 'get_params': d_opt_get_params}
g_opt = {'init': g_opt_init_fun, 'update': g_opt_update_fun, 'get_params': g_opt_get_params}


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


def train(batch_size, num_iter, digit):
    prng_key = jax.random.PRNGKey(0)
    prng1, prng2, prng = jax.random.split(prng_key, 3)
    d_output_shape, d_params = d_init(prng1, (batch_size, 32, 32, 1))
    g_output_shape, g_params = g_init(prng2, (batch_size, 100))
    d_state = d_opt['init'](d_params)
    g_state = g_opt['init'](g_params)

    d_losses = []
    g_losses = []

    start_time = time.time()
    prev_time = time.time()
    i = 0
    dataset = mnist_dataset(batch_size, digit=digit)
    while i < num_iter:
        epoch_start_time = time.time()
        for real_ims, _ in dataset:
            if i >= num_iter:
                break
            if i % 100 == 0:
                print(f"{i}/{num_iter} took {time.time()-prev_time}")
                prev_time = time.time()

                z = jax.random.normal(jax.random.PRNGKey(0), (1, 100))
                fake = g_apply(g_opt["get_params"](g_state), z)
                fake = fake.reshape((28, 28))
                plt.imshow((fake + 1.0) / 2.0, cmap='gray')
                plt.show()

            prng, prng_to_use = jax.random.split(prng, 2)
            d_state, g_state, d_loss_value, g_loss_value = train_step(i, prng_to_use, d_state, g_state, real_ims, batch_size)
            d_losses.append(d_loss_value)
            g_losses.append(g_loss_value)
            i = i + 1
        print(f'epoch finished in {time.time()-epoch_start_time}')
    print(f'finished, took{time.time()-start_time}')

    return d_losses, g_losses, d_state, g_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=False, default=batch_size_default, type=int,
                        help="training batch size")
    parser.add_argument("--num_iter", required=False, default=num_iter_default, type=int,
                        help="number of iterations")
    parser.add_argument("--digit", required=False, default=digit_default, type=int,
                        help="digit")
    args = vars(parser.parse_args())
    train(
        batch_size=args['batch_size'],
        num_iter=args['num_iter'],
        digit=args['digit']
    )




