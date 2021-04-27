from Models import GAN
import ToyData
from jax import random, value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ToyGAN_eval_vis
import ToyData


def gan_train_step(i, d_z, g_z, real_samples, gan, d_state, g_state):
    """

    :param i: train step
    :param d_z: latent variables for discriminator step
    :param g_z: latent variables for generator step
    :param real_samples: real samples for discriminator step
    :param gan:
    :param d_state: current discriminator optimizer state
    :param g_state: current generator optimizer state
    :return: updated discriminator and generator optimizer states and their train step loss values
    """
    k = len(g_z)
    d_params = gan.d_opt['get_params'](d_state)
    g_params = gan.g_opt['get_params'](g_state)

    d_loss_value, d_grads = value_and_grad(gan._d_loss)(d_params, g_params, d_z, real_samples)
    d_state = gan.d_opt['update'](i, d_grads, d_state)

    g_loss_value, g_grads = value_and_grad(gan._g_loss)(g_params, d_params, g_z, k)
    g_state = gan.g_opt['update'](i, g_grads, g_state)

    return d_state, g_state, d_loss_value, g_loss_value


def perform_update(gan_path, num_modes, var, train_step=50000, num_steps=100, seed=0, ):
    total_samples = 1000
    prng = random.PRNGKey(seed)
    prng_toy, prng_update, prng_z, prng_z_disc, prng_z_gen, prng = random.split(prng, 6)
    dataloader = ToyData.GaussianMixture(prng_toy, total_samples, num_modes, var)
    batch = dataloader.get_next_batch()
    gan1, d_state1, g_state1 = GAN.load_gan_from_file(gan_path)
    gan2, d_state2, g_state2 = GAN.load_gan_from_file(gan_path)

    z = random.normal(prng_z, (100, 2))
    s = gan1.generate_samples(z, g_state1)
    modes = ToyData.GaussianMixture.create_2d_mean_matrix(num_modes)
    mode_ids, dists = ToyGAN_eval_vis._get_nearest_modes(s, modes)

    for i in range(train_step, train_step+num_steps):
        z_disc = random.normal(prng_z_disc, (1000, 2))
        z_gen = random.normal(prng_z_gen, (1000, 2))
        prng_z_gen, prng = random.split(prng, 2)
        z_gen_rating1 = gan1.rate_samples(gan1.generate_samples(z_gen, g_state1), d_state1)
        z_gen_rating2 = gan2.rate_samples(gan2.generate_samples(z_gen, g_state1), d_state2)
        z_gen1 = z_gen[z_gen_rating1.argsort()]
        z_gen2 = z_gen[z_gen_rating2.argsort()]
        d_state1, g_state1, d_loss_value1, g_loss_value1 = gan_train_step(i, z_disc, z_gen1[-(total_samples*75//100):], batch,
                                                                          gan1, d_state1, g_state1)
        d_state2, g_state2, d_loss_value2, g_loss_value2 = gan_train_step(i, z_disc, z_gen2[:total_samples*25//100], batch,
                                                                          gan2, d_state2, g_state2)

    s_top = gan1.generate_samples(z, g_state1)
    s_bottom = gan2.generate_samples(z, g_state2)

    dists_top = [jnp.linalg.norm(s_top[i]-modes[mode_ids[i]]) for i in range(total_samples)]
    dists_bottom = [jnp.linalg.norm(s_bottom[i]-modes[mode_ids[i]]) for i in range(total_samples)]

    plt.subplot(2, 2, 1)
    plt.scatter(s[:, 0], s[:, 1], s=2, alpha=0.2)
    plt.gca().set_title('samples before')
    plt.subplot(2, 2, 2)
    plt.scatter(s_top[:, 0], s_top[:, 1], s=2, alpha=0.2)
    plt.gca().set_title('samples after top update')
    plt.subplot(2, 2, 3)
    plt.scatter(s[:, 0], s[:, 1], s=2, alpha=0.2)
    plt.gca().set_title('samples before')
    plt.subplot(2, 2, 4)
    plt.scatter(s_bottom[:, 0], s_bottom[:, 1], label="samples2 after", s=2, alpha=0.2)
    plt.gca().set_title('samples after bottom update')
    plt.show()




