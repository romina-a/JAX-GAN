from Models import GAN
import ToyData
from jax import random, value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import ToyGAN_eval_vis
import ToyData


def adjusted_gan_train_step(i, d_z, g_z, real_samples, gan, d_state, g_state):
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


def perform_update(prng, gan_path, num_modes, var, train_step, num_steps, gradient_samples):
    prng_toy, prng = random.split(prng)
    dataloader = ToyData.GaussianMixture(prng_toy, gradient_samples, num_modes, var)
    gan1, d_state1, g_state1 = GAN.load_gan_from_file(gan_path)
    gan2, d_state2, g_state2 = GAN.load_gan_from_file(gan_path)

    for i in range(train_step, train_step+num_steps):
        print(f"doing step {i}")
        batch = dataloader.get_next_batch()
        prng_z_disc, prng_z_gen, prng = random.split(prng, 3)
        z_disc = random.normal(prng_z_disc, (gradient_samples, 2))
        z_gen = random.normal(prng_z_gen, (gradient_samples, 2))

        z_gen_rating1 = gan1.rate_samples(gan1.generate_samples(z_gen, g_state1), d_state1)
        z_gen_rating2 = gan2.rate_samples(gan2.generate_samples(z_gen, g_state2), d_state2)
        z_gen1 = z_gen[z_gen_rating1.argsort(axis=0).flatten()]
        z_gen2 = z_gen[z_gen_rating2.argsort(axis=0).flatten()]
        d_state1, g_state1, d_loss_value1, g_loss_value1 = adjusted_gan_train_step(i, z_disc, z_gen1[-(gradient_samples * 75 // 100):],
                                                                                   batch, gan1, d_state1, g_state1)
        d_state2, g_state2, d_loss_value2, g_loss_value2 = adjusted_gan_train_step(i, z_disc, z_gen2[:gradient_samples * 25 // 100],
                                                                                   batch, gan2, d_state2, g_state2)

    return d_state1, g_state1, d_state2, g_state2,


def perform_analysis(gan_path, num_modes, var, train_step=50000, num_steps=1, seed=0,
                     gradient_samples=10000, visualization_samples=10000):
    prng = random.PRNGKey(seed)
    prng_train, prng_z, prng = random.split(prng, 3)
    gan, d_state, g_state = GAN.load_gan_from_file(gan_path)
    d_state1, g_state1, d_state2, g_state2, = perform_update(prng_train, gan_path, num_modes, var,
                                                             train_step=train_step,
                                                             num_steps=num_steps,
                                                             gradient_samples=gradient_samples,)

    z = random.normal(prng_z, (visualization_samples, 2))
    s = gan.generate_samples(z, g_state)
    s_top = gan.generate_samples(z, g_state1)
    s_bottom = gan.generate_samples(z, g_state2)

    modes = ToyData.GaussianMixture.create_2d_mean_matrix(num_modes)
    mode_ids, dists = ToyGAN_eval_vis._get_nearest_modes(s, modes)

    dists_top = [jnp.linalg.norm(s_top[i]-modes[mode_ids[i]]) for i in range(visualization_samples)]
    dists_bottom = [jnp.linalg.norm(s_bottom[i]-modes[mode_ids[i]]) for i in range(visualization_samples)]

    plt.subplot(2, 2, 1)
    plt.scatter(s[:, 0], s[:, 1], s=2, alpha=0.2)
    plt.gca().axis([-0.3, 0.3, -0.3, 0.3])
    plt.gca().set_title('samples before')
    plt.subplot(2, 2, 2)
    plt.scatter(s_top[:, 0], s_top[:, 1], s=2, alpha=0.2)
    plt.gca().axis([-0.3, 0.3, -0.3, 0.3])
    plt.gca().set_title('samples after top update')
    plt.subplot(2, 2, 3)
    plt.scatter(s[:, 0], s[:, 1], s=2, alpha=0.2)
    plt.gca().axis([-0.3, 0.3, -0.3, 0.3])
    plt.gca().set_title('samples before')
    plt.subplot(2, 2, 4)
    plt.scatter(s_bottom[:, 0], s_bottom[:, 1], label="samples2 after", s=2, alpha=0.2)
    plt.gca().axis([-0.3, 0.3, -0.3, 0.3])
    plt.gca().set_title('samples after bottom update')
    plt.tight_layout()
    plt.savefig("./output_ims/position_separate.png")
    plt.show()

    plt.scatter(s[:, 0], s[:, 1], s=2, alpha=0.2, label="before")
    plt.scatter(s_top[:, 0], s_top[:, 1], s=2, alpha=0.2, label="top")
    plt.scatter(s[:, 0], s[:, 1], s=2, alpha=0.2, label="bottom")
    plt.gca().axis([-0.3, 0.3, -0.3, 0.3])
    plt.legend()
    plt.savefig("./output_ims/position.png")
    plt.show()

    dif1 = np.array(dists - dists_top)
    dif2 = np.array(dists - dists_bottom)

    sd = np.sqrt(var)

    plt.subplot(2, 1, 1)
    plt.bar([1, 2, 3, 4, 5], [sum(dif1[dists <= sd]),
                              sum(dif1[(dists > sd) & (dists <= 2*sd)]),
                              sum(dif1[(dists > 2*sd) & (dists <= 3*sd)]),
                              sum(dif1[(dists > 3*sd) & (dists <= 4*sd)]),
                              sum(dif1[(dists > 4*sd)])])
    plt.gca().set_title('change in distance with top update')

    plt.subplot(2, 1, 2)
    plt.bar([1, 2, 3, 4, 5], [sum(dif2[dists <= sd]),
                              sum(dif2[(dists > sd) & (dists <= 2 * sd)]),
                              sum(dif2[(dists > 2 * sd) & (dists <= 3 * sd)]),
                              sum(dif2[(dists > 3 * sd) & (dists <= 4 * sd)]),
                              sum(dif2[(dists > 4 * sd)])])
    plt.gca().set_title('change in distance with bottom update')
    plt.tight_layout()
    plt.savefig("./output_ims/dist.png")
    plt.show()

    return dists, dists_top, dists_bottom





