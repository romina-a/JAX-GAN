from Models import GAN
import jax
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path_to_gan", required=True, type=str, help="path to the GAN pickle file")
parser.add_argument("--path_to_images", required=True, type=str, help="path to save the images")
parser.add_argument("--seed", required=False, type=int, default=10, help="seed for random latent generation")
parser.add_argument("--iter", required=False, type=int, default=10, help="num of samples, iter")
parser.add_argument("--batch", required=False, type=int, default=1, help="num of samples, per iter")


args = vars(parser.parse_args())
path_to_GAN = args["path_to_gan"]
path_to_images =args["path_to_images"]
seed = args["seed"]
iter = args["iter"]
batch = args["batch"]


gan, d_state, g_state = GAN.load_gan_from_file(path_to_GAN)
prng = jax.random.PRNGKey(seed)

count = 0
for i in range(iter):
  print(i)
  prng_to_use, prng = jax.random.split(prng)
  z = jax.random.normal(prng, (batch, 100))

  ims = gan.generate_samples(z, g_state)
  ims.block_until_ready()
  ims= np.array(ims)
  plt.imshow((ims[0].reshape(32, 32, 3)+1.0)/2.0)
  plt.show()
  for j in range(batch):
    im = np.clip((ims[j].reshape(32, 32, 3)+1.0)/2.0, 0, 1)
    plt.imsave(path_to_images+'{:05d}.png'.format(count), im)
    count += 1
    print("count is", count)