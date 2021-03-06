import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import time
import numpy as np
from jax import device_put


# NOTE: Random is different from numpy, the state is external
#  1. reproducing 2. parallel execution ~ global state and other reasons


# https://jax.readthedocs.io/en/latest/notebooks/quickstart.html

# key = random.PRNGKey(0)
# x = random.normal(key, (10,))
# print(x)
#
# size = 3000
# x = random.normal(key, (size, size), dtype=jnp.float32)
# t = time.time()
# jnp.dot(x, x.T).block_until_ready()  # runs on the GPU
# print("jnp took: {}s".format(time.time()-t))
#
# x = np.random.normal(size=(size, size)).astype(np.float32)
# t = time.time()
# jnp.dot(x, x.T).block_until_ready()
# print("np array jnp dot took: {}s".format(time.time()-t))
#
# x = np.random.normal(size=(size, size)).astype(np.float32)
# x = device_put(x)
# t = time.time()
# jnp.dot(x, x.T).block_until_ready()
# print("np array jnp dot with device_put took: {}s".format(time.time()-t))
#
# x = np.random.normal(size=(size, size)).astype(np.float32)
# t = time.time()
# np.dot(x, x.T)
# print("np took: {}s".format(time.time()-t))

# note~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))


key = random.PRNGKey(0)
# key, subkey = random.split(key)
x = random.normal(key, (1000000,))
t = time.time()
selu(x).block_until_ready()
print("time taken no jit:", time.time()-t)
selu_jit = jit(selu)
key, subkey = random.split(key)
x = random.normal(subkey, (1000000,))
t = time.time()
selu(x).block_until_ready()
print("time taken with jit:", time.time()-t)
print("BYE")

# note~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
