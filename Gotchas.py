import numpy as np
from jax import grad, jit
from jax import lax
from jax import random
from jax import make_jaxpr
import jax
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams

rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False


# NOTE: ----------------------------  Don't use global with jit ----------------------------
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions
def impure_print_side_effect(x):
    print("This is the side effect")
    return x


def test_impure_print_side_effect():
    print("hi")
    # The side-effects appear during the first run
    print("First call: ", jit(impure_print_side_effect)(4.))

    # Subsequent runs with parameters of same type and shape may not show the side-effect
    # This is because JAX now invokes a cached compilation of the function
    print("Second call: ", jit(impure_print_side_effect)(5.))

    # JAX re-runs the Python function when the type or shape of the argument changes
    print("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))


g = 0.


def impure_uses_globals(x):
    print("inside g is: ", g)
    return x + g


def test_impure_uses_globals():
    # JAX captures the value of the global during the first run
    print("First call: ", jit(impure_uses_globals)(4.))
    global g
    g = 10  # Update the global

    # Subsequent runs may silently use the cached value of the globals
    print("Second call: ", jit(impure_uses_globals)(5.))

    # JAX re-runs the Python function when the type or shape of the argument changes
    # This will end up reading the latest value of the global
    print("Third call, different type: ", jit(impure_uses_globals)(jnp.array([4.])))


def impure_saves_global(x):
    global g
    g = x
    return x


def test_impure_saves_global():
    # JAX runs once the transformed function with special Traced values for arguments
    # in other words, changes the type of global if you changed it inside the function!!
    print("global before: ", g)
    print("First call: ", jit(impure_saves_global)(4.))
    print("Saved global: ", g)  # Saved global has an internal JAX value


# NOTE: ----------------------------  Don't use iterators with jit ----------------------------


def test_iterator():
    # lax.fori_loop
    array = jnp.arange(10)
    print(lax.fori_loop(0, 10, lambda i, x: x + array[i], 0))  # expected result 45
    iterator = iter(range(10))
    print(lax.fori_loop(0, 10, lambda i, x: x + next(iterator), 0))  # unexpected result 0


# NOTE: ----------------------------  Random with jit ----------------------------
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#%F0%9F%94%AA-Random-Numbers


from jax import random


def test_random():
    key = random.PRNGKey(0)
    print(random.normal(key, shape=(1,)))
    print(key)
    # No no no!
    print(random.normal(key, shape=(1,)))
    print(key)
    print("old key", key)
    key, subkey = random.split(key) # <---
    normal_pseudorandom = random.normal(subkey, shape=(1,))
    print("    \---SPLIT --> new key   ", key)
    print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
    # NOTE: [propagate the key and use subkey each time you need a new random
    key, subkey = random.split(key)  # <---
    normal_pseudorandom = random.normal(subkey, shape=(1,))
    print("    \---SPLIT --> new key   ", key)
    print("             \--> new subkey", subkey, "--> normal", normal_pseudorandom)
    # NOTE: you can generate more than one subkey
    print("subkeys:")
    key, *subkeys = random.split(key, 4)
    for subkey in subkeys:
        print(random.normal(subkey, shape=(1,)))


if __name__ == '__main__':
    test_random()
