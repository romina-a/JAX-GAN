from Models import BCE_from_logits, MSE, BCE
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad

logit_output = jnp.array([10e100, -10e100, -1, 0, 1, 10e-100, -10e-100, 10e100, -10e100, -1, 0, 1, 10e-100, -10e-100, ])
logit_output_labels = jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
prob_output = jnp.array([0, 1e-10, 0.5, 0.999999, 1, 0, 1e-10, 0.5 ,0.999999, 1])
prob_output_labels = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def _test_loss_function_logit(func, output, labels):
    value, grad = value_and_grad(func)(output, labels)
    assert jnp.isnan(grad).any() == False, func.__name__+": found nan in grad"
    assert jnp.isnan(value).any() == False, func.__name__+": found nan in value"
    value, grad = value_and_grad(jit(func))(output, labels)
    assert jnp.isnan(grad).any() == False, func.__name__ + "_jitted: found nan in grad"
    assert jnp.isnan(value).any() == False, func.__name__ + "_jitted: found nan in value"


def test_BCE_from_logits():
    _test_loss_function_logit(BCE_from_logits, logit_output, logit_output_labels)


def test_MSE():
    _test_loss_function_logit(MSE, logit_output, logit_output_labels)


def test_BCE():
    _test_loss_function_logit(BCE, prob_output, prob_output_labels)





