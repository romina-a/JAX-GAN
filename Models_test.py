from Models import BCE_from_logits, MSE
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad


def test_BCE_from_logits():
    logits = jnp.array([10e100, -10e100, -1, 0, 1, 10e-100, -10e-100, 10e100, -10e100, -1, 0, 1, 10e-100, -10e-100, ])
    labels = jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    value, grad = value_and_grad(BCE_from_logits)(logits, labels)
    assert jnp.isnan(grad).any() == False, "test_BCE_from_logits found nan in grad"
    assert jnp.isnan(value).any() == False, "test_BCE_from_logits found nan in value"


def test_jitted_BCE_from_logits():
    logits = jnp.array([10e100, -10e100, 0, 10e-100, -10e-100, 10e100, -10e100, 0, 10e-100, -10e-100, ])
    labels = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    jitted_BCE_from_logits = jit(BCE_from_logits)
    value, grad = value_and_grad(jitted_BCE_from_logits)(logits, labels)
    assert jnp.isnan(grad).any() == False, "test_BCE_from_logits found nan in grad"
    assert jnp.isnan(value).any() == False, "test_BCE_from_logits found nan in value"


def test_MSE():
    logits = jnp.array([10e100, -10e100, -1, 0, 1, 10e-100, -10e-100, 10e100, -10e100, -1, 0, 1, 10e-100, -10e-100, ])
    labels = jnp.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    value, grad = value_and_grad(MSE)(logits, labels)
    assert jnp.isnan(grad).any() == False, "test_MSE found nan in grad"
    assert jnp.isnan(value).any() == False, "test_MSE found nan in value"


def test_jitted_MSE():
    logits = jnp.array([10e100, -10e100, 0, 10e-100, -10e-100, 10e100, -10e100, 0, 10e-100, -10e-100, ])
    labels = jnp.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    jitted_BCE_from_logits = jit(MSE)
    value, grad = value_and_grad(jitted_BCE_from_logits)(logits, labels)
    assert jnp.isnan(grad).any() == False, "test_jitted_MSE found nan in grad"
    assert jnp.isnan(value).any() == False, "test_jitted_MSE found nan in value"