"""Pallas kernels with nested / compound operations."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# ===================== Kernel functions =====================

def normalize_kernel(x_ref, o_ref):
    """Layer-norm style normalize (nested arithmetic).

    Steps: mean -> variance -> (x - mean) / sqrt(var + eps)
    """
    x = x_ref[...]
    mean = jnp.mean(x)
    var = jnp.mean((x - mean) ** 2)
    o_ref[...] = (x - mean) / jnp.sqrt(var + 1e-5)


def compound_kernel(x_ref, y_ref, o_ref):
    """Compound nested expression mixing several activations.

    result = softplus(x) * sigmoid(y) + relu(x - y)
    """
    x = x_ref[...]
    y = y_ref[...]
    softplus_x = jnp.log1p(jnp.exp(x))                # softplus
    sigmoid_y = 1.0 / (1.0 + jnp.exp(-y))              # sigmoid (explicit)
    relu_diff = jnp.maximum(x - y, 0.0)                 # relu
    o_ref[...] = softplus_x * sigmoid_y + relu_diff


def nested_loop_kernel(x_ref, o_ref):
    """Nested fori_loop: row-wise sum of absolute differences.

    o[i] = sum_j |x[i] - x[j]|
    """
    n = x_ref.shape[0]

    def outer_body(i, _carry):
        def inner_body(j, acc):
            return acc + jnp.abs(x_ref[i] - x_ref[j])
        o_ref[i] = jax.lax.fori_loop(0, n, inner_body, jnp.float32(0.0))
        return _carry

    jax.lax.fori_loop(0, n, outer_body, jnp.float32(0.0))


# ===================== Wrappers =====================

def normalize(x, *, interpret=False):
    return pl.pallas_call(
        normalize_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)


def compound_op(x, y, *, interpret=False):
    return pl.pallas_call(
        compound_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, y)


def pairwise_diff_sum(x, *, interpret=False):
    return pl.pallas_call(
        nested_loop_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)
