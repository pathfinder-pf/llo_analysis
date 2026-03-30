"""Pallas kernels with control flow: conditionals and loops."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# ===================== Conditional kernels =====================

def relu_kernel(x_ref, o_ref):
    """ReLU: jnp.where conditional."""
    x = x_ref[...]
    o_ref[...] = jnp.where(x > 0, x, 0.0)


def clamp_kernel(x_ref, o_ref):
    """Clamp to [-1, 1]: nested jnp.where."""
    x = x_ref[...]
    o_ref[...] = jnp.where(x > 1.0, 1.0, jnp.where(x < -1.0, -1.0, x))


def cond_kernel(x_ref, o_ref):
    """Branch on program_id: even blocks *2, odd blocks negate."""
    pid = pl.program_id(axis=0)
    x = x_ref[...]
    o_ref[...] = jnp.where(pid % 2 == 0, x * 2.0, -x)


# ===================== Loop kernels =====================

def cumsum_kernel(x_ref, o_ref):
    """Cumulative sum via jax.lax.fori_loop."""
    n = x_ref.shape[0]

    def body(i, acc):
        new_acc = acc + x_ref[i]
        o_ref[i] = new_acc
        return new_acc

    jax.lax.fori_loop(0, n, body, jnp.float32(0.0))


def poly_eval_kernel(coeffs_ref, x_ref, o_ref):
    """Polynomial evaluation via Horner's method (fori_loop).

    coeffs = [a0, a1, ..., an]  (ascending powers)
    result  = a0 + a1*x + a2*x^2 + ... + an*x^n
    """
    n = coeffs_ref.shape[0]
    x = x_ref[...]

    def body(i, acc):
        return acc * x + coeffs_ref[n - 1 - i]

    o_ref[...] = jax.lax.fori_loop(0, n, body, jnp.zeros_like(x))


# ===================== Wrappers =====================

def _tiled_unary(kernel, x, block_size, interpret):
    n = x.shape[0]
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        grid=(n // block_size,),
        in_specs=[
            pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
        ],
        out_specs=pl.BlockSpec(block_shape=(block_size,), index_map=lambda i: (i,)),
        interpret=interpret,
    )(x)


def vector_relu(x, *, block_size=128, interpret=False):
    return _tiled_unary(relu_kernel, x, block_size, interpret)


def vector_clamp(x, *, block_size=128, interpret=False):
    return _tiled_unary(clamp_kernel, x, block_size, interpret)


def vector_cond(x, *, block_size=128, interpret=False):
    return _tiled_unary(cond_kernel, x, block_size, interpret)


def vector_cumsum(x, *, interpret=False):
    """No grid – entire vector is one block."""
    return pl.pallas_call(
        cumsum_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x)


def poly_eval(coeffs, x, *, interpret=False):
    """No grid – entire arrays are single blocks."""
    return pl.pallas_call(
        poly_eval_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(coeffs, x)
