"""Pallas matrix multiplication kernels."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


# ===================== Kernel functions =====================

def simple_matmul_kernel(x_ref, y_ref, o_ref):
    """Single-block matmul: o = x @ y."""
    o_ref[...] = jnp.dot(x_ref[...], y_ref[...])


def tiled_matmul_kernel(x_ref, y_ref, o_ref):
    """Tiled matmul: accumulate partial dot-products.

    Grid iterates over (M/bm, N/bn, K/bk).
    Multiple K-tiles map to the same output block -> += accumulates.
    """
    o_ref[...] += jnp.dot(x_ref[...], y_ref[...])


# ===================== Wrappers =====================

def simple_matmul(x, y, *, interpret=False):
    """Whole-matrix matmul in a single Pallas block (no grid)."""
    m, _ = x.shape
    _, n = y.shape
    return pl.pallas_call(
        simple_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        interpret=interpret,
    )(x, y)


def tiled_matmul(x, y, *, bm=32, bn=32, bk=32, interpret=False):
    """Tiled matmul with grid over (M, N, K) tile indices."""
    m, k = x.shape
    _, n = y.shape
    return pl.pallas_call(
        tiled_matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec(block_shape=(bm, bk), index_map=lambda i, j, k: (i, k)),
            pl.BlockSpec(block_shape=(bk, bn), index_map=lambda i, j, k: (k, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(bm, bn), index_map=lambda i, j, k: (i, j)),
        interpret=interpret,
    )(x, y)
