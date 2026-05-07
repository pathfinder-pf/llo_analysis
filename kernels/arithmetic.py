"""Element-wise arithmetic Pallas kernels: add / sub / mul / div."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import jax.experimental.pallas.tpu as pltpu


# --------------- kernel functions ---------------

def add_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]


def sub_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] - y_ref[...]


def mul_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] * y_ref[...]


def div_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] / y_ref[...]

def mod_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] % 2

def round_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] // 2

# --------------- wrappers (single-block, no grid) ---------------

def _binary_op(kernel, x, y, interpret):
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
        interpret=interpret,
    )(x, y)


def _multi_dimession_op(kernel, x, y, interpret):
    assert x.shape == y.shape
    x = x.reshape(-1, x.shape[2], x.shape[3])
    y = y.reshape(-1, y.shape[2], y.shape[3])
    assert len(x.shape) >= 3
    return pl.pallas_call(
        kernel,
        grid=(x.shape[0],),
        out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype)],
        out_specs = [pl.BlockSpec((1, x.shape[1], x.shape[2]), lambda x: (x,0, 0))],
        in_specs=[pl.BlockSpec((1, x.shape[1], x.shape[2]), lambda x: (x, 0, 0)), pl.BlockSpec((1,x.shape[1], x.shape[2]), lambda x: (x,0,0))],
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
            ),
            disable_bounds_checks=True
        ),
    )(x, y)

def matrix_add(x, y, *, interpret=False):
    return _multi_dimession_op(add_kernel, x, y, interpret)

def vector_add(x, y, *, interpret=False):
    return _binary_op(add_kernel, x, y, interpret)


def vector_sub(x, y, *, interpret=False):
    return _binary_op(sub_kernel, x, y, interpret)


def vector_mul(x, y, *, interpret=False):
    return _binary_op(mul_kernel, x, y, interpret)


def vector_div(x, y, *, interpret=False):
    return _binary_op(div_kernel, x, y, interpret)

def vector_mod(x, *, interpret=False):
    return _binary_op(mod_kernel, x, x, interpret)

def vector_round(x, *, interpret=False):
    return _binary_op(round_kernel, x, x, interpret)
