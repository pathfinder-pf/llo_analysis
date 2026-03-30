"""Element-wise arithmetic Pallas kernels: add / sub / mul / div."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


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
