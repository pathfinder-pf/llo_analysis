#!/usr/bin/env python3
# python -m pytest test_and_dump.py::test_add
import argparse
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kernels.arithmetic import vector_add, vector_sub, vector_mul, vector_div, vector_mod, vector_round
from kernels.control_flow import (
    vector_relu, vector_clamp, vector_cond, vector_cumsum
)
from kernels.nested import normalize, compound_op, pairwise_diff_sum
from kernels.matmul import simple_matmul, tiled_matmul


HLO_DUMP_PATH = "dumps/hlo"
LLO_DUMP_PATH = "dumps/llo"

os.environ["XLA_FLAGS"] = (
    f"--xla_dump_hlo_as_text "
    f"--xla_dump_to={HLO_DUMP_PATH} "
    f"--xla_dump_hlo_pass_re=.* "
)

os.environ["LIBTPU_INIT_ARGS"] = (
    f"--xla_jf_dump_to={LLO_DUMP_PATH} "
    f"--xla_jf_dump_hlo_text=true "
    f"--xla_jf_dump_llo_text=true "
    f"--xla_jf_dump_llo_html=false "
    f"--xla_jf_dump_llo_static_gaps=true "
    f"--xla_jf_emit_annotations=true "
    f"--xla_jf_debug_level=2"
)

N = 256
x = jnp.array(np.random.randn(N).astype(np.float32))
y = jnp.array(np.random.randn(N).astype(np.float32) + 2.0)

BS = 128
x_c = jnp.array(np.random.randn(N).astype(np.float32) * 3)

x_n = jnp.array(np.random.randn(128).astype(np.float32))
y_n = jnp.array(np.random.randn(128).astype(np.float32))

M, K, Nb = 64, 64, 64
x_m = jnp.array(np.random.randn(M, K).astype(np.float32))
y_m = jnp.array(np.random.randn(K, Nb).astype(np.float32))


def test_add():
    fn_i = partial(vector_add, interpret=False)
    z = fn_i(x, y)

def test_sub():
    fn_i = partial(vector_sub, interpret=False)
    z = fn_i(x, y)

def test_mul():
    fn_i = partial(vector_mul, interpret=False)
    z = fn_i(x, y)

def test_div():
    fn_i = partial(vector_div, interpret=False)
    z = fn_i(x, y)

def test_mod():
    x = jnp.arange(10)
    fn_i = partial(vector_mod, interpret=False)
    y = fn_i(x)

def test_round():
    x = jnp.arange(10)
    fn_i = partial(vector_round, interpret=False)
    y = fn_i(x)

def test_relu():
    fn_relu = partial(vector_relu, block_size=BS, interpret=False)
    y = fn_relu(x_c)

def test_clamp():
    fn_clamp = partial(vector_clamp, block_size=BS, interpret=False)
    y = fn_clamp(x_c)

def test_cond():
    fn_cond = partial(vector_cond, block_size=BS, interpret=False)
    res_cond = fn_cond(x_c)

def test_cumsum():
    x_cs = jnp.array(np.random.randn(64).astype(np.float32))
    fn_cs = partial(vector_cumsum, interpret=False)
    y = fn_cs(x_cs)

def test_norm():
    fn_norm = partial(normalize, interpret=False)
    y = fn_norm(x_n)

def test_op():
    fn_comp = partial(compound_op, interpret=False)
    y = fn_comp(x_n, y_n)

def test_pairwise():
    x_nl = jnp.array(np.random.randn(32).astype(np.float32))
    fn_nl = partial(pairwise_diff_sum, interpret=False)
    y = fn_nl(x_nl)

def test_matmul():
    fn_sm = partial(simple_matmul, interpret=False)
    y = fn_sm(x_m, y_m)

def test_tiled():
    fn_tm = partial(tiled_matmul, bm=32, bn=32, bk=32, interpret=False)
    y = fn_tm(x_m, y_m)

if __name__ == "__main__":
    pytest.main([__file__])