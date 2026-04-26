# AMJax

JAX implementation of algebraic multigrid (AMG) solvers for sparse linear systems.

Built on top of [PyAMG](https://github.com/pyamg/pyamg) for hierarchy construction and [JAX](https://github.com/google/jax) for JIT-compiled, GPU-compatible solves.

## Installation

```bash
pip install amjax
```

## Usage

```python
import pyamg
import jax
import jax.numpy as jnp
from amjax import MultilevelSolverJAX

A  = pyamg.gallery.poisson((100, 100), format="csr")
b  = jnp.ones(A.shape[0])

ml = MultilevelSolverJAX.from_pyamg(pyamg.ruge_stuben_solver(A))

solve = jax.jit(lambda ml, b: ml.solve(b, tol=1e-10, maxiter=100))
x     = solve(ml, b)
```

## Features

- V-cycle solver compiled with `jax.jit`
- GPU-compatible via XLA
- AMG-preconditioned CG via `jax.scipy.sparse.linalg.cg`
- Compatible with `jax.vmap` for batched solves

## Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/rss.ipynb)

## Tests

```bash
uv run pytest tests/
```
