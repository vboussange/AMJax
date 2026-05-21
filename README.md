# AMJax

JAX implementation of algebraic multigrid (AMG) solvers for sparse linear systems. 

> **AMJax is a two-phase solver:**
>
> <u>Phase 1:</u> Hierarchy construction using PyAMG and NumPy
>
> PyAMG builds the AMG hierarchy: coarsening, prolongation and restriction operators. This is a one-time setup step, run on CPU.
>
> <u>Phase 2:</u> Solve using JAX
>
> The hierarchy is converted to JAX BCOO sparse arrays. All solve steps can be JIT-compiled, GPU-accelerated, and are compatible with `jax.vmap` for batched right-hand sides.

## Installation
For now,

```bash
uv add git+https://github.com/vboussange/AMJax.git
```

but we will soon add the package to PyPi 🙃.

## Usage

### Direct solve

```python
import pyamg
import jax
import jax.numpy as jnp
from amjax import AMJAXSolver

A  = pyamg.gallery.poisson((100, 100), format="csr")
b  = jnp.ones(A.shape[0])

ml = AMJAXSolver.from_pyamg(pyamg.ruge_stuben_solver(A))

solve = jax.jit(lambda b: ml.solve(b, tol=1e-10, maxiter=100))
x = solve(b)
```

### Preconditioning

`AMJAXSolver` exposes a preconditioner compatible with any JAX Krylov solver:

```python
from jax.experimental import sparse as jsparse

A_jax = jsparse.BCOO.from_scipy_sparse(A)
M = ml.aspreconditioner(cycle='V')

x, info = jax.scipy.sparse.linalg.cg(A_jax, b, M=M, tol=1e-10, maxiter=30)
```

### Batched solve with `jax.vmap`

```python
import numpy as np

B = jnp.array(np.random.rand(4, A.shape[0]))  # (n_rhs, n)
solve_batch = jax.jit(jax.vmap(lambda b: ml.solve(b, tol=1e-8, maxiter=100)))
X = solve_batch(B)
```

## Features

- V-cycle compiled with `jax.jit`
- AMG preconditioning for JAX Krylov solvers
- Compatible with `jax.vmap` for batched right-hand sides
- Planned: `jax.grad` support

## Solvers

`AMJAXSolver.from_pyamg` accepts any hierarchy produced by a PyAMG factory:

| Factory | Intended for |
|---------|--------------|
| `pyamg.smoothed_aggregation_solver` | SPD systems, standard aggregation AMG |
| `pyamg.rootnode_solver` | SPD systems, robust for anisotropic problems |
| `pyamg.pairwise_solver` | SPD systems, fast setup, weaker convergence |
| `pyamg.ruge_stuben_solver` | General SPD systems, classical C/F splitting |
| `pyamg.air_solver` | Non-symmetric systems |

**Current limitations:** V-cycle only. `jacobi` coarse solver only.

## Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/benchmark.ipynb)