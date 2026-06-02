# AMJax

AMJax bridges PyAMG and JAX for algebraic multigrid (AMG) solvers: it converts PyAMG-constructed hierarchies into `jax.{jit,grad,vmap}`-compatible, multi-level solvers and preconditioners for large sparse linear systems.

## Installation
```bash
uv add amjax
```

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

### Differentiating through the solve with `jax.grad`

```python
f = lambda b: jnp.sum(ml.solve(b, tol=1e-10, maxiter=100))
grad = jax.grad(f)(b)
```

### Differentiation with preconditioning

```python
f = lambda b: jnp.sum(jax.scipy.sparse.linalg.cg(A_jax, b, M=M, tol=1e-10)[0])
grad = jax.grad(f)(b)
```

## Features

- V, W and F cycles compiled with `jax.jit`
- Coarse solvers: `jacobi`, `lu`, `qr`, `pinv`
- Smoothers: `jacobi`
- AMG preconditioning for JAX Krylov solvers (e.g. `jax.scipy.sparse.linalg.cg`)
- `jax.vmap` support for batched right-hand sides
- `jax.grad` support through both direct solve and preconditioned Krylov solvers

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

An exhaustive benchmark can be run in Colab: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/benchmark.ipynb) 
<!-- And results are reported at XXX -->

Some key insights on **speedup gains vs PyAMG-based counterpart**:


| Scenario | Method | CPU | GPU |
|----------|--------|----:|----:|
| Single solve ($Ax=b$, $b \in \mathbb{R}^n$) | AMJax | - | ~16× |
| Single solve ($Ax=b$, $b \in \mathbb{R}^n$) | AMJax + CG | - | ~17× |
| Batched solve ($AX=B$, $B \in \mathbb{R}^{n \times K}$, $K=64$, `jax.vmap`) | AMJax | 0.7× | ~21× |
| Batched solve ($AX=B$, $B \in \mathbb{R}^{n \times K}$, $K=64$, `jax.vmap`) | AMJax + CG | - | ~23× |


> Settings: Ruge-Stüben hierarchy, V-cycle, Jacobi smoother, `pinv` coarse solver, $n = 1{,}000$, f64, rtol $= 10^{-10}$, max 100 iterations. JAX times exclude JIT compilation. GPU speedup is relative to the PyAMG CPU counterpart.