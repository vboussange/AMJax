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
from amjax import MultilevelSolver

A  = pyamg.gallery.poisson((100, 100), format="csr")
b  = jnp.ones(A.shape[0])

ml = MultilevelSolver.from_pyamg(pyamg.ruge_stuben_solver(A))

solve = jax.jit(lambda b: ml.solve(b, tol=1e-10, maxiter=100))
x = solve(b)
```

### Preconditioning

`MultilevelSolver` exposes a preconditioner compatible with any JAX Krylov solver:

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

### Differentiation solve with `jax.grad`

```python
A_dense = jnp.array(A.toarray())
f = lambda A: jnp.sum(ml.solve(b, A=A, tol=1e-10, maxiter=100))
grad = jax.grad(f)(A_dense)
```

### Differentiation with preconditioning

```python
f = lambda A: jnp.sum(jax.scipy.sparse.linalg.cg(A, b, M=M, tol=1e-10)[0])
grad = jax.grad(f)(A_dense)
```

## Features

- V, W and F cycles compiled with `jax.jit`
- Coarse solvers: `jacobi`, `lu`, `qr`, `pinv`
- Smoothers: `jacobi`
- AMG preconditioning for JAX Krylov solvers
- Compatible with `jax.vmap` for batched right-hand sides
- Compatible with `jax.grad`

## Solvers

`MultilevelSolver.from_pyamg` accepts any hierarchy produced by a PyAMG factory:

| Factory | Intended for |
|---------|--------------|
| `pyamg.smoothed_aggregation_solver` | SPD systems, standard aggregation AMG |
| `pyamg.rootnode_solver` | SPD systems, robust for anisotropic problems |
| `pyamg.pairwise_solver` | SPD systems, fast setup, weaker convergence |
| `pyamg.ruge_stuben_solver` | General SPD systems, classical C/F splitting |
| `pyamg.air_solver` | Non-symmetric systems |

**Current limitations:** `jacobi` smoother only.

## Benchmark

JAX solve times exclude JIT compilation.

| | |
|---|---|
| Hierarchy | Ruge-Stüben |
| Coarse solver | `pinv` |
| Cycle | V |
| Smoother | Jacobi |
| n | 500 |
| dtype | f64 |
| tol | 1e-10 |
| maxiter | 100 |
| Residual | ‖b − Ax‖ / ‖b‖ |


- **Single solve** (1 RHS)

| Speedup | CPU | GPU |
|--|----:|----:|
| AMJax vs PyAMG | 0.39 | 20.7 |
| AMJax + CG vs PyAMG + CG | 0.42 | 33.0 |

- **Batched solve** (K=64, `jax.vmap`)

| Speedup | CPU | GPU |
|--|----:|----:|
| AMJax vs PyAMG | 0.60 | 48.0 |
| AMJax + CG vs PyAMG + CG | 0.70 | 61.6 |


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/benchmark.ipynb)

For a more detailed benchmark, see [fannymissillier.github.io/AMJax-docs](https://fannymissillier.github.io/AMJax-docs/).
