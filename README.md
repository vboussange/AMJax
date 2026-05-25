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

### Differentiation solve with `jax.grad`

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
- AMG preconditioning for JAX Krylov solvers
- Compatible with `jax.vmap` for batched right-hand sides
- Compatible with `jax.grad`

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

For JAX methods, reported times are solve times only (JIT compilation excluded).

| | |
|---|---|
| Hierarchy | Ruge-Stüben |
| Coarse solver | `pinv` |
| Cycle | V |
| Smoother | Jacobi |
| n | 1,000 |
| dtype | f64 |
| tol | 1e-10 |
| maxiter | 100 |
| Residual | ‖b − Ax‖ / ‖b‖ |


- **Single solve** (1 RHS, seconds)

| Device | PyAMG | PyAMG + CG | AMJax | AMJax + CG |
|--------|------:|-----------:|------:|-----------:|
| CPU    | 1.897 | 1.508      | —     | —          |
| GPU    | —     | —          | 0.119 | 0.091      |

- **Batched solve** (K=64, `jax.vmap`, seconds)

| Device | PyAMG | PyAMG + CG | AMJax | AMJax + CG |
|--------|------:|-----------:|------:|-----------:|
| CPU    | 139.4 | 110.7      | 192.4 |          |
| GPU    | —     | —          | 6.553 | 4.820      |


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/benchmark.ipynb)