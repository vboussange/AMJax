# AMJax

`AMJax` brings algebraic multigrid (AMG) methods to JAX for solving large sparse
linear systems. It bridges [`PyAMG`](https://github.com/pyamg/pyamg) and JAX by
converting PyAMG-built hierarchies into `jax.jit`, `jax.vmap`, GPU-compatible,
and differentiable multilevel solvers and preconditioners.

Documentation: <https://vboussange.github.io/AMJax/>

## Installation

```bash
uv add amjax
```

## Usage

### Direct solve

```python
import jax
import jax.numpy as jnp
import pyamg

from amjax import MultilevelSolver

A = pyamg.gallery.poisson((100, 100), format="csr")
b = jnp.ones(A.shape[0])

ml = MultilevelSolver.from_pyamg(pyamg.ruge_stuben_solver(A))

solve = jax.jit(lambda rhs: ml.solve(rhs, tol=1e-10, maxiter=100, cycle="V"))
x = solve(b)
```

### Preconditioning

`MultilevelSolver` exposes a preconditioner compatible with JAX Krylov solvers:

```python
import jax.scipy.sparse.linalg
from jax.experimental import sparse as jsparse

A_jax = jsparse.BCOO.from_scipy_sparse(A)
M = ml.aspreconditioner(cycle="V")

x, info = jax.scipy.sparse.linalg.cg(A_jax, b, M=M, tol=1e-10, maxiter=30)
```

### Batched solve with `jax.vmap`

```python
B = jnp.ones((64, A.shape[0]))
solve_batch = jax.jit(jax.vmap(lambda rhs: ml.solve(rhs, tol=1e-8, maxiter=100)))
X = solve_batch(B)
```

### Differentiating through the solve with `jax.grad`

```python
def objective(rhs):
    return jnp.sum(ml.solve(rhs, tol=1e-10, maxiter=100))

grad_b = jax.grad(objective)(b)
```

## Recommendation

For 2D Poisson problems, start with a Ruge-Stuben hierarchy, V-cycle, Jacobi
smoothing, and a `pinv` coarse solve. Use AMJax as a preconditioner for
conjugate gradient (`AMJax + PCG`) when runtime and convergence both matter.
Use `f64` for tight residuals; use `f32` only for speed-first workloads. When
solving many right-hand sides, batch with `jax.vmap` and use `k=64` when memory
allows.

The detailed benchmark summary is generated from the latest committed
`benchmarks/latest_summary.json` and published in the
[benchmark docs](https://vboussange.github.io/AMJax/benchmarks/). The full
benchmark can be rerun from
[`benchmarks/benchmark.ipynb`](benchmarks/benchmark.ipynb), or from the shell:

```bash
benchmarks/run_full_benchmark.sh
```

## Features

- V, W, and F cycles compiled with `jax.jit`
- Coarse solvers: `jacobi`, `lu`, `qr`, `pinv`
- Smoothers: `jacobi`
- AMG preconditioning for JAX Krylov solvers
- `jax.vmap` support for batched right-hand sides
- `jax.grad` support through direct solves and preconditioned Krylov solves

## PyAMG interop

`MultilevelSolver.from_pyamg` accepts hierarchies produced by PyAMG solver
factories, including:

| Factory | Typical use |
|---------|-------------|
| `pyamg.ruge_stuben_solver` | Classical AMG, strong Poisson baseline |
| `pyamg.smoothed_aggregation_solver` | SPD systems, aggregation AMG |
| `pyamg.rootnode_solver` | SPD systems, robust aggregation variant |
| `pyamg.pairwise_solver` | Fast setup; use with care for large standalone solves |
| `pyamg.air_solver` | Non-symmetric systems |

For AMG setup details, use the
[PyAMG documentation](https://pyamg.readthedocs.io/). AMJax documents only the
JAX conversion and solve layer.

## Limitations

- Hierarchy construction is delegated to PyAMG, so setup happens in Python and
  is not differentiable through the hierarchy itself.
- A fully native JAX hierarchy is currently blocked by sparse-sparse Galerkin
  products such as `P.T @ A @ P`, whose sparsity pattern is not known at JIT
  trace time.
- Benchmark speedups combine solver implementation, GPU execution, JIT
  compilation, and batching, so they are practical comparisons rather than
  solver-only hardware-controlled comparisons.
- Precise GPU memory accounting is not yet reported.
- Pairwise should not be the default standalone hierarchy for large systems;
  it is safer as a preconditioner.

## Roadmap

- Add more smoothers and coarse-grid solvers that fit JAX's static compilation
  model.
- Investigate native JAX hierarchy construction for the Pairwise case, whose
  binary prolongator gives a predictable sparsity pattern.
- Add rigorous GPU memory profiling.
- Explore complex matrices and additional Krylov/preconditioner combinations
  when concrete use cases justify the maintenance cost.
