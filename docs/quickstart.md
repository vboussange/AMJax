# Quickstart

AMJax starts from a PyAMG hierarchy. Build the hierarchy with PyAMG, convert it
once, then use the resulting `MultilevelSolver` in JAX code.

## Direct solve

```python
import jax
import jax.numpy as jnp
import pyamg

from amjax import MultilevelSolver

A = pyamg.gallery.poisson((100, 100), format="csr")
b = jnp.ones(A.shape[0])

pyamg_ml = pyamg.ruge_stuben_solver(A)
ml = MultilevelSolver.from_pyamg(pyamg_ml)

solve = jax.jit(lambda rhs: ml.solve(rhs, tol=1e-10, maxiter=100, cycle="V"))
x = solve(b)
```

## Preconditioned Krylov solve

`aspreconditioner` exposes one multigrid cycle as a JAX callable. Pass it to a
JAX Krylov solver when you want the multigrid hierarchy to act as a
preconditioner.

```python
import jax.scipy.sparse.linalg
from jax.experimental import sparse as jsparse

A_jax = jsparse.BCOO.from_scipy_sparse(A)
M = ml.aspreconditioner(cycle="V")

x, info = jax.scipy.sparse.linalg.cg(A_jax, b, M=M, tol=1e-10, maxiter=30)
```

## Batched right-hand sides

Use `jax.vmap` when the same hierarchy is applied to many right-hand sides.

```python
B = jnp.ones((64, A.shape[0]))
solve_batch = jax.jit(jax.vmap(lambda rhs: ml.solve(rhs, tol=1e-8, maxiter=100)))
X = solve_batch(B)
```

## Differentiation

AMJax's direct solve has a custom VJP rule, so scalar objectives that depend on
the solution can be differentiated with JAX.

```python
def objective(rhs):
    return jnp.sum(ml.solve(rhs, tol=1e-10, maxiter=100))

grad_b = jax.grad(objective)(b)
```

To differentiate with respect to the stored nonzero entries of the finest-level
matrix, keep the sparsity pattern fixed and expose only the values as JAX
variables. The `A` keyword is an AMJax extension; the PyAMG hierarchy remains
the one built from the original matrix.

```python
A_coo = A.tocoo()
rows = jnp.asarray(A_coo.row)
cols = jnp.asarray(A_coo.col)
values0 = jnp.asarray(A_coo.data)

def matrix_from_values(values):
    return jnp.zeros(A.shape, dtype=values.dtype).at[rows, cols].set(values)

def objective_nnz(values):
    A_values = matrix_from_values(values)
    x = ml.solve(b, A=A_values, tol=1e-8, maxiter=100, cycle="V")
    return jnp.sum(x)

grad_values = jax.grad(objective_nnz)(values0)
```

The same pattern can be used with a fixed AMJax preconditioner inside
Krylov-based methods.

```python
M = ml.aspreconditioner(cycle="V")

def objective_nnz_pcg(values):
    A_values = matrix_from_values(values)
    x, info = jax.scipy.sparse.linalg.cg(
        A_values,
        b,
        M=M,
        tol=1e-8,
        maxiter=30,
    )
    return jnp.sum(x)

grad_values_pcg = jax.grad(objective_nnz_pcg)(values0)
```

We have observed that this pattern gives smooth convergence in practice for our inverse problems.
