"""Tests de convergence pour MultilevelSolverJAX (V-cycle et PCG)"""

import numpy as np
import jax
import jax.numpy as jnp
import pyamg
import pytest

from amjax import MultilevelSolverJAX
from scipy.sparse.linalg import spsolve

jax.config.update("jax_enable_x64", True)


def make_solver(n):
    """Construit un solveur JAX AMG pour la grille n×n de Poisson 2D."""
    A  = pyamg.gallery.poisson((n, n), format="csr")
    ml = MultilevelSolverJAX.from_pyamg(
        pyamg.ruge_stuben_solver(A),
        presmoother =("jacobi", {"iterations": 1, "withrho": True}),
        postsmoother=("jacobi", {"iterations": 1, "withrho": True}),
        coarse_solver="jacobi",
    )
    return A, ml


@pytest.mark.parametrize("n", [10, 20, 50])
def test_vcycle_residual(n):
    """Le résidu relatif après solve doit être < 1e-8."""
    A, ml = make_solver(n)
    b = jnp.array(np.random.default_rng(0).random(n * n))

    x = jax.jit(lambda ml, b: ml.solve(b, tol=1e-10, maxiter=100))(ml, b)

    residual = float(jnp.linalg.norm(b - ml.levels[0].A @ x) / jnp.linalg.norm(b))
    assert residual < 1e-8, f"résidu trop grand ({residual:.2e}) pour n={n}"


@pytest.mark.parametrize("n", [10, 20, 50])
def test_vcycle_accuracy(n):
    """La solution JAX doit être proche de spsolve (RMSE < 1e-7)."""

    A, ml = make_solver(n)
    b     = np.random.default_rng(1).random(n * n)
    x_ref = spsolve(A, b)

    x = jax.jit(lambda ml, b: ml.solve(b, tol=1e-10, maxiter=100))(ml, jnp.array(b))
    x = np.array(x)

    rmse = float(np.sqrt(np.mean((x - x_ref) ** 2)))
    assert rmse < 1e-7, f"RMSE trop grand ({rmse:.2e}) pour n={n}"


@pytest.mark.parametrize("n", [10, 20, 50])
def test_pcg_residual(n):
    """PCG préconditionné par un V-cycle doit converger (résidu < 1e-8)."""
    A, ml = make_solver(n)
    b = jnp.array(np.random.default_rng(2).random(n * n))

    @jax.jit
    def pcg(ml, b):
        M    = ml.aspreconditioner()
        x, _ = jax.scipy.sparse.linalg.cg(
            lambda v: ml.levels[0].A @ v, b,
            M=M, tol=1e-10, maxiter=500,
        )
        return x

    x        = pcg(ml, b)
    residual = float(jnp.linalg.norm(b - ml.levels[0].A @ x) / jnp.linalg.norm(b))
    assert residual < 1e-8, f"résidu PCG trop grand ({residual:.2e}) pour n={n}"
