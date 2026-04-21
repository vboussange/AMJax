"""Tests du smoother Jacobi : vérifie que le résidu diminue après lissage"""

import numpy as np
import jax
import jax.numpy as jnp
import pyamg
import pytest

from amjax import MultilevelSolverJAX
from amjax.relaxation_jax import jacobi, inverse_diagonal
from jax.experimental.sparse import BCOO

jax.config.update("jax_enable_x64", True)


def make_level(n):
    """Retourne le niveau fin d'un solveur AMG pour la grille n×n."""
    A  = pyamg.gallery.poisson((n, n), format="csr")
    ml = MultilevelSolverJAX.from_pyamg(pyamg.ruge_stuben_solver(A))
    return ml.levels[0]


@pytest.mark.parametrize("n", [10, 20, 50])
def test_jacobi_reduces_residual(n):
    """Une itération Jacobi doit réduire le résidu"""
    lvl = make_level(n)
    rng = np.random.default_rng(0)
    b   = jnp.array(rng.random(n * n))
    x0  = jnp.array(rng.random(n * n))

    r_before = float(jnp.linalg.norm(b - lvl.A @ x0))
    x1       = jacobi(lvl.A, x0, b, lvl.Dinv, iterations=1, omega=1.0)
    r_after  = float(jnp.linalg.norm(b - lvl.A @ x1))

    assert r_after < r_before, (
        f"Jacobi n'a pas réduit le résidu pour n={n} "
        f"({r_before:.2e} -> {r_after:.2e})"
    )


@pytest.mark.parametrize("n", [10, 20, 50])
def test_jacobi_withrho_reduces_residual(n):
    """Jacobi withrho=True doit atténuer le résidu
    """
    A  = pyamg.gallery.poisson((n, n), format="csr")
    ml = MultilevelSolverJAX.from_pyamg(
        pyamg.ruge_stuben_solver(A),
        presmoother =("jacobi", {"iterations": 1, "withrho": True}),
        postsmoother=("jacobi", {"iterations": 1, "withrho": True}),
    )
    lvl = ml.levels[0]
    rng = np.random.default_rng(1)
    b   = jnp.array(rng.random(n * n))
    x   = jnp.array(rng.random(n * n))

    r_before = float(jnp.linalg.norm(b - lvl.A @ x))
    for _ in range(10):
        x = lvl.presmoother(lvl.A, x, b)
    r_after = float(jnp.linalg.norm(b - lvl.A @ x))

    assert r_after < r_before, (
        f"smoother withrho n'a pas réduit le résidu pour n={n} "
        f"({r_before:.2e} -> {r_after:.2e})"
    )


@pytest.mark.parametrize("n", [10, 20, 50])
def test_inverse_diagonal(n):
    """inverse_diagonal doit retourner 1/diag(A) correctement."""
    A    = pyamg.gallery.poisson((n, n), format="csr")
    Ajax = BCOO.from_scipy_sparse(A)
    Dinv = inverse_diagonal(Ajax)

    diag = np.array(A.diagonal())
    assert np.allclose(np.array(Dinv), 1.0 / diag, rtol=1e-6), \
        "inverse_diagonal incorrect"
