"""Relaxation methods for JAX-compatible multigrid solvers."""

import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
import numpy as np


def _bcoo_diag(A):
    """Extract the diagonal of a BCOO sparse matrix as a 1-D dense array."""
    rows = A.indices[:, 0]
    cols = A.indices[:, 1]
    mask = (rows == cols).astype(A.data.dtype)
    diag = jnp.zeros((A.shape[0],), dtype=A.data.dtype)
    return diag.at[rows].add(A.data * mask)


def matrix_diagonal(A):
    """Return the diagonal of A as a 1-D array.

    Parameters
    ----------
    A : BCOO or ndarray
        Input matrix.

    Returns
    -------
    ndarray
        1-D array of diagonal entries.
    """
    if isinstance(A, jsparse.BCOO):
        return _bcoo_diag(A)
    return jnp.diag(jnp.asarray(A))


def inverse_diagonal(A):
    """Return the element-wise inverse of the diagonal of A.

    Zero diagonal entries are mapped to zero in the output.

    Parameters
    ----------
    A : BCOO or ndarray
        Input matrix.

    Returns
    -------
    ndarray
        1-D array of inverse diagonal entries.

    Notes
    -----
    JAX evaluates both branches of ``jnp.where`` simultaneously, so a
    naive ``1.0 / d`` would produce a NaN even when guarded by the
    condition.  The safe divisor substitutes 1.0 for zero entries before
    division, then masks the result back to zero.
    """
    d = matrix_diagonal(A)
    safe = jnp.where(d != 0, d, 1.0)
    return jnp.where(d != 0, 1.0 / safe, 0.0)


def jacobi(A, x, b, Dinv, iterations=1, omega=1.0):
    """Perform damped Jacobi relaxation on the linear system Ax = b.

    Applies ``iterations`` sweeps of::

        x <- (1 - omega) * x + omega * (x + D^{-1}(b - A x))

    Parameters
    ----------
    A : BCOO sparse matrix
        n x n system matrix.
    x : ndarray
        Current iterate, length n.
    b : ndarray
        Right-hand side, length n.
    Dinv : ndarray
        Pre-computed element-wise inverse of the diagonal of A, length n.
    iterations : int
        Number of sweeps to perform.
    omega : scalar
        Damping parameter.

    Returns
    -------
    ndarray
        Updated iterate after ``iterations`` sweeps.
    """

    def body(_, xk):
        temp = Dinv * (b - A @ xk) + xk
        return (1.0 - omega) * xk + omega * temp

    return lax.fori_loop(0, iterations, body, x)


def approximate_spectral_radius(A, Dinv, n_iter=15, seed=0):
    """Estimate the spectral radius of D^{-1} A by power iteration.

    Called once at construction time, outside of JIT.

    Parameters
    ----------
    A : BCOO sparse matrix
        n x n system matrix.
    Dinv : ndarray
        Element-wise inverse of the diagonal of A.
    n_iter : int
        Number of power iterations.
    seed : int
        Random seed for the initial vector.

    Returns
    -------
    float
        Estimated spectral radius of D^{-1} A.
    """
    rng = np.random.default_rng(seed)
    v = jnp.array(rng.standard_normal(A.shape[0]), dtype=Dinv.dtype)
    v = v / jnp.linalg.norm(v)
    rho = 1.0
    for _ in range(n_iter):
        w = Dinv * (A @ v)
        rho = float(jnp.linalg.norm(w))
        if rho == 0.0:
            break
        v = w / rho
    return rho


def none(A, x, b):
    """Return x unchanged (no relaxation)."""
    return x
