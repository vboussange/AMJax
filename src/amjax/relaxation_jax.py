"""Méthodes de relaxation — pur JAX."""

import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
import numpy as np


def _bcoo_diag(A):
    # extrait diagonale d'une matrice BCOO
    rows = A.indices[:, 0]
    cols = A.indices[:, 1]
    mask = (rows == cols).astype(A.data.dtype)
    diag = jnp.zeros((A.shape[0],), dtype=A.data.dtype)
    return diag.at[rows].add(A.data * mask)


def matrix_diagonal(A):
    if isinstance(A, jsparse.BCOO):
        return _bcoo_diag(A)
    return jnp.diag(jnp.asarray(A))


def inverse_diagonal(A):
    d = matrix_diagonal(A)
    safe = jnp.where(d != 0, d, 1.0) 
    # obligé de mettre un "safe", car JAX va calculer les 3 conditions en même temps 
    # et donc il va effectuer la division par 0 alors que NumPy aurait d'abord calculé 
    # la condition et se serait arrêté là si 0.
    return jnp.where(d != 0, 1.0 / safe, 0.0)


def jacobi(A, x, b, Dinv, iterations=1, omega=1.0):
    """Relaxation de Jacobi amortie : x_new = x + omega * D^{-1} * (b - A @ x)

    Paramètres
    ----------
    A          : matrice sparse
    x          : itéré courant
    b          : second membre
    Dinv       : inverse de la diagonale de A (pré-calculé)
    iterations : nombre de sweeps
    omega      : paramètre d'amortissement
    """
    print("jacobi_jax called")
    print("type(A)   =", type(A))
    print("type(x)   =", type(x))
    print("type(b)   =", type(b))
    print("type(A@x) =", type(A @ x))

    def body(_, xk):
        # Jacobi pas besoin du numéro d'itération i, on fait la même chose à chaque tour donc "_"
        temp = Dinv * (b - A @ xk) + xk   # D^{-1}(b - (A-D)xk)
        return (1.0 - omega) * xk + omega * temp

    return lax.fori_loop(0, iterations, body, x)


def approximate_spectral_radius(A, Dinv, n_iter=15, seed=0):
    """Estime le rayon spectral de D^{-1}A par itération de puissance.

    Appelée une fois à la construction (hors JIT).
    -> Pour withrho=True

    Paramètres
    ----------
    A      : matrice sparse BCOO
    Dinv   : inverse de la diagonale de A
    n_iter : nombre d'itérations
    seed   : graine aléatoire
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
    """Pas de relaxation"""
    return x
