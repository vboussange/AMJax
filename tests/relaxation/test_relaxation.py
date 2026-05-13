"""Test JAX relaxation methods."""
import numpy as np
from numpy.testing import TestCase, assert_almost_equal
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from scipy.sparse import diags_array
from pyamg.gallery import poisson

from amjax.relaxation.relaxation import jacobi, inverse_diagonal, approximate_spectral_radius

jax.config.update("jax_enable_x64", True)


class TestCommonRelaxation(TestCase):
    def setUp(self):
        # AMJax only implements Jacobi relaxation
        self.cases = [jacobi]

    def test_single_precision(self):
        for method in self.cases:
            A_scipy = poisson((4,), format='csr').astype('float32')
            A = jsparse.BCOO.from_scipy_sparse(A_scipy)
            Dinv = inverse_diagonal(A)
            b = jnp.arange(A.shape[0], dtype=jnp.float32)
            x = jnp.zeros_like(b)
            x = method(A, x, b, Dinv)
            assert x.dtype == jnp.float32

    def test_double_precision(self):
        for method in self.cases:
            A_scipy = poisson((4,), format='csr').astype('float64')
            A = jsparse.BCOO.from_scipy_sparse(A_scipy)
            Dinv = inverse_diagonal(A)
            b = jnp.arange(A.shape[0], dtype=jnp.float64)
            x = jnp.zeros_like(b)
            x = method(A, x, b, Dinv)
            assert x.dtype == jnp.float64


class TestRelaxation(TestCase):
    def test_jacobi(self):
        def make_A(N):
            A_scipy = diags_array([2*np.ones(N), -np.ones(N), -np.ones(N)],
                                  offsets=[0, -1, 1], shape=(N, N), format='csr')
            A = jsparse.BCOO.from_scipy_sparse(A_scipy)
            return A, inverse_diagonal(A)

        # unlike PyAMG, jacobi returns a new x instead of modifying in place
        N = 1
        A, Dinv = make_A(N)
        x = jacobi(A, jnp.arange(N, dtype=jnp.float64), jnp.zeros(N, dtype=jnp.float64), Dinv)
        assert_almost_equal(np.array(x), [0])

        N = 3
        A, Dinv = make_A(N)
        x = jacobi(A, jnp.zeros(N, dtype=jnp.float64), jnp.arange(N, dtype=jnp.float64), Dinv)
        assert_almost_equal(np.array(x), [0.0, 0.5, 1.0])

        N = 3
        A, Dinv = make_A(N)
        x = jacobi(A, jnp.arange(N, dtype=jnp.float64), jnp.zeros(N, dtype=jnp.float64), Dinv)
        assert_almost_equal(np.array(x), [0.5, 1.0, 0.5])

        N = 1
        A, Dinv = make_A(N)
        x = jacobi(A, jnp.arange(N, dtype=jnp.float64), jnp.array([10], dtype=jnp.float64), Dinv)
        assert_almost_equal(np.array(x), [5])

        N = 3
        A, Dinv = make_A(N)
        x = jacobi(A, jnp.arange(N, dtype=jnp.float64), jnp.array([10, 20, 30], dtype=jnp.float64), Dinv)
        assert_almost_equal(np.array(x), [5.5, 11.0, 15.5])

        N = 3
        A, Dinv = make_A(N)
        x0 = jnp.arange(N, dtype=jnp.float64)
        x = jacobi(A, x0, jnp.array([10, 20, 30], dtype=jnp.float64), Dinv, omega=1.0/3.0)
        assert_almost_equal(np.array(x),
                            2.0/3.0*np.array(x0) + 1.0/3.0*np.array([5.5, 11.0, 15.5]))

    def test_approximate_spectral_radius(self):
        from scipy.sparse import diags as sp_diags
        from scipy.sparse.linalg import eigsh

        # For a diagonal matrix, D^{-1}A = I -> spectral radius = 1.0 exactly
        N = 4
        A = jsparse.BCOO.fromdense(jnp.diag(2 * jnp.ones(N, dtype=jnp.float64)))
        Dinv = inverse_diagonal(A)
        rho = approximate_spectral_radius(A, Dinv)
        assert_almost_equal(rho, 1.0, decimal=5)

        # For a Poisson problem, compare the estimate against scipy's largest eigenvalue
        A_scipy = poisson((10,), format='csr').astype('float64')
        A = jsparse.BCOO.from_scipy_sparse(A_scipy)
        Dinv = inverse_diagonal(A)
        rho = approximate_spectral_radius(A, Dinv)

        DinvA = sp_diags(1.0 / A_scipy.diagonal()) @ A_scipy
        rho_ref = eigsh(DinvA, k=1, which='LM', return_eigenvectors=False)[0]
        assert abs(rho - rho_ref) / rho_ref < 0.1
