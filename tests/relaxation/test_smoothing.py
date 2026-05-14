"""Test AMJax smoother configuration."""
import numpy as np
from numpy.testing import TestCase
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
import pyamg
from pyamg.gallery import poisson

from amjax.multilevel import MultilevelSolver as MultilevelSolver
from amjax.relaxation.smoothing import change_smoothers, rebuild_smoother

jax.config.update("jax_enable_x64", True)

# symmetric: jacobi method 
methods_sym = [
    ('jacobi', {'iterations': 1, 'withrho': True}),
    ('jacobi', {'iterations': 2, 'withrho': True}),
]

# None method: symmetric but too slow to check residual
methods_none = [None]

# asymmetric: different methods or different iteration counts
methods_asym = [
    (('jacobi', {'iterations': 2}), ('jacobi', {'iterations': 1})),
    ('jacobi', None),
    (None, 'jacobi'),
]


class TestSmoothing(TestCase):
    def setUp(self):
        np.random.seed(0)
        self.A_scipy = poisson((20, 20), format='csr')
        self.A = jsparse.BCOO.from_scipy_sparse(self.A_scipy)
        self.b = jnp.array(np.random.rand(self.A_scipy.shape[0]))
        self.pyamg_ml = pyamg.ruge_stuben_solver(self.A_scipy, coarse_solver='jacobi')

    def test_solver_parameters(self):
        for method in methods_sym:
            ml = MultilevelSolver.from_pyamg(
                self.pyamg_ml,
                presmoother=method,
                postsmoother=method,
            )
            x = ml.solve(self.b, tol=1e-5, maxiter=100)
            assert jnp.linalg.norm(self.b - self.A @ x) < 1e-5 * jnp.linalg.norm(self.b)
            assert ml.symmetric_smoothing

        for method in methods_none:
            ml = MultilevelSolver.from_pyamg(
                self.pyamg_ml,
                presmoother=method,
                postsmoother=method,
            )
            assert ml.symmetric_smoothing

        for pre, post in methods_asym:
            ml = MultilevelSolver.from_pyamg(self.pyamg_ml)
            change_smoothers(ml, presmoother=pre, postsmoother=post)
            assert not ml.symmetric_smoothing


class TestRebuildSmoother(TestCase):
    def test_rebuild_smoother(self):
        """rebuild_smoother preserves smoother names and adapts to a new matrix."""
        from scipy.sparse import eye_array
        from amjax.relaxation.relaxation import inverse_diagonal

        A_scipy = poisson((20,), format='csr')
        pyamg_ml = pyamg.ruge_stuben_solver(A_scipy, coarse_solver='jacobi')
        ml = MultilevelSolver.from_pyamg(
            pyamg_ml,
            presmoother=('jacobi', {'iterations': 1}),
            postsmoother=None,
        )

        lvl = ml.levels[0]
        assert lvl.presmoother.__name__ == 'jacobi'
        assert lvl.postsmoother.__name__ == 'none'

        # update A and Dinv, then rebuild closures
        A_new = jsparse.BCOO.from_scipy_sparse(eye_array(A_scipy.shape[0], format='csr'))
        lvl.A = A_new
        lvl.Dinv = inverse_diagonal(A_new)
        rebuild_smoother(lvl)

        assert lvl.presmoother.__name__ == 'jacobi'
        assert lvl.postsmoother.__name__ == 'none'

        # D^{-1} = I for identity matrix: one Jacobi step from x=0 gives x=b exactly
        b = jnp.ones(A_scipy.shape[0], dtype=jnp.float64)
        x = lvl.presmoother(lvl.A, jnp.zeros_like(b), b)
        assert jnp.allclose(x, b)
