"""Test AMJAXSolver class."""
import numpy as np
from numpy.testing import TestCase, assert_almost_equal, assert_equal
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from pyamg.gallery import poisson

from amjax.multilevel import coarse_grid_solver, MultilevelSolver as AMJAXSolver
from amjax.relaxation.relaxation import inverse_diagonal

jax.config.update("jax_enable_x64", True)


class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = []
        cases.append(jsparse.BCOO.fromdense(jnp.array(np.diag(np.arange(1, 5, dtype=float)))))
        cases.append(jsparse.BCOO.from_scipy_sparse(poisson((4,), format='csr')))
        cases.append(jsparse.BCOO.from_scipy_sparse(poisson((4, 4), format='csr')))

        # only 'jacobi' is supported as a coarse solver in AMJax
        for A in cases:
            lvl = AMJAXSolver.Level()
            lvl.A = A
            lvl.Dinv = inverse_diagonal(A)

            # method should be approximately exact for small matrices
            s = coarse_grid_solver('jacobi', lvl, iterations=500, omega=2/3)
            b = jnp.arange(A.shape[0], dtype=float)

            x = s(A, jnp.zeros_like(b), b)
            assert_almost_equal(np.array(A @ x), np.array(b), decimal=3)

            # subsequent calls use the same pre-computed Dinv
            x = s(A, jnp.zeros_like(b), b)
            assert_almost_equal(np.array(A @ x), np.array(b), decimal=3)

    def test_aspreconditioner(self):
        import pyamg
        np.random.seed(1331277597)

        A_scipy = poisson((50, 50), format='csr')
        A = jsparse.BCOO.from_scipy_sparse(A_scipy)
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        # AMJax builds its hierarchy from a PyAMG hierarchy
        pyamg_ml = pyamg.ruge_stuben_solver(A_scipy, coarse_solver='jacobi')
        ml = AMJAXSolver.from_pyamg(
            pyamg_ml,
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        # AMJax only supports the V cycle;
        # aspreconditioner() returns a JAX callable —> use jax.scipy CG to stay fully in JAX
        for cycle in ['V']:
            M = ml.aspreconditioner(cycle=cycle)
            x, _info = jax.scipy.sparse.linalg.cg(A, b, M=M, tol=1e-8, maxiter=30)
            # cg satisfies convergence in the Euclidean norm
            assert jnp.linalg.norm(b - A @ x) < 1e-8 * jnp.linalg.norm(b)

    def test_solve(self):
        import pyamg
        np.random.seed(30459128)

        A_scipy = poisson((50, 50), format='csr')
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        # AMJax builds its hierarchy from a PyAMG hierarchy
        pyamg_ml = pyamg.ruge_stuben_solver(A_scipy, coarse_solver='jacobi')
        ml = AMJAXSolver.from_pyamg(
            pyamg_ml,
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        # AMJax solve is JIT-compatible
        solve = jax.jit(lambda b: ml.solve(b, maxiter=100, tol=1e-8))
        x = solve(b)
        assert jnp.linalg.norm(b - jsparse.BCOO.from_scipy_sparse(A_scipy) @ x) < 1e-8 * jnp.linalg.norm(b)

    def test_vmap_compatibility(self):
        import pyamg
        np.random.seed(1883275855)

        A_scipy = poisson((50, 50), format='csr')
        B = jnp.array(np.random.rand(4, A_scipy.shape[0]))

        ml = AMJAXSolver.from_pyamg(
            pyamg.ruge_stuben_solver(A_scipy, coarse_solver='jacobi'),
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        solve = jax.jit(jax.vmap(lambda b: ml.solve(b, maxiter=100, tol=1e-8)))
        X = solve(B)
        A_jax = jsparse.BCOO.from_scipy_sparse(A_scipy)
        assert jnp.all(jax.vmap(lambda b, x: jnp.linalg.norm(b - A_jax @ x) < 1e-8 * jnp.linalg.norm(b))(B, X))

    def test_cycle_complexity(self):
        def dummy_solver(A, x, b):
            del A, b
            return x

        # four levels — BCOO.fromdense replaces the non-existent BCOO.csr_array
        levels = []
        levels.append(AMJAXSolver.Level())
        levels[0].A = jsparse.BCOO.fromdense(jnp.ones((10, 10)))
        levels[0].P = jsparse.BCOO.fromdense(jnp.ones((10, 5)))
        levels.append(AMJAXSolver.Level())
        levels[1].A = jsparse.BCOO.fromdense(jnp.ones((5, 5)))
        levels[1].P = jsparse.BCOO.fromdense(jnp.ones((5, 3)))
        levels.append(AMJAXSolver.Level())
        levels[2].A = jsparse.BCOO.fromdense(jnp.ones((3, 3)))
        levels[2].P = jsparse.BCOO.fromdense(jnp.ones((3, 2)))
        levels.append(AMJAXSolver.Level())
        levels[3].A = jsparse.BCOO.fromdense(jnp.ones((2, 2)))

        # one level hierarchy
        mg = AMJAXSolver(levels[:1], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 100.0/100.0)  # 1
        assert_equal(mg.cycle_complexity(cycle='W'), 100.0/100.0)  # 1

        # two level hierarchy
        mg = AMJAXSolver(levels[:2], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 225.0/100.0)  # 2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 225.0/100.0)  # 2,1

        # three level hierarchy
        mg = AMJAXSolver(levels[:3], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 259.0/100.0)  # 2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 318.0/100.0)  # 2,4,2

        # four level hierarchy
        mg = AMJAXSolver(levels[:4], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 272.0/100.0)  # 2,2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 388.0/100.0)  # 2,4,8,4

        # AMLI and F cycles are not implemented in AMJax
        self.assertRaises(NotImplementedError, mg.cycle_complexity, 'AMLI')
        self.assertRaises(NotImplementedError, mg.cycle_complexity, 'F')

    def test_from_pyamg_compatibility(self):
        import pyamg
        A = poisson((20, 20), format='csr')
        for factory in [
            pyamg.smoothed_aggregation_solver,
            pyamg.rootnode_solver,
            pyamg.pairwise_solver,
            pyamg.ruge_stuben_solver,
            pyamg.air_solver,
        ]:
            ml = AMJAXSolver.from_pyamg(factory(A, coarse_solver='jacobi'))
            self.assertGreater(len(ml.levels), 1)


class TestPrecisionMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        # JAX defaults to float32 -> verify the coarse solver works in both precisions
        for dtype in [jnp.float32, jnp.float64]:
            diag_vals = jnp.array(np.arange(1, 5, dtype=np.float64), dtype=dtype)
            A = jsparse.BCOO.fromdense(jnp.diag(diag_vals))

            lvl = AMJAXSolver.Level()
            lvl.A = A
            lvl.Dinv = inverse_diagonal(A)

            b = diag_vals  # solution is x = [1, 1, 1, 1]
            s = coarse_grid_solver('jacobi', lvl, iterations=1, omega=1.0)
            x = s(A, jnp.zeros_like(b), b)
            assert_almost_equal(np.array(A @ x), np.array(b), decimal=5)

            # subsequent calls use the same pre-computed Dinv
            x = s(A, jnp.zeros_like(b), b)
            assert_almost_equal(np.array(A @ x), np.array(b), decimal=5)