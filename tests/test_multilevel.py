"""Test MultilevelSolver class."""
import numpy as np
from numpy.testing import TestCase, assert_almost_equal, assert_equal
import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from pyamg.gallery import poisson

from amjax.multilevel import coarse_grid_solver, MultilevelSolver
from amjax.relaxation.relaxation import inverse_diagonal

jax.config.update("jax_enable_x64", True)


class TestMultilevel(TestCase):
    def test_coarse_grid_solver(self):
        cases = [
            jsparse.BCOO.fromdense(jnp.array(np.diag(np.arange(1, 5, dtype=float)))),
            jsparse.BCOO.from_scipy_sparse(poisson((4,), format='csr')),
            jsparse.BCOO.from_scipy_sparse(poisson((4, 4), format='csr')),
        ]

        for A in cases:
            b = jnp.arange(A.shape[0], dtype=float)

            for solver, kwargs in [
                ('jacobi', {'iterations': 500, 'omega': 2/3}),
                ('pinv', {}),
                ('lu', {}),
                ('qr', {}),
            ]:
                lvl = MultilevelSolver.Level()
                lvl.A = A
                lvl.Dinv = inverse_diagonal(A)

                if solver in ('pinv', 'lu', 'qr'):
                    lvl.A_dense = A.todense()
                    if solver == 'pinv':
                        lvl.A_inv = jnp.linalg.pinv(lvl.A_dense)
                    elif solver == 'lu':
                        lvl.lu_factor, lvl.piv = jax.scipy.linalg.lu_factor(lvl.A_dense)
                    elif solver == 'qr':
                        lvl.Q, lvl.R_mat = jnp.linalg.qr(lvl.A_dense)

                cs = coarse_grid_solver(solver, lvl, **kwargs)
                x = cs(A, jnp.zeros_like(b), b)
                assert_almost_equal(np.array(A @ x), np.array(b), decimal=3)


    def test_aspreconditioner(self):
        import pyamg
        np.random.seed(1331277597)

        A_scipy = poisson((50, 50), format='csr')
        A = jsparse.BCOO.from_scipy_sparse(A_scipy)
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        # AMJax builds its hierarchy from a PyAMG hierarchy
        pyamg_ml = pyamg.ruge_stuben_solver(A_scipy, coarse_solver='jacobi')
        ml = MultilevelSolver.from_pyamg(
            pyamg_ml,
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        # aspreconditioner() returns a JAX callable —> use jax.scipy CG to stay fully in JAX
        for cycle in ['V', 'W', 'F']:
            M = ml.aspreconditioner(cycle=cycle)
            x, _info = jax.scipy.sparse.linalg.cg(A, b, M=M, tol=1e-8, maxiter=30)
            assert jnp.linalg.norm(b - A @ x) < 1e-8 * jnp.linalg.norm(b)

    def test_solve(self):
        import pyamg
        np.random.seed(30459128)

        A_scipy = poisson((20, 20), format='csr')
        b = jnp.array(np.random.rand(A_scipy.shape[0]))
        A_jax = jsparse.BCOO.from_scipy_sparse(A_scipy)

        for coarse_solver in ['pinv', 'lu', 'qr', 'jacobi']:
            ml = MultilevelSolver.from_pyamg(
                pyamg.ruge_stuben_solver(A_scipy),
                presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
                postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
                coarse_solver=coarse_solver,
            )
            for cycle in ['V', 'W', 'F']:
                solve = jax.jit(lambda b, c=cycle: ml.solve(b, maxiter=100, tol=1e-8, cycle=c))
                x = solve(b)
                assert jnp.linalg.norm(b - A_jax @ x) < 1e-8 * jnp.linalg.norm(b)

    def test_vmap_compatibility(self):
        import pyamg
        np.random.seed(1883275855)

        A_scipy = poisson((50, 50), format='csr')
        B = jnp.array(np.random.rand(4, A_scipy.shape[0]))

        ml = MultilevelSolver.from_pyamg(
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

        levels = []
        levels.append(MultilevelSolver.Level())
        levels[0].A = jsparse.BCOO.fromdense(jnp.ones((10, 10)))
        levels[0].P = jsparse.BCOO.fromdense(jnp.ones((10, 5)))
        levels.append(MultilevelSolver.Level())
        levels[1].A = jsparse.BCOO.fromdense(jnp.ones((5, 5)))
        levels[1].P = jsparse.BCOO.fromdense(jnp.ones((5, 3)))
        levels.append(MultilevelSolver.Level())
        levels[2].A = jsparse.BCOO.fromdense(jnp.ones((3, 3)))
        levels[2].P = jsparse.BCOO.fromdense(jnp.ones((3, 2)))
        levels.append(MultilevelSolver.Level())
        levels[3].A = jsparse.BCOO.fromdense(jnp.ones((2, 2)))

        # one level hierarchy
        mg = MultilevelSolver(levels[:1], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 100.0/100.0)  # 1
        assert_equal(mg.cycle_complexity(cycle='W'), 100.0/100.0)  # 1
        assert_equal(mg.cycle_complexity(cycle='F'), 100.0/100.0)  # 1

        # two level hierarchy
        mg = MultilevelSolver(levels[:2], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 225.0/100.0)  # 2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 225.0/100.0)  # 2,1
        assert_equal(mg.cycle_complexity(cycle='F'), 225.0/100.0)  # 2,1

        # three level hierarchy
        mg = MultilevelSolver(levels[:3], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 259.0/100.0)  # 2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 318.0/100.0)  # 2,4,2
        assert_equal(mg.cycle_complexity(cycle='F'), 318.0/100.0)  # 2,4,2

        # four level hierarchy
        mg = MultilevelSolver(levels[:4], dummy_solver)
        assert_equal(mg.cycle_complexity(cycle='V'), 272.0/100.0)  # 2,2,2,1
        assert_equal(mg.cycle_complexity(cycle='W'), 388.0/100.0)  # 2,4,8,4
        assert_equal(mg.cycle_complexity(cycle='F'), 366.0/100.0)  # 2,4,6,3

        # AMLI not implemented
        self.assertRaises(NotImplementedError, mg.cycle_complexity, 'AMLI')

    def test_vjp(self):
        import pyamg
        np.random.seed(42)

        A_scipy = poisson((10, 10), format='csr')
        A_jax = jnp.array(A_scipy.toarray())
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        ml = MultilevelSolver.from_pyamg(
            pyamg.ruge_stuben_solver(A_scipy),
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        rng = np.random.default_rng(0)
        d = jnp.array(rng.standard_normal(A_jax.shape))
        eps = 1e-4

        for cycle in ['V', 'W', 'F']:
            f = lambda A, c=cycle: jnp.sum(ml.solve(b, A=A, tol=1e-10, maxiter=100, cycle=c))

            # compare VJP gradient against central finite differences in a random direction
            grad_vjp = jax.grad(f)(A_jax)
            directional_vjp = jnp.sum(grad_vjp * d)
            directional_fd  = (f(A_jax + eps * d) - f(A_jax - eps * d)) / (2 * eps)
            np.testing.assert_allclose(float(directional_vjp), float(directional_fd), rtol=1e-3)

            assert grad_vjp.shape == A_jax.shape

    def test_vjp_jit_and_vmap(self):
        import pyamg
        np.random.seed(42)

        A_scipy = poisson((10, 10), format='csr')
        A_jax = jnp.array(A_scipy.toarray())
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        ml = MultilevelSolver.from_pyamg(
            pyamg.ruge_stuben_solver(A_scipy),
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        f = lambda A: jnp.sum(ml.solve(b, A=A, tol=1e-10, maxiter=100))

        # jit compiled gradient
        grad_jit = jax.jit(jax.grad(f))(A_jax)
        assert grad_jit.shape == A_jax.shape

        # vmapped gradient
        B = jnp.array(np.random.rand(4, b.shape[0]))
        grad_vmap = jax.vmap(lambda b_i: jax.grad(lambda A: jnp.sum(ml.solve(b_i, A=A, tol=1e-10, maxiter=100)))(A_jax))(B)
        assert grad_vmap.shape == (4,) + A_jax.shape

    def test_vjp_with_fixed_sparsity_values(self):
        import pyamg
        np.random.seed(42)

        A_scipy = poisson((5, 5), format='csr')
        A_jax = jnp.array(A_scipy.toarray())
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        ml = MultilevelSolver.from_pyamg(
            pyamg.ruge_stuben_solver(A_scipy),
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        A_coo = A_scipy.tocoo()
        rows = jnp.asarray(A_coo.row)
        cols = jnp.asarray(A_coo.col)
        values0 = jnp.asarray(A_coo.data)

        def matrix_from_values(values):
            return jnp.zeros(A_scipy.shape, dtype=values.dtype).at[rows, cols].set(values)

        def objective_values(values):
            A_values = matrix_from_values(values)
            return jnp.sum(ml.solve(b, A=A_values, tol=1e-10, maxiter=100))

        def objective_dense(A):
            return jnp.sum(ml.solve(b, A=A, tol=1e-10, maxiter=100))

        grad_values = jax.grad(objective_values)(values0)
        grad_dense = jax.grad(objective_dense)(A_jax)

        assert grad_values.shape == values0.shape
        np.testing.assert_allclose(
            np.array(grad_values),
            np.array(grad_dense[rows, cols]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_vjp_optimization(self):
        import pyamg
        np.random.seed(42)

        A_scipy = poisson((10, 10), format='csr')
        A_jax = jnp.array(A_scipy.toarray())
        b = jnp.array(np.random.rand(A_scipy.shape[0]))

        ml = MultilevelSolver.from_pyamg(
            pyamg.ruge_stuben_solver(A_scipy),
            presmoother=('jacobi', {'iterations': 1, 'withrho': True}),
            postsmoother=('jacobi', {'iterations': 1, 'withrho': True}),
        )

        x_target = jnp.ones(A_scipy.shape[0])

        def loss(A):
            x = ml.solve(b, A=A, tol=1e-10, maxiter=100)
            return jnp.sum((x - x_target) ** 2)

        # gradient descent on A
        lr = 1e-4
        A = A_jax
        losses = [float(loss(A))]
        for _ in range(20):
            A = A - lr * jax.grad(loss)(A)
            losses.append(float(loss(A)))

        assert losses[-1] < losses[0]

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
            ml = MultilevelSolver.from_pyamg(factory(A, coarse_solver='jacobi'))
            self.assertGreater(len(ml.levels), 1)

    def test_not_implemented_smoothers_raise(self):
        import pyamg
        from amjax.relaxation.smoothing import _NOT_IMPLEMENTED_SMOOTHERS

        self.assertTrue(len(_NOT_IMPLEMENTED_SMOOTHERS) > 0,
                        "_NOT_IMPLEMENTED_SMOOTHERS is empty")

        A_scipy = poisson((10, 10), format='csr')
        ml_pyamg = pyamg.ruge_stuben_solver(A_scipy)

        for name in _NOT_IMPLEMENTED_SMOOTHERS:
            with self.assertRaises(NotImplementedError):
                MultilevelSolver.from_pyamg(ml_pyamg, presmoother=(name, {}))

    def test_not_implemented_coarse_solvers_raise(self):
        import pyamg
        from amjax.multilevel import _NOT_IMPLEMENTED_SOLVERS

        self.assertTrue(len(_NOT_IMPLEMENTED_SOLVERS) > 0,
                        "_NOT_IMPLEMENTED_SOLVERS is empty")

        A_scipy = poisson((10, 10), format='csr')
        ml_pyamg = pyamg.ruge_stuben_solver(A_scipy)

        for name in _NOT_IMPLEMENTED_SOLVERS:
            with self.assertRaises(NotImplementedError):
                MultilevelSolver.from_pyamg(ml_pyamg, coarse_solver=name)

    def test_unknown_coarse_solver_raises(self):
        lvl = MultilevelSolver.Level()
        lvl.A = jsparse.BCOO.fromdense(jnp.eye(4))
        lvl.Dinv = inverse_diagonal(lvl.A)
        with self.assertRaises(ValueError):
            coarse_grid_solver('bad_solver', lvl)


class TestPrecisionMultilevel(TestCase):
    def test_dtype_inferred(self):
        import pyamg

        A_scipy = poisson((8, 8), format='csr').astype(np.float32)

        for coarse_solver, coarse_attrs in [
            ('pinv', ['A_inv']),
            ('lu', ['lu_factor']),
            ('qr', ['Q', 'R_mat']),
            ('jacobi', []),
        ]:
            ml = MultilevelSolver.from_pyamg(
                pyamg.smoothed_aggregation_solver(A_scipy),
                coarse_solver=coarse_solver,
            )

            for lvl in ml.levels:
                assert_equal(lvl.A.dtype, jnp.float32)
                assert_equal(lvl.Dinv.dtype, jnp.float32)
                if lvl.P is not None:
                    assert_equal(lvl.P.dtype, jnp.float32)
                if lvl.R is not None:
                    assert_equal(lvl.R.dtype, jnp.float32)

            coarse = ml.levels[-1]
            if coarse.A_dense is not None:
                assert_equal(coarse.A_dense.dtype, jnp.float32)
            for attr in coarse_attrs:
                assert_equal(getattr(coarse, attr).dtype, jnp.float32)

    def test_dtype_explicit(self):
        import pyamg

        A_scipy = poisson((8, 8), format='csr')  # f64 par défaut

        for coarse_solver, coarse_attrs in [
            ('pinv', ['A_inv']),
            ('lu', ['lu_factor']),
            ('qr', ['Q', 'R_mat']),
            ('jacobi', []),
        ]:
            ml = MultilevelSolver.from_pyamg(
                pyamg.smoothed_aggregation_solver(A_scipy),
                coarse_solver=coarse_solver,
                dtype=jnp.float32,
            )

            for lvl in ml.levels:
                assert_equal(lvl.A.dtype, jnp.float32)
                assert_equal(lvl.Dinv.dtype, jnp.float32)
                if lvl.P is not None:
                    assert_equal(lvl.P.dtype, jnp.float32)
                if lvl.R is not None:
                    assert_equal(lvl.R.dtype, jnp.float32)

            coarse = ml.levels[-1]
            if coarse.A_dense is not None:
                assert_equal(coarse.A_dense.dtype, jnp.float32)
            for attr in coarse_attrs:
                assert_equal(getattr(coarse, attr).dtype, jnp.float32)

            b = jnp.ones(A_scipy.shape[0], dtype=jnp.float32)
            x = ml.solve(b, tol=1e-4, maxiter=20)
            assert_equal(x.dtype, jnp.float32)

    def test_aspreconditioner_dtype(self):
        import pyamg

        A_scipy = poisson((8, 8), format='csr').astype(np.float32)
        ml = MultilevelSolver.from_pyamg(
            pyamg.smoothed_aggregation_solver(A_scipy, coarse_solver='jacobi')
        )
        M = ml.aspreconditioner()

        b = jnp.ones(A_scipy.shape[0], dtype=jnp.float32)
        assert_equal(jax.eval_shape(M, b).dtype, b.dtype)
        assert_equal(M(b).dtype, b.dtype)

    def test_coarse_solver_dtype(self):
        for dtype in [jnp.float32, jnp.float64]:
            diag_vals = jnp.array(np.arange(1, 5, dtype=np.float64), dtype=dtype)
            A = jsparse.BCOO.fromdense(jnp.diag(diag_vals))
            b = diag_vals

            for solver, kwargs in [
                ('pinv', {}),
                ('lu', {}),
                ('qr', {}),
                ('jacobi', {'iterations': 1, 'omega': 1.0}),
            ]:
                lvl = MultilevelSolver.Level()
                lvl.A = A
                lvl.Dinv = inverse_diagonal(A)

                if solver in ('pinv', 'lu', 'qr'):
                    lvl.A_dense = A.todense()
                    if solver == 'pinv':
                        lvl.A_inv = jnp.linalg.pinv(lvl.A_dense)
                    elif solver == 'lu':
                        lvl.lu_factor, lvl.piv = jax.scipy.linalg.lu_factor(lvl.A_dense)
                    elif solver == 'qr':
                        lvl.Q, lvl.R_mat = jnp.linalg.qr(lvl.A_dense)

                cs = coarse_grid_solver(solver, lvl, **kwargs)
                x = cs(A, jnp.zeros_like(b), b)
                assert_almost_equal(np.array(A @ x), np.array(b), decimal=5)
                assert_equal(x.dtype, dtype)
