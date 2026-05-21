"""JAX-compatible algebraic multigrid solver."""

from warnings import warn

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
from scipy.linalg import pinv
from pyamg.multilevel import MultilevelSolver as PyAMGMultilevelSolver

from .relaxation import relaxation as relaxation
from .relaxation import smoothing as smoothing


class _Level:
    """One level of the multigrid hierarchy.

    Attributes
    ----------
    A : BCOO sparse matrix
        System matrix on this level.
    Dinv : ndarray
        Element-wise inverse of the diagonal of A.
    P : BCOO sparse matrix or None
        Prolongation operator to the next finer level.  None at the coarsest level.
    R : BCOO sparse matrix or None
        Restriction operator to the next coarser level.  None at the coarsest level.
    presmoother : callable
        Pre-smoother ``(A, x, b) -> x``.
    postsmoother : callable
        Post-smoother ``(A, x, b) -> x``.
    _presmoother_spec : tuple of (str, dict)
        Name and kwargs of the pre-smoother, used by ``rebuild_smoother``.
    _postsmoother_spec : tuple of (str, dict)
        Name and kwargs of the post-smoother, used by ``rebuild_smoother``.

    Notes
    -----
    Defined outside ``MultilevelSolver`` so that ``_flatten`` and
    ``_unflatten`` can reference it without a circular dependency.
    """

    def __init__(self):
        self.A     = None
        self.Dinv  = None
        self.A_inv = None  # precomputed pseudo-inverse, used by pinv coarse solver
        self.P     = None
        self.R     = None
        self.presmoother       = None
        self.postsmoother      = None
        self._presmoother_spec  = None
        self._postsmoother_spec = None


class MultilevelSolver(PyAMGMultilevelSolver):
    """JAX-compatible algebraic multigrid solver.

    Extends ``pyamg.multilevel.MultilevelSolver`` with a fully JAX-compatible
    solve step: JIT-compiled, GPU-accelerated, and registered as a pytree.

    Parameters
    ----------
    levels : list of _Level
        Multigrid hierarchy, finest level first, coarsest level last.
    coarse_solver_fn : callable
        Coarse-grid solver ``(A, x, b) -> x``.
    coarse_solver_name : str
        Name of the coarse solver, stored for pytree serialisation.
    coarse_solver_kwargs : dict, optional
        Keyword arguments forwarded to the coarse-grid solver.
        - ``'pinv'``: ``rcond``
        - ``'jacobi'``: ``iterations``, ``omega``.



    Attributes
    ----------
    levels : list of _Level
        Multigrid hierarchy.
    coarse_solver : callable
        Coarse-grid solver.
    symmetric_smoothing : bool
        True when pre- and post-smoothers are identical and symmetric.

    Methods
    -------
    from_pyamg(pyamg_solver, ...)
        Construct a JAX hierarchy from a PyAMG hierarchy.
    aspreconditioner(cycle='V')
        Return a ``matvec`` callable suitable for use as a JAX preconditioner.
    solve(b, ...)
        Solve Ax = b by repeated multigrid cycles.
    operator_complexity()
        Ratio of total operator nonzeros to finest-level nonzeros.
    cycle_complexity(cycle='V')
        Estimated FLOPs of one cycle relative to the finest level.

    Notes
    -----
    Unlike PyAMG, smoother callables cannot be stored directly inside a
    ``jax.jit``-traced object.  Smoother names and kwargs are stored
    separately so that they can be reconstructed after each pytree unflatten.
    """

    Level = _Level

    def __init__(self, levels, coarse_solver_fn,
                 coarse_solver_name="jacobi", coarse_solver_kwargs=None):
        self.levels                = levels
        self.coarse_solver         = coarse_solver_fn
        self._coarse_solver_name   = coarse_solver_name
        self._coarse_solver_kwargs = coarse_solver_kwargs or {}
        self.symmetric_smoothing   = False

        for lvl in self.levels[:-1]:
            if getattr(lvl, "R", None) is None:
                lvl.R = lvl.P.T

    @classmethod
    def from_pyamg(
        cls,
        pyamg_solver,
        presmoother=("jacobi", {"iterations": 1, "omega": 1.0}),
        postsmoother=("jacobi", {"iterations": 1, "omega": 1.0}),
        coarse_solver="pinv",
        coarse_solver_kwargs=None,
    ):
        """Construct a JAX multigrid hierarchy from a PyAMG hierarchy.

        Parameters
        ----------
        pyamg_solver : pyamg.multilevel.MultilevelSolver
            Hierarchy built by a PyAMG solver factory (e.g. ``ruge_stuben_solver``).
        presmoother : str, tuple, list, or None
            Pre-smoother specification.  
            Passed directly to
                ``change_smoothers``, which applies it to every level of the
                hierarchy. Accepts:
                - a string: ``'jacobi'``
                - a tuple with kwargs: ``('jacobi', {'iterations': 2, 'omega': 0.8})``
                - a list to set a different smoother per level
                - ``None`` to disable smoothing
        postsmoother : str, tuple, list, or None
            Post-smoother specification.
            Same accepted forms as ``presmoother``.
        coarse_solver : str
            Name of the coarse-grid solver. 
            Available: ``'pinv'``, ``'jacobi'``.
            Defaults to ``'pinv'``.
        coarse_solver_kwargs : dict, optional
            Keyword arguments forwarded to the coarse-grid solver.
            Defaults to ``{'iterations': 10, 'omega': 1.0}``.

        Returns
        -------
        MultilevelSolver
            Fully initialised JAX multigrid hierarchy.
        """
        if coarse_solver_kwargs is None:
            coarse_solver_kwargs = {}

        levels = _convert_hierarchy(pyamg_solver)
        if coarse_solver == "pinv":
            A_coarse = levels[-1].A
            levels[-1].A_inv = jnp.linalg.pinv(A_coarse.todense(), **coarse_solver_kwargs)
        coarse_fn = coarse_grid_solver(
            coarse_solver, levels[-1], **coarse_solver_kwargs
        )
        ml = cls(
            levels, coarse_fn,
            coarse_solver_name=coarse_solver,
            coarse_solver_kwargs=coarse_solver_kwargs,
        )
        smoothing.change_smoothers(ml, presmoother, postsmoother)
        return ml

    def __repr__(self):
        """Return a string summary of the multigrid hierarchy."""
        total_nnz = sum(_nnz(lvl.A) for lvl in self.levels)
        out  = "MultilevelSolver\n"
        out += f"Number of Levels:    {len(self.levels)}\n"
        out += f"Operator Complexity: {self.operator_complexity():6.3f}\n"
        out += f"Grid Complexity:     {self.grid_complexity():6.3f}\n"
        out += f"Cycle Complexity:    {self.cycle_complexity():6.3f}\n"
        out += f"Coarse Solver:       {self._coarse_solver_name!r}\n"
        out += "  level   unknowns     nonzeros\n"
        for i, lvl in enumerate(self.levels):
            nnz   = _nnz(lvl.A)
            ratio = 100 * nnz / total_nnz
            out  += f"{i:>6} {lvl.A.shape[0]:>11} {nnz:>12} [{ratio:2.2f}%]\n"
        return out

    def operator_complexity(self):
        """Return the operator complexity of the hierarchy.

        Returns
        -------
        float
            Ratio of the total number of stored nonzeros across all levels to
            the number of nonzeros on the finest level.
        """
        nnz = [_nnz(lvl.A) for lvl in self.levels]
        return sum(nnz) / float(nnz[0])

    def cycle_complexity(self, cycle="V"):
        """Return an estimated FLOP count for one multigrid cycle, normalised by the finest level.

        Parameters
        ----------
        cycle : {'V'}
            Cycle type.

        Returns
        -------
        float
            Estimated work per cycle relative to a single fine-level
            matrix-vector product.
        """
        cycle = str(cycle).upper()
        nnz   = [_nnz(lvl.A) for lvl in self.levels]

        def V(l):
            if len(self.levels) == 1:
                return nnz[0]
            if l == len(self.levels) - 2:
                return 2 * nnz[l] + nnz[l + 1]
            return 2 * nnz[l] + V(l + 1)

        def W(l):
            if len(self.levels) == 1:
                return nnz[0]
            if l == len(self.levels) - 2:
                return 2 * nnz[l] + nnz[l + 1]
            return 2 * nnz[l] + 2 * W(l + 1)

        if cycle == "V":
            flops = V(0)
        elif cycle == "W":
            flops = W(0)
        else:
            raise NotImplementedError(f"Cycle complexity for {cycle!r} not implemented.")

        return float(flops) / float(nnz[0])

    def aspreconditioner(self, cycle='V'):
        """Return a JAX-compatible preconditioner applying one multigrid cycle.

        Parameters
        ----------
        cycle : str
            Cycle type. Available: ``'V'``.

        Returns
        -------
        callable
            A ``matvec(b) -> x`` function compatible with any JAX Krylov
            solver accepting an ``M`` preconditioner. Returns an approximation
            of M⁻¹ b ≈ A⁻¹ b.
        """
        def matvec(b):
            b = jnp.ravel(jnp.asarray(b))
            x = jnp.zeros_like(b)
            return self._cycle(x, b, cycle=cycle)

        return matvec

    def _cycle(self, x, b, cycle="V"):
        """Apply one multigrid cycle.

        Python loops are unrolled at JAX compile time, making this
        compatible with ``lax.while_loop``.

        Parameters
        ----------
        x : ndarray
            Current iterate, length n.
        b : ndarray
            Right-hand side, length n.
        cycle : str
            Cycle type. Available: ``'V'``.

        Returns
        -------
        ndarray
            Updated iterate after one cycle.
        """
        if cycle != "V":
            raise NotImplementedError(f"Cycle {cycle!r} not yet implemented.")

        xs, bs = [], []

        # Fine to coarse 
        for l in range(len(self.levels) - 1):
            lvl = self.levels[l]
            x   = lvl.presmoother(lvl.A, x, b)
            xs.append(x)
            bs.append(b)
            b = lvl.R @ (b - lvl.A @ x)
            x = jnp.zeros_like(b)

        # Coarse-grid solve 
        x = self.coarse_solver(self.levels[-1].A, x, b)

        # Coarse to fine 
        for l in range(len(self.levels) - 2, -1, -1):
            lvl = self.levels[l]
            x   = xs[l] + lvl.P @ x
            x   = lvl.postsmoother(lvl.A, x, bs[l])

        return x

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle="V"):
        """Solve Ax = b by repeated multigrid cycles.

        Compatible with ``jax.jit``::

            solve_jit = jax.jit(MultilevelSolver.solve)
            x = solve_jit(ml, b)

        Parameters
        ----------
        b : ndarray
            Right-hand side, length n.
        x0 : ndarray, optional
            Initial guess.  Defaults to the zero vector.
        tol : float
            Convergence tolerance on the relative residual ``||r|| / ||b||``.
        maxiter : int
            Maximum number of cycles.
        cycle : str
            Cycle type. Available: ``'V'``.

        Returns
        -------
        ndarray
            Approximate solution to Ax = b.
        """
        if cycle != "V":
            raise NotImplementedError(f"Cycle {cycle!r} not yet implemented.")
        b = jnp.ravel(jnp.asarray(b))
        if x0 is not None:
            x0 = jnp.ravel(jnp.asarray(x0))
            b  = b - self.levels[0].A @ x0
        x = _solve_vjp(self, b, jnp.asarray(tol), jnp.asarray(maxiter))
        if x0 is not None:
            x = x + x0
        return x


def _solve(ml, b, tol, maxiter, cycle):
    A0 = ml.levels[0].A
    normb = jnp.linalg.norm(b)
    normb = jnp.where(normb == 0.0, 1.0, normb)
    x = jnp.zeros_like(b)
    normr = jnp.linalg.norm(b - A0 @ x)

    def cond(state):
        _, it, normr_ = state
        return (it < maxiter) & (normr_ >= tol * normb)

    def body(state):
        x_, it, _ = state
        x_new = ml._cycle(x_, b, cycle=cycle)
        normr_ = jnp.linalg.norm(b - A0 @ x_new)
        return x_new, it + 1, normr_

    x, _, _ = lax.while_loop(cond, body, (x, jnp.array(0), normr))
    return x


@jax.custom_vjp
def _solve_vjp(ml, b, tol, maxiter):
    return _solve(ml, b, tol, maxiter, 'V')


def _solve_fwd(ml, b, tol, maxiter):
    x = _solve(ml, b, tol, maxiter, 'V')
    return x, (ml, tol, maxiter)


def _solve_bwd(res, v):
    ml, tol, maxiter = res
    lam = _solve(ml, v, tol, maxiter, 'V')
    return jax.tree_util.tree_map(jnp.zeros_like, ml), lam, jnp.zeros_like(tol), jnp.zeros_like(maxiter)


_solve_vjp.defvjp(_solve_fwd, _solve_bwd)


def coarse_grid_solver(solver_name, lvl, **kwargs):
    """Return a coarse-grid solver callable with signature ``(A, x, b) -> x``.

    Parameters
    ----------
    solver_name : str
        Name of the coarse-grid solver. Available: ``'pinv'``, ``'jacobi'``.
    lvl : _Level
        Coarsest level; must have ``A`` and ``Dinv`` attributes set.
    **kwargs
        Additional keyword arguments forwarded to the solver: 
            - ``'jacobi'``: `iterations`, `omega`
            -  ``'pinv'``: `rcond`


    Returns
    -------
    callable
        Coarse-grid solver with signature ``(A, x, b) -> x``.

    Notes
    -----
    Same signature as level smoothers so ``_cycle`` can call them uniformly.

    """
    if solver_name == "jacobi":
        return _coarse_jacobi(lvl.Dinv, **kwargs)
    elif solver_name == "pinv":
        return _coarse_pinv(lvl.A_inv)
    else:
        raise ValueError(
            f"Unknown coarse solver {solver_name!r}. Available: ['jacobi', 'pinv']"
        )


def _coarse_jacobi(Dinv, iterations=10, omega=1.0):
    """Return a Jacobi coarse-grid solver with a pre-computed Dinv."""
    def solve(A, x, b):
        return relaxation.jacobi(A, x, b, Dinv, iterations=iterations, omega=omega)

    solve.__name__ = "jacobi"
    return solve

def _coarse_pinv(A_inv):
    """Return a coarse-grid solver that applies a precomputed pseudo-inverse."""
    def solve(A, x, b):
        return A_inv @ b

    solve.__name__ = "pinv"
    return solve


def _nnz(A):
    """Return the number of stored entries of a JAX sparse or dense array."""
    if hasattr(A, "nnz"):
        return A.nnz
    if hasattr(A, "nse"):
        return A.nse
    raise TypeError(f"Cannot determine nnz for type {type(A)}")


def _to_jax(M):
    """Convert a SciPy sparse matrix to a JAX BCOO sparse matrix."""
    if isinstance(M, jsparse.BCOO):
        return M
    if hasattr(M, "tocoo"):
        return jsparse.BCOO.from_scipy_sparse(M)
    return jnp.asarray(M)


def _convert_hierarchy(pyamg_solver):
    """Convert a PyAMG multigrid hierarchy to a list of JAX _Level objects.

    Parameters
    ----------
    pyamg_solver : pyamg.multilevel.MultilevelSolver
        Source hierarchy.

    Returns
    -------
    list of _Level
        JAX hierarchy with BCOO matrices and pre-computed diagonal inverses.
    """
    levels = []
    for py_lvl in pyamg_solver.levels:
        lvl      = _Level()
        lvl.A    = _to_jax(py_lvl.A)
        lvl.Dinv = relaxation.inverse_diagonal(lvl.A)
        if hasattr(py_lvl, "P"):
            lvl.P = _to_jax(py_lvl.P)
        if hasattr(py_lvl, "R"):
            lvl.R = _to_jax(py_lvl.R)
        levels.append(lvl)

    for lvl in levels[:-1]:
        if lvl.R is None:
            lvl.R = lvl.P.T.conj()

    return levels


# JAX pytree registration
# Allows jax.jit, jax.grad, etc. to treat MultilevelSolver as a pytree.
# _flatten  : decomposes the hierarchy into JAX array leaves + static metadata.
# _unflatten: reconstructs the hierarchy from leaves + metadata.

def _flatten(ml):
    # Non-coarse levels contribute 4 arrays each: [A, Dinv, P, R].
    # The coarsest level contributes 2 arrays: [A, Dinv].
    """
    Parameters
    ----------
    ml : MultilevelSolver

    Returns
    -------
    leaves : list of JAX arrays
        [A, Dinv, P, R] per non-coarse level, then [A, Dinv] for the coarsest level.
    aux : dict
        Non-array solver configuration metadata: 
            - names
            - kwargs 
            - per-level smoother (name, kwargs) pairs

    Notes
    -----
    JAX cannot trace through Python callables, so smoothers and the coarse solver
    are stored as (name, kwargs) and rebuilt from scratch in _unflatten.
    kwargs is converted to a sorted tuple because JAX uses aux as a JIT cache key,
    which must be hashable.
    """
    leaves = []
    for lvl in ml.levels[:-1]:
        leaves += [lvl.A, lvl.Dinv, lvl.P, lvl.R]
    coarse = ml.levels[-1]
    has_A_inv = coarse.A_inv is not None
    leaves += [coarse.A, coarse.Dinv]
    if has_A_inv:
        leaves.append(coarse.A_inv)

    aux = {
        "n_levels":            len(ml.levels),
        "has_A_inv":           has_A_inv,
        "smoother_specs":      tuple(
            (lvl._presmoother_spec, lvl._postsmoother_spec)
            for lvl in ml.levels[:-1]
        ),
        "symmetric_smoothing": ml.symmetric_smoothing,
        "coarse_solver_name":  ml._coarse_solver_name,
        "coarse_solver_kwargs": tuple(sorted(ml._coarse_solver_kwargs.items())),
    }
    return leaves, aux


def _unflatten(aux, leaves):
    """
    Parameters
    ----------
    aux : dict
        Non-array solver configuration metadata produced by _flatten.
    leaves : list of JAX arrays
        Array leaves produced by _flatten.

    Returns
    -------
    MultilevelSolver
        Fully reconstructed hierarchy with smoother and coarse-solver callables rebuilt.
    """
    n      = aux["n_levels"]
    levels = []

    for i in range(n - 1):
        base     = i * 4
        lvl      = _Level()
        lvl.A    = leaves[base]
        lvl.Dinv = leaves[base + 1]
        lvl.P    = leaves[base + 2]
        lvl.R    = leaves[base + 3]
        levels.append(lvl)

    coarse_base      = (n - 1) * 4
    coarse_lvl       = _Level()
    coarse_lvl.A     = leaves[coarse_base]
    coarse_lvl.Dinv  = leaves[coarse_base + 1]
    if aux["has_A_inv"]:
        coarse_lvl.A_inv = leaves[coarse_base + 2]
    levels.append(coarse_lvl)

    coarse_kw = dict(aux["coarse_solver_kwargs"])
    coarse_fn = coarse_grid_solver(
        aux["coarse_solver_name"], coarse_lvl, **coarse_kw
    )

    ml = MultilevelSolver(
        levels, coarse_fn,
        coarse_solver_name=aux["coarse_solver_name"],
        coarse_solver_kwargs=coarse_kw,
    )
    ml.symmetric_smoothing = aux["symmetric_smoothing"]

    for i, (pre_spec, post_spec) in enumerate(aux["smoother_specs"]):
        lvl = ml.levels[i]
        lvl._presmoother_spec  = pre_spec
        lvl._postsmoother_spec = post_spec
        smoothing.rebuild_smoother(lvl)

    return ml


# Register MultilevelSolver as a JAX pytree so it can be passed to jit, vmap, and grad.
jax.tree_util.register_pytree_node(
    MultilevelSolver,
    _flatten,
    _unflatten,
)


class multilevel_solver(MultilevelSolver):  # noqa: N801
    """Deprecated alias for ``MultilevelSolver``.

    .. deprecated::
        Use ``MultilevelSolver`` instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        warn(
            "multilevel_solver is deprecated. Use MultilevelSolver.",
            DeprecationWarning,
            stacklevel=2,
        )
