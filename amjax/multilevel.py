"""JAX-compatible algebraic multigrid solver."""

from warnings import warn

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
from pyamg.multilevel import MultilevelSolver as PyAMGSolver

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
        self.A    = None
        self.Dinv = None
        self.P    = None
        self.R    = None
        self.presmoother       = None
        self.postsmoother      = None
        self._presmoother_spec  = None
        self._postsmoother_spec = None


class MultilevelSolver(PyAMGSolver):
    """JAX-compatible algebraic multigrid solver.

    Extends ``pyamg.multilevel.MultilevelSolver`` with a JAX-traceable V-cycle
    and a pytree registration that allows ``jax.jit`` and ``jax.grad`` to treat
    the hierarchy as a first-class JAX value.

    Parameters
    ----------
    levels : list of _Level
        Multigrid hierarchy, finest level first, coarsest level last.
    coarse_solver_fn : callable
        Coarse-grid solver ``(A, x, b) -> x``.
    coarse_solver_name : str
        Name of the coarse solver, stored for pytree serialisation.
    coarse_solver_kwargs : dict, optional
        Keyword arguments forwarded to the coarse solver.

    Attributes
    ----------
    levels : list of _Level
        Multigrid hierarchy.
    coarse_solver : callable
        Coarse-grid solver.
    symmetric_smoothing : bool
        True when pre- and post-smoothers are identical and symmetric,
        allowing conjugate-gradient compatibility.

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
        coarse_solver="jacobi",
        coarse_solver_kwargs=None,
    ):
        """Construct a JAX multigrid hierarchy from a PyAMG hierarchy.

        Parameters
        ----------
        pyamg_solver : pyamg.multilevel.MultilevelSolver
            Hierarchy built by a PyAMG solver factory (e.g. ``ruge_stuben_solver``).
        presmoother : str, tuple, list, or None
            Pre-smoother specification.  See ``change_smoothers`` for accepted forms.
        postsmoother : str, tuple, list, or None
            Post-smoother specification.
        coarse_solver : str
            Name of the coarse-grid solver.  Currently only ``'jacobi'`` is supported.
        coarse_solver_kwargs : dict, optional
            Keyword arguments forwarded to the coarse-grid solver.
            Defaults to ``{'iterations': 10, 'omega': 1.0}``.

        Returns
        -------
        MultilevelSolver
            Fully initialised JAX multigrid hierarchy.
        """
        if coarse_solver_kwargs is None:
            coarse_solver_kwargs = {"iterations": 10, "omega": 1.0}

        levels    = _convert_hierarchy(pyamg_solver)
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
        cycle : {'V', 'W'}
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
            Cycle type (currently only ``'V'`` is supported).

        Returns
        -------
        callable
            A ``matvec(b) -> x`` function compatible with
            ``jax.scipy.sparse.linalg.cg`` and ``jax.scipy.sparse.linalg.gmres``.

        See Also
        --------
        MultilevelSolver.solve
        """
        def matvec(b):
            b = jnp.ravel(jnp.asarray(b))
            x = jnp.zeros_like(b)
            return self._cycle(x, b, cycle=cycle)

        return matvec

    def _cycle(self, x, b, cycle="V"):
        """Apply one V-cycle of the multigrid algorithm.

        Python loops are unrolled at JAX compile time, making this
        compatible with ``lax.while_loop``.

        Parameters
        ----------
        x : ndarray
            Current iterate, length n.
        b : ndarray
            Right-hand side, length n.
        cycle : str
            Cycle type (currently only ``'V'`` is supported).

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
            Cycle type (currently only ``'V'`` is supported).

        Returns
        -------
        ndarray
            Approximate solution to Ax = b.
        """
        b = jnp.ravel(jnp.asarray(b))
        x = jnp.zeros_like(b) if x0 is None else jnp.ravel(jnp.asarray(x0))

        A0    = self.levels[0].A
        normb = jnp.linalg.norm(b)
        normb = jnp.where(normb == 0.0, 1.0, normb)  # use absolute tolerance when b = 0
        normr = jnp.linalg.norm(b - A0 @ x)

        def cond(state):
            _, it, normr_ = state
            return (it < maxiter) & (normr_ >= tol * normb)

        def body(state):
            x_, it, _ = state
            x_new  = self._cycle(x_, b, cycle=cycle)
            normr_ = jnp.linalg.norm(b - A0 @ x_new)
            return x_new, it + 1, normr_

        x, _, _ = lax.while_loop(cond, body, (x, jnp.array(0), normr))
        return x


def coarse_grid_solver(solver_name, lvl, **kwargs):
    """Return a coarse-grid solver callable with signature ``(A, x, b) -> x``.

    Parameters
    ----------
    solver_name : str
        Name of the coarse-grid solver.  Currently only ``'jacobi'`` is supported.
    lvl : _Level
        Coarsest level; must have ``A`` and ``Dinv`` attributes set.
    **kwargs
        Additional keyword arguments forwarded to the solver.

    Returns
    -------
    callable
        Coarse-grid solver with signature ``(A, x, b) -> x``.

    Notes
    -----
    The returned callable shares the same signature as the level smoothers so
    that ``_cycle`` can treat them uniformly.
    """
    _REGISTRY = {
        "jacobi": _coarse_jacobi,
    }
    if solver_name not in _REGISTRY:
        raise ValueError(
            f"Unknown coarse solver {solver_name!r}. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[solver_name](lvl.Dinv, **kwargs)


def _coarse_jacobi(Dinv, iterations=10, omega=1.0):
    """Return a Jacobi coarse-grid solver with a pre-computed Dinv."""
    def solve(A, x, b):
        return relaxation.jacobi(A, x, b, Dinv, iterations=iterations, omega=omega)

    solve.__name__ = "jacobi"
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
    leaves = []
    for lvl in ml.levels[:-1]:
        leaves += [lvl.A, lvl.Dinv, lvl.P, lvl.R]
    coarse = ml.levels[-1]
    leaves += [coarse.A, coarse.Dinv]

    aux = {
        "n_levels":            len(ml.levels),
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

    coarse_base     = (n - 1) * 4
    coarse_lvl      = _Level()
    coarse_lvl.A    = leaves[coarse_base]
    coarse_lvl.Dinv = leaves[coarse_base + 1]
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
