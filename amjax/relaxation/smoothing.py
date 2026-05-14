"""Smoother setup and configuration for the JAX multigrid solver.

All ``setup_*`` functions share the same interface:

Parameters
----------
lvl : _Level
    The multigrid level for which to build the smoother.
**kwargs
    Method-specific keyword arguments (e.g. ``iterations``, ``omega``).

Returns
-------
callable
    A smoother with signature ``(A, x, b) -> x``.

See Also
--------
change_smoothers : Assign smoothers to all levels of a hierarchy.
"""

from .. import relaxation as relaxation

# Default number of smoothing sweeps per level
DEFAULT_NITER = 1

# Smoothers that satisfy pre == post symmetry for conjugate-gradient compatibility
SYMMETRIC_RELAXATION = ["jacobi", None]


def _unpack_arg(v):
    """Return a ``(name, kwargs)`` pair from a smoother specification."""
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def _as_list(v):
    """Wrap a smoother specification in a list if it is not already one."""
    if isinstance(v, list):
        return v
    if isinstance(v, (str, tuple)) or v is None:
        return [v]
    raise ValueError(f"Unrecognized smoother specification: {v!r}")


def _expand_to_n(specs, n):
    """Repeat the last specification until the list has ``n`` entries."""
    if not specs:
        raise ValueError("Empty smoother specification.")
    return (specs + [specs[-1]] * n)[:n]


def _bake_kwargs(kw, smoother):
    """Replace ``withrho=True`` with the concrete omega resolved at construction time.

    Ensures that ``rebuild_smoother`` never triggers spectral-radius
    estimation during JAX tracing.

    Parameters
    ----------
    kw : dict
        Raw keyword arguments as supplied by the user.
    smoother : callable
        Smoother returned by ``setup_jacobi``, carrying a ``_omega`` attribute.

    Returns
    -------
    dict
        Keyword arguments with ``withrho`` removed and ``omega`` set to the
        concrete scalar computed at setup time.
    """
    if not kw.get("withrho", False):
        return dict(kw)
    return {**kw, "omega": smoother._omega, "withrho": False}


def setup_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=False):
    """Build a damped Jacobi smoother for the given level.

    Parameters
    ----------
    lvl : _Level
        Multigrid level.  ``lvl.Dinv`` is computed here if not already set.
    iterations : int
        Number of Jacobi sweeps per call.
    omega : scalar
        Damping parameter.  When ``withrho=True`` this value is divided by
        the estimated spectral radius of D^{-1} A.
    withrho : bool
        If True, scale ``omega`` by ``1 / rho(D^{-1} A)``.

    Returns
    -------
    callable
        Smoother ``(A, x, b) -> x`` implementing damped Jacobi.
    """
    if getattr(lvl, "Dinv", None) is None:
        lvl.Dinv = relaxation.inverse_diagonal(lvl.A)
    Dinv = lvl.Dinv

    if withrho:
        rho = relaxation.approximate_spectral_radius(lvl.A, Dinv)
        omega = omega / rho

    def smoother(A, x, b):
        return relaxation.jacobi(A, x, b, Dinv, iterations=iterations, omega=omega)

    smoother.__name__ = "jacobi"
    smoother._omega = omega
    return smoother


def setup_none(lvl, **kwargs):
    """Build an identity smoother that leaves x unchanged.

    Returns
    -------
    callable
        Smoother ``(A, x, b) -> x`` that returns x unmodified.
    """
    del lvl, kwargs

    def smoother(A, x, b):
        del A, b
        return x

    smoother.__name__ = "none"
    return smoother


def _setup_call(fn):
    """Return the setup function registered under smoother name ``fn``."""
    _REGISTRY = {
        "jacobi": setup_jacobi,
        "none":   setup_none,
    }

    if fn is None:
        fn = "none"

    if not isinstance(fn, str):
        raise ValueError(f"Smoother name must be a string or None, got {fn!r}")

    if fn not in _REGISTRY:
        raise ValueError(
            f"Unknown smoother {fn!r}. Available: {sorted(_REGISTRY)}"
        )

    return _REGISTRY[fn]


def change_smoothers(ml, presmoother, postsmoother):
    """Assign pre- and post-smoothers to every non-coarse level of ``ml``.

    Each level receives ``presmoother``, ``postsmoother``,
    ``_presmoother_spec``, and ``_postsmoother_spec`` attributes.
    Sets ``ml.symmetric_smoothing = True`` when pre- and post-smoothers are
    identical and listed in ``SYMMETRIC_RELAXATION``.

    Parameters
    ----------
    ml : MultilevelSolver
        Multigrid hierarchy to configure.
    presmoother : str, tuple, list, or None
        Smoother specification.  Accepted forms:

        * ``str``   — smoother name, e.g. ``'jacobi'``
        * ``tuple`` — ``('method', kwargs_dict)``
        * ``list``  — one specification per non-coarse level (last entry is
          repeated if the list is shorter than the number of levels)
        * ``None``  — identity smoother (no relaxation)
    postsmoother : str, tuple, list, or None
        Same format as ``presmoother``.
    """
    n = len(ml.levels[:-1])
    if n == 0:
        return

    pre_specs  = _expand_to_n(_as_list(presmoother),  n)
    post_specs = _expand_to_n(_as_list(postsmoother), n)

    ml.symmetric_smoothing = True

    for i in range(n):
        pre_fn,  pre_kw  = _unpack_arg(pre_specs[i])
        post_fn, post_kw = _unpack_arg(post_specs[i])

        lvl = ml.levels[i]
        lvl.presmoother  = _setup_call(pre_fn)(lvl,  **pre_kw)
        lvl.postsmoother = _setup_call(post_fn)(lvl, **post_kw)

        lvl._presmoother_spec  = (pre_fn,  _bake_kwargs(pre_kw,  lvl.presmoother))
        lvl._postsmoother_spec = (post_fn, _bake_kwargs(post_kw, lvl.postsmoother))

        it1 = pre_kw.get("iterations",  DEFAULT_NITER)
        it2 = post_kw.get("iterations", DEFAULT_NITER)
        if it1 != it2 or pre_fn != post_fn or pre_fn not in SYMMETRIC_RELAXATION:
            ml.symmetric_smoothing = False


def rebuild_smoother(lvl):
    """Reconstruct pre- and post-smoother callables from stored specifications.

    Called after a JAX pytree unflatten so that closures capture the
    freshly reconstructed ``lvl.Dinv``.

    Parameters
    ----------
    lvl : _Level
        A non-coarse level with ``_presmoother_spec`` and
        ``_postsmoother_spec`` attributes set.

    Raises
    ------
    AttributeError
        If ``_presmoother_spec`` or ``_postsmoother_spec`` is missing.
    """
    if getattr(lvl, "_presmoother_spec", None) is None:
        raise AttributeError("Missing _presmoother_spec on level.")
    if getattr(lvl, "_postsmoother_spec", None) is None:
        raise AttributeError("Missing _postsmoother_spec on level.")

    fn1, kw1 = lvl._presmoother_spec
    fn2, kw2 = lvl._postsmoother_spec
    lvl.presmoother  = _setup_call(fn1)(lvl, **kw1)
    lvl.postsmoother = _setup_call(fn2)(lvl, **kw2)
