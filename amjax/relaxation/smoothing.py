"""Smoother setup and configuration for the JAX multigrid solver.

The setup_smoother_name functions are helper functions for
parsing user input and assigning each level the appropriate smoother for
the functions in 'change_smoothers'. They share the same interface:

Parameters
----------
lvl : _Level
    The multigrid level for which to build the smoother.
iterations : int
    Number of smoother iterations
**kwargs
    Method-specific keyword arguments such as ``omega``.

Returns
-------
callable
    Function pointer to the setup function for smoother ``fn``,
    with signature ``(lvl, **kwargs) -> smoother``.

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
    """Freeze the resolved omega into kwargs, replacing ``withrho=True``.

    When ``withrho=True``, omega is divided by the spectral radius of D⁻¹A at
    setup time. This function stores the resulting scalar so that
    ``rebuild_smoother`` can reconstruct the smoother without recomputing the
    spectral radius, which would fail under JAX tracing.

    Parameters
    ----------
    kw : dict
        Raw kwargs as supplied by the user, possibly containing ``withrho=True``.
    smoother : callable
        Already-built smoother carrying the resolved ``_omega`` attribute.

    Returns
    -------
    dict
        kwargs with ``withrho`` removed and ``omega`` set to the concrete scalar.
        Returned unchanged if ``withrho`` is False or absent.
    """
    if not kw.get("withrho", False):
        return dict(kw)
    return {**kw, "omega": smoother._omega, "withrho": False}


def setup_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=False):
    """Set up block Jacobi."""
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
    """Register setup functions.

    This is a helper function to call the setup methods and avoids use of eval().
    """
 
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
    """Initialize pre- and post-smoothers throughout a MultilevelSolver.

    For each level of ``ml`` (except the coarsest level), initialize the
    ``.presmoother()`` and ``.postsmoother()`` methods used in the multigrid cycle,
    with the option of having different smoothers at different levels.

    Parameters
    ----------
    ml : MultilevelSolver
        Data structure that stores the multigrid hierarchy.
    presmoother : str, tuple, list, or None
        presmoother can be (1) the name of a supported smoother, e.g. ``'jacobi'``,
        (2) a tuple of the form ``('method', kwargs_dict)`` where ``'method'`` is
        the name of a supported smoother and ``kwargs_dict`` a dict of keyword
        arguments, or (3) a list of instances of options 1 or 2.

        If presmoother is a list, ``presmoother[i]`` determines the smoothing
        strategy for level i. Else, the same strategy is used for all levels.

        If ``len(presmoother) < len(ml.levels)``, then ``presmoother[-1]``
        is used for all remaining levels.
    postsmoother : str, tuple, list, or None
        Defines postsmoother in identical fashion to presmoother.

    Returns
    -------
    None
        ml is modified in place::

            ml.levels[i].presmoother  <== presmoother[i]
            ml.levels[i].postsmoother <== postsmoother[i]

        ``ml.symmetric_smoothing`` is set to True or False depending on whether
        the smoothing scheme is symmetric.

    Notes
    -----
    - For jacobi method: when ``withrho=True``, ``omega`` is scaled by the spectral
    radius of D⁻¹A on each level, so ``omega`` should lie in (0, 2).

    Available smoothers
    -------------------
        jacobi
        None
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
    """Rebuild the pre/post smoother on a level.

    Parameters
    ----------
    lvl : Level object
        Single level of the hierarchy

    Notes
    -----
    This rebuilds a smoother on level lvl using the existing pre
    and post smoothers.  If different methods are needed, see
    `change_smoothers`.

    """
    if getattr(lvl, "_presmoother_spec", None) is None:
        raise AttributeError("Missing _presmoother_spec on level.")
    if getattr(lvl, "_postsmoother_spec", None) is None:
        raise AttributeError("Missing _postsmoother_spec on level.")

    fn1, kw1 = lvl._presmoother_spec
    fn2, kw2 = lvl._postsmoother_spec
    lvl.presmoother  = _setup_call(fn1)(lvl, **kw1)
    lvl.postsmoother = _setup_call(fn2)(lvl, **kw2)
