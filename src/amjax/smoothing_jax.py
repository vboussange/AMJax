"""Configuration des lisseurs pour le solveur multigrille JAX."""

import src.amjax.relaxation_jax as relaxation

# Nombre d'itérations par défaut
DEFAULT_NITER = 1

# Schémas symétriques -> presmother = postsmoother
SYMMETRIC_RELAXATION = ["jacobi", None]


# ---------------------------------------------------------------------------
# Helpers internes
# ---------------------------------------------------------------------------

def _unpack_arg(v):
    # Un utilisateur peut spécifier un smoother de deux façons,
    # cette fonction normalise les deux formats en (nom, kwargs).
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def _as_list(v):
    """Encapsule une spec lisseur dans une liste si nécessaire."""
    if isinstance(v, list):
        return v
    if isinstance(v, (str, tuple)) or v is None:
        return [v]
    raise ValueError(f"Unrecognized smoother specification: {v!r}")


def _expand_to_n(specs, n):
    """Étendre au bon nombre de niveaux
    """
    # si la hiérarchie a 4 niveaux non-grossiers mais l'utilisateur n'a spécifié qu'un smoother, ça le répète 4x
    if not specs:
        raise ValueError("Empty smoother specification.")
    return (specs + [specs[-1]] * n)[:n]


def _bake_kwargs(kw, smoother):
    """Remplace withrho=True par l'omega concret calculé à la construction.

    Nécessaire pour que rebuild_smoother ne calcule pas le rayon spectral
    pendant le tracing JIT.
    """
    if not kw.get("withrho", False):
        return dict(kw)
    return {**kw, "omega": smoother._omega, "withrho": False}
    # ici on copie les kwargs, on écrase oméga avec la avleur concrète déjà calculée 
    # et on passe withrho à False


# ---------------------------------------------------------------------------
# Fonctions de construction des lisseurs
# ---------------------------------------------------------------------------

def setup_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=False):
    """Construit un lisseur Jacobi amorti."""
    if getattr(lvl, "Dinv", None) is None:
        lvl.Dinv = relaxation.inverse_diagonal(lvl.A)
    Dinv = lvl.Dinv

    if withrho:
        rho = relaxation.approximate_spectral_radius(lvl.A, Dinv)
        omega = omega / rho

    def smoother(A, x, b):
        return relaxation.jacobi(A, x, b, Dinv, iterations=iterations, omega=omega)

    smoother.__name__ = "jacobi"
    smoother._omega = omega   # omega concret, utilisé par _bake_kwargs
    return smoother


def setup_none(lvl, **kwargs):
    """Lisseur identité, ne fait rien
    -> utile pour désactiver le pré/post lissage sur niveau spécifique
    """
    del lvl, kwargs

    def smoother(A, x, b):
        del A, b
        return x

    smoother.__name__ = "none"
    return smoother


# ---------------------------------------------------------------------------
# Registre des lisseurs
# ---------------------------------------------------------------------------

def _setup_call(fn):
    """Retourne la fonction de construction correspondant au nom du lisseur."""
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


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def change_smoothers(ml, presmoother, postsmoother):
    """Assigne les pré- et post-lisseurs à chaque niveau non-grossier de ml.

    Chaque niveau reçoit :
      - lvl.presmoother        : callable (A, x, b) -> x
      - lvl.postsmoother       : callable (A, x, b) -> x
      - lvl._presmoother_spec  : (nom, kwargs) utilisés par rebuild_smoother
      - lvl._postsmoother_spec : (nom, kwargs)

    Met ml.symmetric_smoothing = True si pré == post et les deux sont dans
    SYMMETRIC_RELAXATION (requis pour la compatibilité gradient conjugué).

    Paramètres
    ----------
    ml           : MultilevelSolverJAX
    presmoother  : str, tuple, liste ou None
        (1) nom d'un lisseur, (2) tuple ('méthode', kwargs),
        ou (3) liste de ces options (une par niveau).
    postsmoother : str, tuple, liste ou None — même format que presmoother.
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

        # On stocke l'omega concret (pas withrho=True) pour que rebuild_smoother
        # soit sûr pendant le tracing JIT.
        lvl._presmoother_spec  = (pre_fn,  _bake_kwargs(pre_kw,  lvl.presmoother))
        lvl._postsmoother_spec = (post_fn, _bake_kwargs(post_kw, lvl.postsmoother))

        it1 = pre_kw.get("iterations",  DEFAULT_NITER)
        it2 = post_kw.get("iterations", DEFAULT_NITER)
        if it1 != it2 or pre_fn != post_fn or pre_fn not in SYMMETRIC_RELAXATION:
            ml.symmetric_smoothing = False


def rebuild_smoother(lvl):
    """Reconstruit les callables pré/post-lisseur depuis les specs stockées.

    Appelée après un unflatten JAX pour que les closures capturent
    le bon lvl.Dinv fraîchement reconstruit.
    """
    if getattr(lvl, "_presmoother_spec", None) is None:
        raise AttributeError("Missing _presmoother_spec on level.")
    if getattr(lvl, "_postsmoother_spec", None) is None:
        raise AttributeError("Missing _postsmoother_spec on level.")

    # La version JAX stocke les kwargs explicitement:
    # lvl._presmoother_spec[0], tuple stocké explicitement
    # lvl._presmoother_spec[1], dict stocké explicitement
    fn1, kw1 = lvl._presmoother_spec
    fn2, kw2 = lvl._postsmoother_spec
    lvl.presmoother  = _setup_call(fn1)(lvl, **kw1)
    lvl.postsmoother = _setup_call(fn2)(lvl, **kw2)
