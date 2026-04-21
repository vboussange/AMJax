"""Generic AMG solver, compatible with JAX."""

from warnings import warn

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse as jsparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pyamg"))
from pyamg.multilevel import MultilevelSolver

from . import relaxation_jax as relaxation
from . import smoothing_jax as smoothing


# ---------------------------------------------------------------------------
# Classe Level
# ---------------------------------------------------------------------------

class _Level:
    """Un niveau de la hiérarchie multigrille.

    Définie en dehors de MultilevelSolverJAX pour être accessible dans
    _flatten/_unflatten sans dépendance circulaire.

    Attributs
    ---------
    A    : matrice sparse BCOO — système linéaire sur ce niveau
    Dinv : tableau 1D          — inverse de la diagonale de A
    P    : matrice sparse BCOO — prolongation (None au niveau grossier)
    R    : matrice sparse BCOO — restriction  (None au niveau grossier)
    presmoother        : callable (A, x, b) -> x
    postsmoother       : callable (A, x, b) -> x
    _presmoother_spec  : (str, dict) — utilisé par rebuild_smoother
    _postsmoother_spec : (str, dict)
    """

    def __init__(self):
        self.A    = None
        self.Dinv = None
        self.P    = None
        self.R    = None
        # Nom + paramètres de presmoother -> stockés pour pouvoir le reconstruire après un pytree unflatten JAX.
        self.presmoother       = None
        self.postsmoother      = None
        self._presmoother_spec  = None
        self._postsmoother_spec = None


# ---------------------------------------------------------------------------
# Classe solveur
# ---------------------------------------------------------------------------

class MultilevelSolverJAX(MultilevelSolver):
    """Solveur AMG multigrille compatible JAX.

    Paramètres
    ----------
    levels               : liste de _Level
    coarse_solver_fn     : callable (A, x, b) -> x
    coarse_solver_name   : str  — nom du solveur grossier (pour sérialisation pytree)
    coarse_solver_kwargs : dict — arguments du solveur grossier (pour sérialisation pytree)
    """

    Level = _Level  # exposé comme attribut de classe

    def __init__(self, levels, coarse_solver_fn,
                 coarse_solver_name="jacobi", coarse_solver_kwargs=None):
        self.levels                = levels
        self.coarse_solver         = coarse_solver_fn
        self._coarse_solver_name   = coarse_solver_name
        self._coarse_solver_kwargs = coarse_solver_kwargs or {}
        self.symmetric_smoothing   = False  # force change_smoothers to set to True
        # différence avce PyAMG: ici on a besoin de pouvoir retarcer les arguments du solver, de les stocker quelque part
        
        # for ok car code exécuté avant le jit
        for lvl in self.levels[:-1]:
            if getattr(lvl, "R", None) is None:
                lvl.R = lvl.P.T.conj()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pyamg(
        cls,
        pyamg_solver,
        presmoother=("jacobi", {"iterations": 1, "omega": 1.0}),
        postsmoother=("jacobi", {"iterations": 1, "omega": 1.0}),
        coarse_solver="jacobi",
        coarse_solver_kwargs=None,
    ):
        """Construit un solveur JAX à partir d'une hiérarchie PyAMG.
        -> Dans PyAMG,  on stocke directement les fonction sur les niveaux. Dans JAX, 
        jax.jit défait er refait le solveur à chaque compilation (il ne peut pas copier les fonctions).
        -> from_pyamg stocke à la place les noms et paramètres des lisseurs, pour pouvoir les recréer plus tard. 

        Paramètres
        ----------
        pyamg_solver         : pyamg MultilevelSolver
        presmoother          : spec du lisseur, ex. ("jacobi", {"iterations": 2})
        postsmoother         : spec du lisseur
        coarse_solver        : str  — nom du solveur grossier
        coarse_solver_kwargs : dict — arguments du solveur grossier
                               (défaut : {"iterations": 10, "omega": 1.0})
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

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def __repr__(self):
        total_nnz = sum(_nnz(lvl.A) for lvl in self.levels)
        out  = "MultilevelSolverJAX\n"
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

    # grid_complexity héritée de MultilevelSolver (PyAMG)

    def operator_complexity(self):
        """Complexité opérateur : somme des nnz sur tous les niveaux / nnz au niveau fin."""
        nnz = [_nnz(lvl.A) for lvl in self.levels]
        return sum(nnz) / float(nnz[0])

    def cycle_complexity(self, cycle="V"):
        """Estimation des FLOPs d'un cycle multigrille relatifs au niveau fin.

        Paramètres
        ----------
        cycle : {'V', 'W'}
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

    # ------------------------------------------------------------------
    # Préconditionneur
    # ------------------------------------------------------------------

    def aspreconditioner(self, cycle='V'):
        """Préconditionneur JAX : applique un cycle multigrille à un vecteur.

        Retourne un callable matvec(b) -> x compatible jax.jit,
        utilisable comme préconditionneur dans jax.scipy.sparse.linalg.cg ou gmres.

        Paramètres
        ----------
        cycle : str — type de cycle ('V' uniquement pour l'instant)
        """
        def matvec(b):
            b = jnp.ravel(jnp.asarray(b)) # converti b en JAX
            x = jnp.zeros_like(b) # initialise solution à zéro
            return self._cycle(x, b, cycle=cycle) # applique 1 seul cycle V

        return matvec

    # ------------------------------------------------------------------
    # Cycle multigrille — pur JAX
    # ------------------------------------------------------------------

    def _cycle(self, x, b, cycle="V"):
        """Un cycle multigrille V.

        Les boucles Python sont déroulées à la compilation JAX.
        -> compatible avec lax.while_loop.

        Paramètres
        ----------
        x     : itéré courant
        b     : second membre
        cycle : type de cycle ('V' uniquement pour l'instant)
        """
        # Sécurité si on utilise un cycle différent que V
        if cycle != "V":
            raise NotImplementedError(f"Cycle {cycle!r} not yet implemented.")

        xs, bs = [], []

        # --- Du niveau fin vers le niveau grossier ---
        for l in range(len(self.levels) - 1):
            lvl = self.levels[l]
            x   = lvl.presmoother(lvl.A, x, b)
            # JAX retourne un nouveau x, car on ne peut pas modifier les niveuax en place avec JAX
            xs.append(x)
            bs.append(b)
            b = lvl.R @ (b - lvl.A @ x)   # calcul résidu sur niveau courant puis le projette sur le niveau grossier
            x = jnp.zeros_like(b)         # initialise x à 0 sur le niveau le plus grossier

        # --- Résolution au niveau grossier ---
        x = self.coarse_solver(self.levels[-1].A, x, b)

        # --- Du niveau grossier vers le niveau fin ---
        for l in range(len(self.levels) - 2, -1, -1):
            lvl = self.levels[l]
            x   = xs[l] + lvl.P @ x       # coarse-grid correction, on récupère le bon x de l'array xs qu'on a initilisé
            x   = lvl.postsmoother(lvl.A, x, bs[l]) # pareil pour bs ici

        return x

    # ------------------------------------------------------------------
    # Boucle de résolution
    # ------------------------------------------------------------------

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, cycle="V"):
        """Résout Ax = b par cycles multigrilles répétés.

        Compatible avec jax.jit :

            solve_jit = jax.jit(MultilevelSolverJAX.solve)
            x = solve_jit(ml, b)

        Paramètres
        ----------
        b       : second membre
        x0      : estimation initiale (défaut : zéros)
        tol     : tolérance sur le résidu relatif r[k]/||b||
        maxiter : nombre maximum de cycles
        cycle   : type de cycle ('V' uniquement pour l'instant)
        """
        b = jnp.ravel(jnp.asarray(b))
        x = jnp.zeros_like(b) if x0 is None else jnp.ravel(jnp.asarray(x0))

        A0    = self.levels[0].A
        normb = jnp.linalg.norm(b)
        normb = jnp.where(normb == 0.0, 1.0, normb)   # tolérance absolue si b = 0
        normr = jnp.linalg.norm(b - A0 @ x)

        # Condition d'arrêt, on continue si on n'a pas dépassé maxiter et si le résidu relatif est encore trop grand
        def cond(state):
            _, it, normr_ = state
            return (it < maxiter) & (normr_ >= tol * normb)

        # Applique le cycle, calcul le nouveau résidu et retourne nouvel état
        def body(state):
            x_, it, _ = state
            x_new  = self._cycle(x_, b, cycle=cycle)
            normr_ = jnp.linalg.norm(b - A0 @ x_new)
            return x_new, it + 1, normr_

        x, _, _ = lax.while_loop(cond, body, (x, jnp.array(0), normr))
        return x


# ---------------------------------------------------------------------------
# Solveur au niveau grossier
# ---------------------------------------------------------------------------

def coarse_grid_solver(solver_name, lvl, **kwargs):
    """Retourne un solveur grossier callable : (A, x, b) -> x.

    Même interface que les lisseurs pour que _cycle les traite uniformément.
    Pour ajouter un solveur, ajouter une entrée dans _REGISTRY.

    Paramètres
    ----------
    solver_name : str    — nom du solveur (ex. "jacobi")
    lvl         : _Level — niveau le plus grossier (doit avoir A et Dinv)
    **kwargs    : dict   — transmis au solveur
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
    """Solveur grossier Jacobi — Dinv passé explicitement pour éviter un recalcul sous JIT."""
    def solve(A, x, b):
        return relaxation.jacobi(A, x, b, Dinv, iterations=iterations, omega=omega)

    solve.__name__ = "jacobi"
    return solve


# ---------------------------------------------------------------------------
# Conversion de la hiérarchie PyAMG -> JAX
# ---------------------------------------------------------------------------

# Au cas où pour gérer les 2 cas de matrices sparse (c'est une sécurité, normalement que BCOO)
def _nnz(A):
    """Nombre d'entrées stockées dans une matrice sparse ou dense JAX."""
    if hasattr(A, "nnz"):
        return A.nnz
    if hasattr(A, "nse"):
        return A.nse
    raise TypeError(f"Cannot determine nnz for type {type(A)}")


def _to_jax(M):
    """Convertit une matrice scipy sparse en BCOO JAX."""
    if isinstance(M, jsparse.BCOO):
        return M
    if hasattr(M, "tocoo"):
        return jsparse.BCOO.from_scipy_sparse(M)
    return jnp.asarray(M)


def _convert_hierarchy(pyamg_solver):
    """Convertit un MultilevelSolver PyAMG en liste de _Level JAX.

    Paramètres
    ----------
    pyamg_solver : pyamg MultilevelSolver
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


# ---------------------------------------------------------------------------
# Enregistrement du pytree JAX
# ---------------------------------------------------------------------------
#
# Permet à jax.jit, jax.grad, etc. de traiter MultilevelSolverJAX comme un pytree.
#
# _flatten -> décompose en feuilles (tableaux JAX) + métadonnées statiques
# _unflatten -> reconstruit depuis les feuilles + métadonnées

def _flatten(ml):
    # Pour leaves: Chaque niveau de fine level est en 4 tableaux [A, Dinv, P, R]
    #              VS coarse level en 2 tableaux [A, Dinv] sans P et R

    leaves = []
    for lvl in ml.levels[:-1]:           # niveaux non-grossiers
        leaves += [lvl.A, lvl.Dinv, lvl.P, lvl.R]
    coarse = ml.levels[-1]
    leaves += [coarse.A, coarse.Dinv]    # niveau grossier

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

    # Reconstruction des fine levels, 4 arguments: P, R, Dinv, A
    for i in range(n - 1):
        base     = i * 4
        lvl      = _Level()
        lvl.A    = leaves[base]
        lvl.Dinv = leaves[base + 1]
        lvl.P    = leaves[base + 2]
        lvl.R    = leaves[base + 3]
        levels.append(lvl)

    # Reconstruction du corase level, 2 arguments: Dinv, A
    coarse_base     = (n - 1) * 4
    coarse_lvl      = _Level()
    coarse_lvl.A    = leaves[coarse_base]
    coarse_lvl.Dinv = leaves[coarse_base + 1]
    levels.append(coarse_lvl)

    # Reconstruction du solveur grossier
    coarse_kw = dict(aux["coarse_solver_kwargs"])
    coarse_fn = coarse_grid_solver(
        aux["coarse_solver_name"], coarse_lvl, **coarse_kw
    )

    ml = MultilevelSolverJAX(
        levels, coarse_fn,
        coarse_solver_name=aux["coarse_solver_name"],
        coarse_solver_kwargs=coarse_kw,
    )
    ml.symmetric_smoothing = aux["symmetric_smoothing"]

    # Reconstruction des lisseurs
    for i, (pre_spec, post_spec) in enumerate(aux["smoother_specs"]):
        lvl = ml.levels[i]
        lvl._presmoother_spec  = pre_spec
        lvl._postsmoother_spec = post_spec
        smoothing.rebuild_smoother(lvl)

    return ml


jax.tree_util.register_pytree_node(
    MultilevelSolverJAX,
    _flatten,
    _unflatten,
)

# ---------------------------------------------------------------------------
# Alias déprécié
# ---------------------------------------------------------------------------

class multilevel_solver(MultilevelSolverJAX):  # noqa: N801
    """Déprécié : utiliser MultilevelSolverJAX."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            "multilevel_solver is deprecated. Use MultilevelSolverJAX.",
            DeprecationWarning,
            stacklevel=2,
        )
