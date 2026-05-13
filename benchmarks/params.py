"""Benchmark parameter defaults for the JAX multigrid solver."""

import jax

TOL = 1e-10
MAXITER_VCYCLE = 100
MAXITER_SOLV = 500
GRID_SIZE = [50, 100, 200, 500, 1000]
VMAP_K = 64
IS_CPU = jax.devices()[0].platform == "cpu"

# Solver used in "solver_benchmark.ipynb".
# Must be ONE of: 
#   - smoothed_aggregation, 
#   - rootnode, 
#   - pairwise, 
#   - ruge_stuben,
#   - air
SOLVER = "air"

# Solvers to include in the comparison benchmark "solvers_benchmark.ipynb".
# Comment out any you want to skip.
SOLVERS_COMPARAISON = [
    # "smoothed_aggregation",
    # "rootnode",
    # "pairwise",
    "ruge_stuben",
    # "air",
]
