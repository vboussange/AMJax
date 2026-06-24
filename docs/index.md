# AMJax

AMJax converts algebraic multigrid hierarchies built by
[PyAMG](https://pyamg.readthedocs.io/) into JAX-compatible solvers and
preconditioners. PyAMG remains responsible for hierarchy construction; AMJax
focuses on repeated solves that can run under `jax.jit`, `jax.vmap`, GPU
execution, and automatic differentiation.

## Installation

```bash
uv add amjax
```

## Where to start

- Use the [quickstart](quickstart.md) for the PyAMG-to-AMJax solve pattern.
- Use [benchmarks](benchmarks.md) for the current recommendation and the latest
  generated benchmark summary.
- Use the [API reference](api.md) for AMJax's small public surface.

## PyAMG Boundary

Use PyAMG documentation for AMG theory, hierarchy builders, and setup options:

- [Classical solvers](https://pyamg.readthedocs.io/en/latest/generated/pyamg.classical.html),
  including `ruge_stuben_solver`.
- [Aggregation solvers](https://pyamg.readthedocs.io/en/latest/generated/pyamg.aggregation.html),
  including smoothed aggregation, root-node, pairwise, and AIR variants.
- [PyAMG reference documentation](https://pyamg.readthedocs.io/) for smoother,
  coarse-solver, and hierarchy construction details.

After `MultilevelSolver.from_pyamg(...)`, AMJax converts the hierarchy into JAX
arrays and sparse `BCOO` matrices. Hierarchy setup still happens in Python, and
cycle, smoother, and coarse-solver choices should be treated as static JAX
configuration.
