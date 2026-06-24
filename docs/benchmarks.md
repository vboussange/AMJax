# Benchmarks

AMJax's benchmark notebook runs 2D Poisson problems across solver methods,
hierarchies, precision choices, devices, and batch modes. The recommendation
below is generated from the latest committed benchmark summary and should not be
edited by hand.

[Open the benchmark notebook in Colab](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/benchmark.ipynb).

To run the full benchmark matrix locally from the repository root:

```bash
benchmarks/run_full_benchmark.sh
```

<!-- BEGIN GENERATED BENCHMARK SUMMARY -->
## Recommendations

- For 2D Poisson problems, start with **AMJax + PCG** using `ruge_stuben`, a `V`-cycle, `pinv` coarse solve, and `jacobi` smoothing.
- Use `f64` for tight residuals; use `f32` only when speed matters more than final accuracy.
- Batch multiple right-hand sides with `jax.vmap`; the report-backed default is `k=64` when memory allows.
- Treat Pairwise as a preconditioner option rather than the default standalone large-system solver.

## Latest generated summary

Generated at `2026-06-24T03:48:57+00:00` from 16 raw result files.

Slice: `solver=ruge_stuben`, `cycle=V`, `coarse_solver=pinv`, `smoother=jacobi`, `dtype=f64`, `vmap_k=64`.

| Scenario | Method | Grid n | CPU speedup | GPU speedup | Residual |
|---|---|---:|---:|---:|---:|
| Single RHS | AMJax | 500 | 0.39x | 20.3x | 8.12e-09 |
| Single RHS | AMJax + PCG | 500 | 0.42x | 23.5x | 9.58e-10 |
| Batched RHS (vmap) | AMJax | 500 | 0.60x | 32.4x | 8.11e-09 |
| Batched RHS (vmap) | AMJax + PCG | 500 | 0.70x | 43.9x | 9.59e-10 |

Speedups use the PyAMG counterpart on CPU as the baseline. Benchmark timings exclude hierarchy setup, device transfer, and JIT compilation warm-up.
<!-- END GENERATED BENCHMARK SUMMARY -->

The raw benchmark JSON files are intentionally not tracked. Commit the compact
`benchmarks/latest_summary.json` file and this generated docs block after a new
benchmark run.
