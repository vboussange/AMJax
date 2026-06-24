# Benchmarks

AMJax's benchmark notebook runs 2D Poisson problems across solver methods,
hierarchies, precision choices, devices, and batch modes. The tables below are
generated from the latest committed benchmark artifact and should not be edited
by hand.

[Open the benchmark notebook in Colab](https://colab.research.google.com/github/vboussange/AMJax/blob/main/benchmarks/benchmark.ipynb).

To run the full benchmark matrix locally from the repository root:

```bash
benchmarks/run_full_benchmark.sh
```

<!-- BEGIN GENERATED BENCHMARK RESULTS -->
## Recommendation

For 2D Poisson problems, start with **AMJax + PCG** using `Smoothed Aggregation`, a `V`-cycle, `pinv` coarse solve, and `jacobi` smoothing.
Use `f64` for tight residuals; use `f32` only when speed matters more than final accuracy. Batch multiple right-hand sides with `jax.vmap` (`k=64` here) when memory allows.

## Experimental Setup

| Quantity | Value |
|---|---|
| Problem | `A_n x = b` with `A_n` the 2D five-point Poisson matrix on an `n x n` grid |
| Unknowns | `N = n^2` |
| Right-hand side | Random uniform vector(s), NumPy seed `42` in the notebook |
| Grid sizes shown | `n=50`, `n=100`, `n=200`, `n=500` |
| Tolerance | `1e-08` on `||b - A x|| / ||b||` |
| Cycle / coarse solve / smoother | `V` / `pinv` / `jacobi` |
| Batch size | `k=64` for `jax.vmap` rows |
| Devices | AMJax on GPU: NVIDIA A100 80GB; PyAMG on CPU: EPFL cluster node; exact CPU model not recorded in the report |
| Timing | Minimum of 10 solves after one JAX warm-up call |
| Excluded from timings | Hierarchy setup, device transfer, first JIT compilation |

## Headline Smoothed Aggregation Numbers

Benchmark slice: solve `A_n x = b`, where `A_n` is the 2D five-point Poisson matrix on an `n x n` grid (`N = n^2` unknowns). Results below use `Smoothed Aggregation`, `V`-cycle, `pinv` coarse solve, `jacobi` smoothing, `f64`, tolerance `1e-08`, and `k=64` for batched solves. AMJax runs on GPU (NVIDIA A100 80GB); PyAMG baselines run on CPU (EPFL cluster node; exact CPU model not recorded in the report).

| Scenario | Method | Grid n (unknowns) | PyAMG CPU baseline | AMJax GPU time | Speedup | Residual |
|---|---|---:|---:|---:|---:|---:|
| Single RHS | AMJax | 500 (250,000) | 452.63 ms | 14.61 ms | 31.0x | 5.93e-09 |
| Single RHS | AMJax + PCG | 500 (250,000) | 397.33 ms | 7.14 ms | 55.6x | 6.94e-09 |
| Batched RHS (vmap) | AMJax | 500 (250,000) | 29.31 s | 771.17 ms | 38.0x | 5.92e-09 |
| Batched RHS (vmap) | AMJax + PCG | 500 (250,000) | 18.40 s | 295.15 ms | 62.3x | 6.97e-09 |

## Speedup Ratios

Ratios greater than `1x` mean the method named after the slash is faster. For example, `PyAMG / AMJax f64 = 20x` means AMJax is 20 times faster than the PyAMG baseline for that row.

### Smoothed Aggregation, single RHS, AMJax on GPU vs PyAMG on CPU

| Grid n (unknowns) | PyAMG / AMJax f64 | PyAMG+PCG / AMJax+PCG f64 | AMJax f64 / f32 | AMJax+PCG f64 / f32 |
|---|---:|---:|---:|---:|
| 50 (2,500) | 2.43x | 2.51x | 0.16x | 1.39x |
| 100 (10,000) | 4.83x | 4.57x | 0.21x | 1.13x |
| 200 (40,000) | 11.2x | 13.5x | 0.20x | 1.06x |
| 500 (250,000) | 31.0x | 55.6x | 0.22x | 1.15x |

### Smoothed Aggregation, batched RHS (`k=64`), AMJax on GPU vs PyAMG loop on CPU

| Grid n (unknowns) | PyAMG / AMJax f64 | PyAMG+PCG / AMJax+PCG f64 | AMJax f64 / f32 | AMJax+PCG f64 / f32 |
|---|---:|---:|---:|---:|
| 50 (2,500) | 64.0x | 63.2x | 0.22x | 1.18x |
| 100 (10,000) | 84.8x | 80.6x | 0.22x | 1.09x |
| 200 (40,000) | 91.7x | 115x | 0.22x | 1.41x |
| 500 (250,000) | 38.0x | 62.3x | 0.24x | 1.51x |

### Hierarchy comparison at `n=500`

| Hierarchy | AMJax speedup | AMJax+PCG speedup | AMJax GPU time | AMJax+PCG GPU time | AMJax residual | AMJax+PCG residual | V-cycle iters |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ruge-Stuben | 20.3x | 23.5x | 7.21 ms | 6.04 ms | 8.12e-09 | 9.58e-10 | 13 |
| Smoothed Aggregation | 31.0x | 55.6x | 14.61 ms | 7.14 ms | 5.93e-09 | 6.94e-09 | 40 |
| Root Node | 32.2x | 38.1x | 14.31 ms | 7.14 ms | 7.99e-09 | 9.41e-09 | 39 |

## Residuals: f32 vs f64

### Smoothed Aggregation, single RHS, AMJax on GPU

| Grid n (unknowns) | AMJax f64 | AMJax f32 | AMJax+PCG f64 | AMJax+PCG f32 |
|---|---:|---:|---:|---:|
| 50 (2,500) | 8.79e-09 | 1.79e-05 | 2.85e-09 | 3.57e-05 |
| 100 (10,000) | 9.86e-09 | 6.97e-05 | 2.78e-09 | 1.50e-04 |
| 200 (40,000) | 5.92e-09 | 2.73e-04 | 3.80e-09 | 5.42e-04 |
| 500 (250,000) | 5.93e-09 | 1.70e-03 | 6.94e-09 | 3.19e-03 |

Pairwise is not shown in the GPU hierarchy comparison because the committed report-era raw JSON files do not contain matching Pairwise GPU rows. Treat Pairwise as a preconditioner option rather than the default standalone large-system solver.
<!-- END GENERATED BENCHMARK RESULTS -->

The raw benchmark JSON files are intentionally not tracked. Commit the compact
`benchmarks/latest_summary.json` file and the generated Markdown blocks after a
new benchmark run.
