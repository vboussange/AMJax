import os

USE_CPU = True  # set to False to use GPU if available
if USE_CPU:
    os.environ["JAX_PLATFORMS"] = "cpu"

import tracemalloc
import numpy as np
import jax
import pyamg
import matplotlib.pyplot as plt
import timeit
import jax.numpy as jnp

from amjax import MultilevelSolverJAX
from pyamg.relaxation.smoothing import change_smoothers
from scipy.sparse.linalg import cg
from amjax.plots import plot_runtime, plot_residual, plot_memory, plot_float_precision
from amjax.params import TOL as tol, MAXITER_VCYCLE as maxiter_vcycle, MAXITER_SOLV as maxiter_solv, GRID_SIZE as grid_size, VMAP_K, IS_CPU

jax.config.update("jax_enable_x64", True)
np.random.seed(42)

# Definition of solvers

@jax.jit
def amjax_solve(ml, b):
    x = ml.solve(b, tol=tol, maxiter=maxiter_vcycle)
    error = jnp.linalg.norm(b - ml.levels[0].A @ x) / jnp.linalg.norm(b)
    return x, error

@jax.jit
def amjax_pcg_solve(ml, b):
    M = ml.aspreconditioner()
    x, _ = jax.scipy.sparse.linalg.cg(ml.levels[0].A, b, M=M, tol=tol, maxiter=maxiter_solv)
    error = jnp.linalg.norm(b - ml.levels[0].A @ x) / jnp.linalg.norm(b)
    return x, error

def pyamg_solve(ml, b):
    x = ml.solve(b, tol=tol, maxiter=maxiter_vcycle, cycle='V')
    error = np.linalg.norm(b - ml.levels[0].A @ x) / np.linalg.norm(b)
    return x, error

def pyamg_pcg_solve(ml, b):
    M = ml.aspreconditioner()
    x, _ = cg(ml.levels[0].A, b, M=M, rtol=tol, maxiter=maxiter_solv)
    error = np.linalg.norm(b - ml.levels[0].A @ x) / np.linalg.norm(b)
    return x, error

def cg_solve(A, b):
    x, _ = cg(A, b, rtol=tol, maxiter=maxiter_solv)
    error = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    return x, error

# batch solvers, pour vmap vs boucle for classique

def pyamg_solve_batch(ml, B):
    results = [pyamg_solve(ml, B[i]) for i in range(len(B))]
    return np.array([x for x, _ in results]), float(np.mean([e for _, e in results]))

def pyamg_pcg_solve_batch(ml, B):
    results = [pyamg_pcg_solve(ml, B[i]) for i in range(len(B))]
    return np.array([x for x, _ in results]), float(np.mean([e for _, e in results]))

def cg_solve_batch(A, B):
    results = [cg_solve(A, B[i]) for i in range(len(B))]
    return np.array([x for x, _ in results]), float(np.mean([e for _, e in results]))

def amjax_solve_batch(ml, B):
    results = [amjax_solve(ml, B[i]) for i in range(len(B))]
    return np.array([x for x, _ in results]), float(np.mean([e for _, e in results]))

def amjax_pcg_solve_batch(ml, B):
    results = [amjax_pcg_solve(ml, B[i]) for i in range(len(B))]
    return np.array([x for x, _ in results]), float(np.mean([e for _, e in results]))

def amjax_vmap_solve_batch(ml, B):
    results = jax.vmap(lambda b: amjax_solve(ml, b))(B)
    return results[0], float(jnp.mean(results[1]))

def amjax_pcg_vmap_solve_batch(ml, B):
    results = jax.vmap(lambda b: amjax_pcg_solve(ml, b))(B)
    return results[0], float(jnp.mean(results[1]))

# benchmark function

def benchmark(method, func, is_jax=False):
    if is_jax:
        jax.block_until_ready(func())
        mem_mb = None
    else:
        tracemalloc.start()
        func()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_mb = peak / 1024**2

    times = timeit.repeat(func, number=1, repeat=10)
    _, error = func()
    print(f"{method}/ time: {min(times):.4f}s/ residual: {float(error):.2e}")
    return min(times), error, mem_mb

# float32 casting function
def cast_solver(ml, dtype):
    for lvl in ml.levels:
        lvl.A = lvl.A.astype(dtype)
        lvl.Dinv = lvl.Dinv.astype(dtype)
        if lvl.P is not None: lvl.P = lvl.P.astype(dtype)
        if lvl.R is not None: lvl.R = lvl.R.astype(dtype)
    return ml


# principal bucle of benchmarking

time_pyamg, time_amjax, time_amjax_pcg, time_cg, time_pyamg_pcg, time_amjax_vmap, time_amjax_pcg_vmap = [], [], [], [], [], [], []
res_pyamg, res_amjax, res_amjax_pcg, res_cg, res_pyamg_pcg, res_amjax_vmap, res_amjax_pcg_vmap  = [], [], [], [], [], [], []
res_amjax_f32, res_amjax_pcg_f32 = [], []
mem_pyamg, mem_cg, mem_pyamg_pcg = [], [], []

for n in grid_size:
    print(f"Grid size: {n}x{n}")
    A = pyamg.gallery.poisson((n, n), format="csr")
    B = np.random.rand(VMAP_K, A.shape[0])
    B_jax = jnp.array(B)
    B_f32 = jnp.array(B, dtype=jnp.float32)

    ml = pyamg.ruge_stuben_solver(A, coarse_solver="jacobi")
    change_smoothers(ml, presmoother="jacobi", postsmoother="jacobi")
    ml_jax = MultilevelSolverJAX.from_pyamg(
        ml,
        presmoother=("jacobi", {"iterations": 1, "withrho": True}),
        postsmoother=("jacobi", {"iterations": 1, "withrho": True}),
    )
    ml_f32 = cast_solver(MultilevelSolverJAX.from_pyamg(
        ml,
        presmoother=("jacobi", {"iterations": 1, "withrho": True}),
        postsmoother=("jacobi", {"iterations": 1, "withrho": True}),
    ), jnp.float32)

    time, error, mem = benchmark("PyAMG-RSS", lambda: pyamg_solve_batch(ml, B), is_jax=False)
    time_pyamg.append(time)
    res_pyamg.append(error)
    mem_pyamg.append(mem)

    time, error, mem = benchmark("PyAMG-RSS-PCG", lambda: pyamg_pcg_solve_batch(ml, B), is_jax=False)
    time_pyamg_pcg.append(time)
    res_pyamg_pcg.append(error)
    mem_pyamg_pcg.append(mem)

    time, error, mem = benchmark("CG", lambda: cg_solve_batch(A, B), is_jax=False)
    time_cg.append(time)
    res_cg.append(error)
    mem_cg.append(mem)

    time, error, _ = benchmark("AMJAX-RSS", lambda: amjax_solve_batch(ml_jax, B_jax), is_jax=True)
    time_amjax.append(time)
    res_amjax.append(error)

    time, error, _ = benchmark("AMJAX-RSS-PCG", lambda: amjax_pcg_solve_batch(ml_jax, B_jax), is_jax=True)
    time_amjax_pcg.append(time)
    res_amjax_pcg.append(error)

    time, error, _ = benchmark("AMJAX-RSS vmap", lambda: amjax_vmap_solve_batch(ml_jax, B_jax), is_jax=True)
    time_amjax_vmap.append(time)
    res_amjax_vmap.append(error)

    time, error, _ = benchmark("AMJAX-RSS-PCG vmap", lambda: amjax_pcg_vmap_solve_batch(ml_jax, B_jax), is_jax=True)
    time_amjax_pcg_vmap.append(time)
    res_amjax_pcg_vmap.append(error)

    _, error, _ = benchmark("AMJAX-RSS f32", lambda: amjax_vmap_solve_batch(ml_f32, B_f32), is_jax=True)
    res_amjax_f32.append(error)

    _, error, _ = benchmark("AMJAX-RSS-PCG f32", lambda: amjax_pcg_solve_batch(ml_f32, B_f32), is_jax=True)
    res_amjax_pcg_f32.append(error)


# results table

W = 22
print(f"\n{'=' * 58}")
print(f"  Benchmark Results — batch K={VMAP_K}")
print(f"{'=' * 58}")

solvers = [
    ("PyAMG-RSS",          time_pyamg,         res_pyamg),
    ("PyAMG-RSS-PCG",      time_pyamg_pcg,      res_pyamg_pcg),
    ("CG",                 time_cg,             res_cg),
    ("AMJAX-RSS",          time_amjax,          res_amjax),
    ("AMJAX-RSS-PCG",      time_amjax_pcg,      res_amjax_pcg),
    ("AMJAX-RSS vmap",     time_amjax_vmap,     res_amjax_vmap),
    ("AMJAX-RSS-PCG vmap", time_amjax_pcg_vmap, res_amjax_pcg_vmap),
]

cpu_solvers = [
    ("PyAMG-RSS",     mem_pyamg),
    ("PyAMG-RSS-PCG", mem_pyamg_pcg),
    ("CG",            mem_cg),
]

for i, n in enumerate(grid_size):
    print(f"\n  Grid size : {n} x {n}\n")
    print(f"  {'Solver':<{W}} {'Time (s)':>10}  {'Residual':>12}")
    print(f"  {'-' * (W + 26)}")
    for name, times, residuals in solvers:
        print(f"  {name:<{W}} {times[i]:>10.4f}  {residuals[i]:>12.2e}")

    if IS_CPU:
        print(f"\n  {'Solver':<{W}} {'RAM (MB)':>10}")
        print(f"  {'-' * (W + 12)}")
        for name, mems in cpu_solvers:
            print(f"  {name:<{W}} {mems[i]:>10.2f}")

print(f"\n{'=' * 58}\n")

# plots

plot_runtime(grid_size, [
    ("PyAMG-RSS",          time_pyamg),
    ("PyAMG-RSS-PCG",      time_pyamg_pcg),
    ("AMJAX-RSS vmap",     time_amjax_vmap),
    ("AMJAX-RSS-PCG vmap", time_amjax_pcg_vmap),
    ("CG",                 time_cg),
], VMAP_K)

plot_residual(grid_size, [
    ("PyAMG-RSS",          res_pyamg),
    ("PyAMG-RSS-PCG",      res_pyamg_pcg),
    ("AMJAX-RSS vmap",     res_amjax_vmap),
    ("AMJAX-RSS-PCG vmap", res_amjax_pcg_vmap),
    ("CG",                 res_cg),
])

if IS_CPU:
    plot_memory(grid_size, [
        ("PyAMG-RSS",     mem_pyamg),
        ("PyAMG-RSS-PCG", mem_pyamg_pcg),
        ("CG",            mem_cg),
    ])

plot_float_precision(grid_size, [
    ("AMJAX-RSS", res_amjax_f32, res_amjax),
    ("AMJAX-PCG", res_amjax_pcg_f32, res_amjax_pcg),
])
