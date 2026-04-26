"""Plotting utilities for JAX multigrid solver benchmarks."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


PLOTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR.mkdir(exist_ok=True)


def plot_runtime(sizes, solvers, vmap_k):
    """Plot solver wall-clock runtime against grid size.

    Parameters
    ----------
    sizes : list of int
        Grid sizes (n for an n x n grid).
    solvers : list of tuple
        Each entry is ``(label, times)`` where ``times`` is a list of
        runtimes in seconds, one per entry in ``sizes``.
    vmap_k : int
        Batch size shown in the plot title.
    """
    plt.figure(figsize=(6, 4))
    for label, times in solvers:
        plt.plot(sizes, times, marker="o", label=label)
    plt.xlabel("Grid size (n x n)")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.title(f"Solver runtime comparison — batch K={vmap_k}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "runtime.png", dpi=150)
    plt.close()


def plot_residual(sizes, solvers):
    """Plot final relative residual against grid size.

    Parameters
    ----------
    sizes : list of int
        Grid sizes (n for an n x n grid).
    solvers : list of tuple
        Each entry is ``(label, residuals)`` where ``residuals`` is a list of
        relative residuals ``||b - Ax|| / ||b||``, one per entry in ``sizes``.
    """
    plt.figure(figsize=(6, 4))
    for label, residuals in solvers:
        plt.plot(sizes, residuals, marker="o", label=label)
    plt.xlabel("Grid size (n x n)")
    plt.ylabel("Relative residual ||b - Ax|| / ||b||")
    plt.yscale("log")
    plt.title("Solver accuracy comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual.png", dpi=150)
    plt.close()


def plot_memory(sizes, solvers):
    """Plot peak CPU memory usage per solver as a grouped bar chart.

    Parameters
    ----------
    sizes : list of int
        Grid sizes (n for an n x n grid).
    solvers : list of tuple
        Each entry is ``(label, mems)`` where ``mems`` is a list of peak RAM
        values in megabytes, one per entry in ``sizes``.  CPU solvers only.
    """
    n_solvers = len(solvers)
    n_sizes = len(sizes)
    width = 0.8 / n_sizes
    x_pos = np.arange(n_solvers)
    plt.figure(figsize=(6, 4))
    for j, n in enumerate(sizes):
        vals = [mems[j] for _, mems in solvers]
        plt.bar(x_pos + j * width, vals, width, label=f"{n}x{n}")
    plt.xticks(x_pos + width * (n_sizes - 1) / 2, [label for label, _ in solvers])
    plt.ylabel("Peak RAM (MB)")
    plt.title("CPU solvers — tracemalloc peak RAM")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "memory.png", dpi=150)
    plt.close()


def plot_float_precision(sizes, solvers):
    """Plot float32 vs float64 residuals side by side for each solver.

    Parameters
    ----------
    sizes : list of int
        Grid sizes (n for an n x n grid).
    solvers : list of tuple
        Each entry is ``(title, f32_residuals, f64_residuals)`` where
        ``f32_residuals`` and ``f64_residuals`` are lists of relative
        residuals, one per entry in ``sizes``.
    """
    _, axes = plt.subplots(1, len(solvers), figsize=(6 * len(solvers), 4))
    if len(solvers) == 1:
        axes = [axes]
    for ax, (title, f32, f64) in zip(axes, solvers):
        ax.plot(sizes, f32, marker="o", label="float32")
        ax.plot(sizes, f64, marker="o", label="float64")
        ax.set_yscale("log")
        ax.set_xlabel("Grid size (n x n)")
        ax.set_ylabel("Relative residual ||b - Ax|| / ||b||")
        ax.set_title(f"{title}: float32 vs float64")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "float_precision.png", dpi=150)
    plt.close()
