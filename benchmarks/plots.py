"""Plotting utilities for JAX multigrid solver benchmarks."""

import json as _json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


PLOTS_DIR = Path(__file__).parent.parent / "results"


class _Encoder(_json.JSONEncoder):
    def default(self, obj):
        try:
            return obj.tolist()
        except AttributeError:
            pass
        try:
            return float(obj)
        except (TypeError, ValueError):
            return super().default(obj)


def save_results(results, filename):
    """Save benchmark results as JSON in PLOTS_DIR.

    Parameters
    ----------
    results : dict
        Arbitrary dict of benchmark results (lists, nested dicts, scalars).
    filename : str
        Output filename, e.g. ``"solver_benchmark_ruge_stuben.json"``.
    """
    out = PLOTS_DIR / filename
    out.write_text(_json.dumps(results, indent=2, cls=_Encoder))
    print(f"Results saved → {out}")


def load_results(filename):
    """Load benchmark results from a JSON file in PLOTS_DIR.

    Parameters
    ----------
    filename : str
        Filename relative to PLOTS_DIR.

    Returns
    -------
    dict
    """
    path = PLOTS_DIR / filename
    results = _json.loads(path.read_text())
    print(f"Results loaded ← {path}")
    return results
PLOTS_DIR.mkdir(exist_ok=True)


def plot_runtime(sizes, solvers, vmap_k, filename="runtime.png", show=False):
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
    filename : str
        Output filename relative to ``PLOTS_DIR``.
    show : bool
        If True, call ``plt.show()`` instead of closing the figure (for notebooks).
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
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.show() if show else plt.close()


def plot_residual(sizes, solvers, filename="residual.png", show=False):
    """Plot final relative residual against grid size.

    Parameters
    ----------
    sizes : list of int
        Grid sizes (n for an n x n grid).
    solvers : list of tuple
        Each entry is ``(label, residuals)`` where ``residuals`` is a list of
        relative residuals ``||b - Ax|| / ||b||``, one per entry in ``sizes``.
    filename : str
        Output filename relative to ``PLOTS_DIR``.
    show : bool
        If True, call ``plt.show()`` instead of closing the figure (for notebooks).
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
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.show() if show else plt.close()


def plot_memory(sizes, solvers, show=False):
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
    plt.show() if show else plt.close()


def plot_hierarchy_complexity(solver_names, metrics, n, filename="hierarchy_complexity.png", show=False):
    """Bar chart comparing hierarchy complexities and setup costs across AMG solvers.

    Parameters
    ----------
    solver_names : list of str
        Solver labels used as x-axis tick labels.
    metrics : list of dict or None
        One entry per solver.  Each dict must contain ``op_cmpl``, ``grid_cmpl``,
        ``cycle_cmpl``, ``n_levels``, and ``build_s``.  Pass ``None`` for a
        solver that failed to build.
    n : int
        Grid size (n x n) shown in the plot title.
    filename : str
        Output filename relative to ``PLOTS_DIR``.
    show : bool
        If True, call ``plt.show()`` instead of closing the figure (for notebooks).
    """
    valid = [(name, m) for name, m in zip(solver_names, metrics) if m is not None]
    names_v = [name for name, _ in valid]
    x = np.arange(len(names_v))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    ax.bar(x - width, [m["op_cmpl"]    for _, m in valid], width, label="Operator complexity")
    ax.bar(x,         [m["grid_cmpl"]  for _, m in valid], width, label="Grid complexity")
    ax.bar(x + width, [m["cycle_cmpl"] for _, m in valid], width, label="Cycle complexity")
    ax.set_xticks(x)
    ax.set_xticklabels(names_v, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Complexity")
    ax.set_title(f"Hierarchy complexities — {n}x{n} grid")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y")

    ax2  = axes[1]
    ax2r = ax2.twinx()
    ax2.bar(x - width / 2,  [m["n_levels"] for _, m in valid], width,
            label="# levels", color="steelblue")
    ax2r.bar(x + width / 2, [m["build_s"]  for _, m in valid], width,
             label="Setup time (s)", color="orange", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_v, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Number of levels")
    ax2r.set_ylabel("Setup time (s)")
    ax2.set_title("Hierarchy depth & setup cost")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8)
    ax2.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150)
    plt.show() if show else plt.close()


def plot_float_precision(sizes, solvers, show=False):
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
    plt.show() if show else plt.close()
