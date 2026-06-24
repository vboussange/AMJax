"""Generate committed benchmark artifacts and documentation tables.

This script consumes raw benchmark JSON files written by ``benchmarks/benchmark.ipynb``.
The raw files are intentionally not committed; the compact JSON artifact and generated
Markdown blocks are committed instead.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOCS_BEGIN_MARKER = "<!-- BEGIN GENERATED BENCHMARK RESULTS -->"
DOCS_END_MARKER = "<!-- END GENERATED BENCHMARK RESULTS -->"
README_BEGIN_MARKER = "<!-- BEGIN GENERATED README BENCHMARK -->"
README_END_MARKER = "<!-- END GENERATED README BENCHMARK -->"

SOLVER_LABELS = {
    "ruge_stuben": "Ruge-Stuben",
    "smoothed_aggregation": "Smoothed Aggregation",
    "rootnode": "Root Node",
    "pairwise": "Pairwise",
}

METHOD_LABELS = {
    "amjax": "AMJax",
    "amjax_pcg": "AMJax + PCG",
    "pyamg": "PyAMG",
    "pyamg_pcg": "PyAMG + PCG",
}

DOC_SOLVERS = ("ruge_stuben", "smoothed_aggregation", "rootnode")
DOC_GRIDS = (50, 100, 200, 500)

DEFAULT_SELECTION = {
    "solver": "smoothed_aggregation",
    "cycle_type": "V",
    "coarse_solver": "pinv",
    "smoother": "jacobi",
    "dtype": "f64",
    "tol": 1e-8,
    "maxiter_vcycle": 250,
    "maxiter_solv": 500,
    "vmap_k": 64,
    "grid_size": 500,
}

DYNAMIC_SELECTION_KEYS = ("tol", "maxiter_solv", "vmap_k")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _config_hash(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _normalize_result(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    config = payload.get("config", {})
    time_seconds = _as_float(payload.get("time"))
    if time_seconds is None:
        return None

    normalized = {
        "path": str(path),
        "solver": config.get("solver"),
        "coarse_solver": config.get("coarse_solver"),
        "dtype": config.get("dtype"),
        "tol": _as_float(config.get("tol")),
        "maxiter_vcycle": _as_int(
            config.get("maxiter_vcycle", config.get("maxiter_cycle"))
        ),
        "maxiter_solv": _as_int(config.get("maxiter_solv")),
        "vmap_k": _as_int(config.get("vmap_k")),
        "cycle_type": config.get("cycle_type"),
        "grid_size": _as_int(config.get("grid_size")),
        "method": config.get("method"),
        "smoother": config.get("smoother"),
        "mode": config.get("mode"),
        "device": config.get("device"),
        "time_seconds": time_seconds,
        "relative_residual": _as_float(payload.get("residual")),
        "n_iter": _as_int(payload.get("n_iter")),
    }

    required = [
        "solver",
        "coarse_solver",
        "dtype",
        "maxiter_vcycle",
        "maxiter_solv",
        "vmap_k",
        "cycle_type",
        "grid_size",
        "method",
        "smoother",
        "mode",
        "device",
    ]
    if any(normalized[key] is None for key in required):
        return None
    return normalized


def load_results(results_dir: Path) -> tuple[list[dict[str, Any]], int]:
    """Load valid benchmark result files from ``results_dir``."""
    files = sorted(results_dir.glob("*.json"))
    results = []
    for path in files:
        result = _normalize_result(path)
        if result is not None:
            results.append(result)
    return results, len(files)


def _same_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=0.0)


def _matches_selection(result: dict[str, Any], selection: dict[str, Any]) -> bool:
    for key, expected in selection.items():
        actual = result.get(key)
        if isinstance(expected, float):
            if not _same_float(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def _matches_partial_selection(
    result: dict[str, Any],
    selection: dict[str, Any],
    *,
    skip: set[str],
) -> bool:
    return _matches_selection(
        result,
        {key: value for key, value in selection.items() if key not in skip},
    )


def resolve_selection(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Resolve run-dependent fields from available results."""
    selection = copy.deepcopy(DEFAULT_SELECTION)
    dynamic = set(DYNAMIC_SELECTION_KEYS)

    for key in DYNAMIC_SELECTION_KEYS:
        candidates = {
            result[key]
            for result in results
            if result.get(key) is not None
            and _matches_partial_selection(
                result,
                selection,
                skip=dynamic | {"maxiter_vcycle", "grid_size", "solver", "dtype"},
            )
        }
        if not candidates:
            continue
        preferred = DEFAULT_SELECTION[key]
        if preferred in candidates:
            selection[key] = preferred
        elif key == "tol":
            selection[key] = min(candidates)
        else:
            selection[key] = max(candidates)

    selection["grid_size"] = _largest_grid_for_headline(results, selection)
    return selection


def _base_match(result: dict[str, Any], selection: dict[str, Any]) -> bool:
    expected = {
        "coarse_solver": selection["coarse_solver"],
        "cycle_type": selection["cycle_type"],
        "smoother": selection["smoother"],
        "tol": selection["tol"],
        "maxiter_solv": selection["maxiter_solv"],
        "vmap_k": selection["vmap_k"],
    }
    return _matches_selection(result, expected)


def _result_sort_key(result: dict[str, Any]) -> tuple[int, int, str]:
    preferred = DEFAULT_SELECTION["maxiter_vcycle"]
    return (
        0 if result["maxiter_vcycle"] == preferred else 1,
        result["maxiter_vcycle"],
        result["path"],
    )


def _find_result(
    results: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    solver: str,
    dtype: str,
    mode: str,
    method: str,
    device: str,
    grid_size: int,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in results
        if _base_match(row, selection)
        and row["solver"] == solver
        and row["dtype"] == dtype
        and row["mode"] == mode
        and row["method"] == method
        and row["device"] == device
        and row["grid_size"] == grid_size
    ]
    if not candidates:
        return None
    return sorted(candidates, key=_result_sort_key)[0]


def _largest_grid_for_headline(
    results: list[dict[str, Any]], selection: dict[str, Any]
) -> int | None:
    grids = []
    for grid_size in DOC_GRIDS:
        required = [
            _find_result(
                results,
                selection,
                solver=selection["solver"],
                dtype=selection["dtype"],
                mode="single",
                method="pyamg",
                device="cpu",
                grid_size=grid_size,
            ),
            _find_result(
                results,
                selection,
                solver=selection["solver"],
                dtype=selection["dtype"],
                mode="single",
                method="amjax",
                device="gpu",
                grid_size=grid_size,
            ),
            _find_result(
                results,
                selection,
                solver=selection["solver"],
                dtype=selection["dtype"],
                mode="single",
                method="pyamg_pcg",
                device="cpu",
                grid_size=grid_size,
            ),
            _find_result(
                results,
                selection,
                solver=selection["solver"],
                dtype=selection["dtype"],
                mode="single",
                method="amjax_pcg",
                device="gpu",
                grid_size=grid_size,
            ),
        ]
        if all(required):
            grids.append(grid_size)
    return max(grids) if grids else DEFAULT_SELECTION["grid_size"]


def _speedup(
    baseline: dict[str, Any] | None, target: dict[str, Any] | None
) -> float | None:
    if baseline is None or target is None:
        return None
    if target["time_seconds"] == 0:
        return None
    return baseline["time_seconds"] / target["time_seconds"]


def _time_ratio(
    numerator: dict[str, Any] | None, denominator: dict[str, Any] | None
) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator["time_seconds"] == 0:
        return None
    return numerator["time_seconds"] / denominator["time_seconds"]


def _row_from_pair(
    *,
    scenario: str,
    method_label: str,
    grid_size: int,
    baseline: dict[str, Any] | None,
    target: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "method": method_label,
        "grid_size": grid_size,
        "unknowns": None if grid_size is None else grid_size * grid_size,
        "baseline_method": None if baseline is None else METHOD_LABELS[baseline["method"]],
        "baseline_device": None if baseline is None else baseline["device"],
        "baseline_time_seconds": None if baseline is None else baseline["time_seconds"],
        "target_device": None if target is None else target["device"],
        "target_time_seconds": None if target is None else target["time_seconds"],
        "speedup": _speedup(baseline, target),
        "relative_residual": None if target is None else target["relative_residual"],
        "n_iter": None if target is None else target["n_iter"],
    }


def _headline_rows(
    results: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    solver = selection["solver"]
    dtype = selection["dtype"]
    grid_size = selection["grid_size"]
    rows = []
    for mode, scenario in (("single", "Single RHS"), ("vmap", "Batched RHS (vmap)")):
        rows.append(
            _row_from_pair(
                scenario=scenario,
                method_label="AMJax",
                grid_size=grid_size,
                baseline=_find_result(
                    results,
                    selection,
                    solver=solver,
                    dtype=dtype,
                    mode=mode,
                    method="pyamg",
                    device="cpu",
                    grid_size=grid_size,
                ),
                target=_find_result(
                    results,
                    selection,
                    solver=solver,
                    dtype=dtype,
                    mode=mode,
                    method="amjax",
                    device="gpu",
                    grid_size=grid_size,
                ),
            )
        )
        rows.append(
            _row_from_pair(
                scenario=scenario,
                method_label="AMJax + PCG",
                grid_size=grid_size,
                baseline=_find_result(
                    results,
                    selection,
                    solver=solver,
                    dtype=dtype,
                    mode=mode,
                    method="pyamg_pcg",
                    device="cpu",
                    grid_size=grid_size,
                ),
                target=_find_result(
                    results,
                    selection,
                    solver=solver,
                    dtype=dtype,
                    mode=mode,
                    method="amjax_pcg",
                    device="gpu",
                    grid_size=grid_size,
                ),
            )
        )
    return rows


def _speedup_rows(
    results: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    solver: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for grid_size in DOC_GRIDS:
        pyamg = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode=mode,
            method="pyamg",
            device="cpu",
            grid_size=grid_size,
        )
        amjax64 = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode=mode,
            method="amjax",
            device="gpu",
            grid_size=grid_size,
        )
        pyamg_pcg = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode=mode,
            method="pyamg_pcg",
            device="cpu",
            grid_size=grid_size,
        )
        amjax_pcg64 = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode=mode,
            method="amjax_pcg",
            device="gpu",
            grid_size=grid_size,
        )
        amjax32 = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f32",
            mode=mode,
            method="amjax",
            device="gpu",
            grid_size=grid_size,
        )
        amjax_pcg32 = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f32",
            mode=mode,
            method="amjax_pcg",
            device="gpu",
            grid_size=grid_size,
        )
        rows.append(
            {
                "grid_size": grid_size,
                "unknowns": grid_size * grid_size,
                "amjax_over_pyamg": _speedup(pyamg, amjax64),
                "amjax_pcg_over_pyamg_pcg": _speedup(pyamg_pcg, amjax_pcg64),
                "amjax_f32_over_f64": _time_ratio(amjax64, amjax32),
                "amjax_pcg_f32_over_f64": _time_ratio(amjax_pcg64, amjax_pcg32),
            }
        )
    return rows


def _residual_rows(
    results: list[dict[str, Any]],
    selection: dict[str, Any],
    *,
    solver: str,
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for grid_size in DOC_GRIDS:
        values = {}
        for dtype in ("f64", "f32"):
            for method in ("amjax", "amjax_pcg"):
                result = _find_result(
                    results,
                    selection,
                    solver=solver,
                    dtype=dtype,
                    mode=mode,
                    method=method,
                    device="gpu",
                    grid_size=grid_size,
                )
                values[f"{method}_{dtype}"] = (
                    None if result is None else result["relative_residual"]
                )
        rows.append(
            {
                "grid_size": grid_size,
                "unknowns": grid_size * grid_size,
                **values,
            }
        )
    return rows


def _hierarchy_rows(
    results: list[dict[str, Any]], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    grid_size = selection["grid_size"]
    for solver in DOC_SOLVERS:
        pyamg = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode="single",
            method="pyamg",
            device="cpu",
            grid_size=grid_size,
        )
        amjax = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode="single",
            method="amjax",
            device="gpu",
            grid_size=grid_size,
        )
        pyamg_pcg = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode="single",
            method="pyamg_pcg",
            device="cpu",
            grid_size=grid_size,
        )
        amjax_pcg = _find_result(
            results,
            selection,
            solver=solver,
            dtype="f64",
            mode="single",
            method="amjax_pcg",
            device="gpu",
            grid_size=grid_size,
        )
        rows.append(
            {
                "solver": solver,
                "solver_label": SOLVER_LABELS[solver],
                "grid_size": grid_size,
                "unknowns": grid_size * grid_size,
                "amjax_speedup": _speedup(pyamg, amjax),
                "amjax_pcg_speedup": _speedup(pyamg_pcg, amjax_pcg),
                "amjax_time_seconds": None if amjax is None else amjax["time_seconds"],
                "amjax_pcg_time_seconds": None
                if amjax_pcg is None
                else amjax_pcg["time_seconds"],
                "amjax_residual": None if amjax is None else amjax["relative_residual"],
                "amjax_pcg_residual": None
                if amjax_pcg is None
                else amjax_pcg["relative_residual"],
                "amjax_n_iter": None if amjax is None else amjax["n_iter"],
                "amjax_pcg_n_iter": None if amjax_pcg is None else amjax_pcg["n_iter"],
                "maxiter_vcycle": None if amjax is None else amjax["maxiter_vcycle"],
            }
        )
    return rows


def _missing_warnings(summary: dict[str, Any]) -> list[str]:
    warnings = []
    for row in summary["headline_rows"]:
        if row["speedup"] is None:
            warnings.append(
                f"Missing README/docs headline pair for {row['scenario']} / {row['method']}."
            )
    for name, rows in summary["speedup_tables"].items():
        for row in rows:
            if row["amjax_over_pyamg"] is None or row["amjax_pcg_over_pyamg_pcg"] is None:
                warnings.append(f"Missing speedup entry in {name} at n={row['grid_size']}.")
    return warnings


def build_summary(
    results_dir: Path,
    *,
    generated_at: str | None = None,
    config_path: Path = Path("benchmarks/config.yaml"),
) -> dict[str, Any]:
    """Build the compact benchmark artifact from raw result JSON files."""
    results, result_file_count = load_results(results_dir)
    if not results:
        raise RuntimeError(f"No valid benchmark results found in {results_dir}.")

    selection = resolve_selection(results)
    summary = {
        "schema_version": 2,
        "generated_at": generated_at or _now_iso(),
        "config_hash": _config_hash(config_path),
        "source": {
            "results_dir": str(results_dir),
            "result_file_count": result_file_count,
        },
        "benchmark": {
            "problem": "2D Poisson equation on an n x n grid",
            "unknowns": "N = n^2",
            "rhs": "Random uniform right-hand side(s), with NumPy seeded to 42 in the notebook",
            "residual_metric": "||b - A x|| / ||b||",
            "timing": "minimum of 10 timed solves after one JAX warm-up call",
            "devices": {
                "amjax_gpu": "NVIDIA A100 80GB",
                "pyamg_cpu": "CPU (unspecified)",
            },
            "timing_excludes": [
                "AMG hierarchy construction",
                "host-to-device transfer",
                "the first JIT compilation call",
            ],
        },
        "selection": selection,
        "recommendations": {
            "poisson_default": "AMJax + PCG",
            "hierarchy": selection["solver"],
            "cycle_type": selection["cycle_type"],
            "coarse_solver": selection["coarse_solver"],
            "smoother": selection["smoother"],
            "dtype_accuracy": "f64",
            "dtype_speed": "f32",
            "batch_size": selection["vmap_k"],
        },
        "headline_rows": _headline_rows(results, selection),
        "speedup_tables": {
            "headline_single_gpu": _speedup_rows(
                results, selection, solver=selection["solver"], mode="single"
            ),
            "headline_vmap_gpu": _speedup_rows(
                results, selection, solver=selection["solver"], mode="vmap"
            ),
        },
        "residual_tables": {
            "headline_single_gpu": _residual_rows(
                results, selection, solver=selection["solver"], mode="single"
            )
        },
        "hierarchy_rows": _hierarchy_rows(results, selection),
    }
    summary["warnings"] = _missing_warnings(summary)
    return summary


def summary_text(summary: dict[str, Any]) -> str:
    """Return the canonical JSON representation for the committed artifact."""
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value >= 100:
        return f"{value:.0f}x"
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def _fmt_residual(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2e}"


def _fmt_time(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 1:
        return f"{value * 1000:.2f} ms"
    if value < 10:
        return f"{value:.3f} s"
    return f"{value:.2f} s"


def _fmt_grid(row: dict[str, Any]) -> str:
    grid = row.get("grid_size")
    unknowns = row.get("unknowns")
    if grid is None:
        return "n/a"
    if unknowns is None:
        return str(grid)
    return f"{grid} ({unknowns:,})"


def _fmt_iter(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def _context_sentence(summary: dict[str, Any]) -> str:
    selection = summary["selection"]
    devices = summary["benchmark"]["devices"]
    vmap_k = selection["vmap_k"]
    return (
        "Benchmark slice: solve $A X = B$, where "
        "$A = A_n \\in \\mathbb{R}^{N \\times N}$ is the 2D five-point Poisson "
        "matrix on an $n \\times n$ grid with $N = n^2$, and "
        f"$X, B \\in \\mathbb{{R}}^{{N \\times m}}$ ($m = 1$ for a single right-hand "
        f"side and $m = {vmap_k}$ for the batched `jax.vmap` rows). Results below "
        f"use `{SOLVER_LABELS[selection['solver']]}`, `{selection['cycle_type']}`-cycle, "
        f"`{selection['coarse_solver']}` coarse solve, `{selection['smoother']}` smoothing, "
        f"`{selection['dtype']}`, tolerance `{selection['tol']:.0e}`, and `k={vmap_k}` "
        f"for batched solves. AMJax runs on GPU ({devices['amjax_gpu']}); PyAMG "
        f"baselines run on {devices['pyamg_cpu']}."
    )


def _headline_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Scenario | Method | Grid n (unknowns) | PyAMG CPU baseline | AMJax GPU time | Speedup | Residual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {scenario} | {method} | {grid} | {baseline} | {target} | {speedup} | {residual} |".format(
                scenario=row["scenario"],
                method=row["method"],
                grid=_fmt_grid(row),
                baseline=_fmt_time(row["baseline_time_seconds"]),
                target=_fmt_time(row["target_time_seconds"]),
                speedup=_fmt_speedup(row["speedup"]),
                residual=_fmt_residual(row["relative_residual"]),
            )
        )
    return lines


def render_readme_markdown(summary: dict[str, Any]) -> str:
    """Render the generated README benchmark block."""
    lines = [
        _context_sentence(summary),
        "",
        *_headline_table(summary["headline_rows"]),
        "",
        (
            "Timings are the minimum of 10 solves after one JAX warm-up call and "
            "exclude hierarchy setup, device transfer, and the first JIT compilation."
        ),
    ]
    if summary.get("warnings"):
        lines.append("")
        lines.append("Incomplete benchmark pairs are shown as `n/a`.")
    return "\n".join(lines).rstrip() + "\n"


def _setup_table(summary: dict[str, Any]) -> list[str]:
    selection = summary["selection"]
    devices = summary["benchmark"]["devices"]
    return [
        "| Quantity | Value |",
        "|---|---|",
        "| Problem | `A_n x = b` with `A_n` the 2D five-point Poisson matrix on an `n x n` grid |",
        "| Unknowns | `N = n^2` |",
        "| Right-hand side | Random uniform vector(s), NumPy seed `42` in the notebook |",
        f"| Grid sizes shown | {', '.join(f'`n={n}`' for n in DOC_GRIDS)} |",
        f"| Tolerance | `{selection['tol']:.0e}` on `||b - A x|| / ||b||` |",
        f"| Cycle / coarse solve / smoother | `{selection['cycle_type']}` / `{selection['coarse_solver']}` / `{selection['smoother']}` |",
        f"| Batch size | `k={selection['vmap_k']}` for `jax.vmap` rows |",
        f"| Devices | AMJax on GPU: {devices['amjax_gpu']}; PyAMG: {devices['pyamg_cpu']} |",
        "| Timing | Minimum of 10 solves after one JAX warm-up call |",
        "| Excluded from timings | Hierarchy setup, device transfer, first JIT compilation |",
    ]


def _speedup_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Grid n (unknowns) | PyAMG / AMJax f64 | PyAMG+PCG / AMJax+PCG f64 | AMJax f64 / f32 | AMJax+PCG f64 / f32 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {grid} | {amg} | {pcg} | {dtype} | {dtype_pcg} |".format(
                grid=_fmt_grid(row),
                amg=_fmt_speedup(row["amjax_over_pyamg"]),
                pcg=_fmt_speedup(row["amjax_pcg_over_pyamg_pcg"]),
                dtype=_fmt_speedup(row["amjax_f32_over_f64"]),
                dtype_pcg=_fmt_speedup(row["amjax_pcg_f32_over_f64"]),
            )
        )
    return lines


def _hierarchy_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Hierarchy | AMJax speedup | AMJax+PCG speedup | AMJax GPU time | AMJax+PCG GPU time | AMJax residual | AMJax+PCG residual | V-cycle iters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {solver} | {amg} | {pcg} | {amg_time} | {pcg_time} | {amg_res} | {pcg_res} | {iters} |".format(
                solver=row["solver_label"],
                amg=_fmt_speedup(row["amjax_speedup"]),
                pcg=_fmt_speedup(row["amjax_pcg_speedup"]),
                amg_time=_fmt_time(row["amjax_time_seconds"]),
                pcg_time=_fmt_time(row["amjax_pcg_time_seconds"]),
                amg_res=_fmt_residual(row["amjax_residual"]),
                pcg_res=_fmt_residual(row["amjax_pcg_residual"]),
                iters=_fmt_iter(row["amjax_n_iter"]),
            )
        )
    return lines


def _residual_table(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Grid n (unknowns) | AMJax f64 | AMJax f32 | AMJax+PCG f64 | AMJax+PCG f32 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {grid} | {amg64} | {amg32} | {pcg64} | {pcg32} |".format(
                grid=_fmt_grid(row),
                amg64=_fmt_residual(row["amjax_f64"]),
                amg32=_fmt_residual(row["amjax_f32"]),
                pcg64=_fmt_residual(row["amjax_pcg_f64"]),
                pcg32=_fmt_residual(row["amjax_pcg_f32"]),
            )
        )
    return lines


def render_docs_markdown(summary: dict[str, Any]) -> str:
    """Render the generated Markdown block for ``docs/benchmarks.md``."""
    recs = summary["recommendations"]
    selection = summary["selection"]

    lines = [
        "## Recommendation",
        "",
        (
            f"For 2D Poisson problems, start with **{recs['poisson_default']}** using "
            f"`{SOLVER_LABELS[recs['hierarchy']]}`, a `{recs['cycle_type']}`-cycle, "
            f"`{recs['coarse_solver']}` coarse solve, and `{recs['smoother']}` smoothing."
        ),
        (
            f"Use `{recs['dtype_accuracy']}` for tight residuals; use "
            f"`{recs['dtype_speed']}` only when speed matters more than final accuracy. "
            f"Batch multiple right-hand sides with `jax.vmap` (`k={recs['batch_size']}` here) "
            "when memory allows."
        ),
        "",
        "## Experimental Setup",
        "",
        *_setup_table(summary),
        "",
        "## Headline Smoothed Aggregation Numbers",
        "",
        _context_sentence(summary),
        "",
        *_headline_table(summary["headline_rows"]),
        "",
        "## Speedup Ratios",
        "",
        (
            "Ratios greater than `1x` mean the method named after the slash is faster. "
            "For example, `PyAMG / AMJax f64 = 20x` means AMJax is 20 times faster than "
            "the PyAMG baseline for that row."
        ),
        "",
        *_speedup_table(
            f"{SOLVER_LABELS[selection['solver']]}, single RHS, AMJax on GPU vs PyAMG on CPU",
            summary["speedup_tables"]["headline_single_gpu"],
        ),
        "",
        *_speedup_table(
            f"{SOLVER_LABELS[selection['solver']]}, batched RHS (`k={selection['vmap_k']}`), AMJax on GPU vs PyAMG loop on CPU",
            summary["speedup_tables"]["headline_vmap_gpu"],
        ),
        "",
        f"### Hierarchy comparison at `n={selection['grid_size']}`",
        "",
        *_hierarchy_table(summary["hierarchy_rows"]),
        "",
        "## Residuals: f32 vs f64",
        "",
        *_residual_table(
            f"{SOLVER_LABELS[selection['solver']]}, single RHS, AMJax on GPU",
            summary["residual_tables"]["headline_single_gpu"],
        ),
        "",
        (
            "Pairwise is not shown in the GPU hierarchy comparison because matching "
            "Pairwise GPU benchmark pairs are not present in the committed benchmark "
            "artifact. Treat Pairwise as a preconditioner option rather than the default "
            "standalone large-system solver."
        ),
    ]

    if summary.get("warnings"):
        lines.extend(["", "!!! warning \"Incomplete benchmark pairs\""])
        for warning in summary["warnings"]:
            lines.append(f"    - {warning}")

    return "\n".join(lines).rstrip() + "\n"


def replace_generated_block(markdown: str, generated: str, begin: str, end: str) -> str:
    """Replace a generated block in a Markdown document."""
    if begin not in markdown or end not in markdown:
        raise RuntimeError("Benchmark generated block markers are missing.")
    before, rest = markdown.split(begin, 1)
    _, after = rest.split(end, 1)
    return f"{before}{begin}\n{generated}{end}{after}"


def write_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    docs_path: Path,
    readme_path: Path,
) -> None:
    summary_path.write_text(summary_text(summary))
    docs = docs_path.read_text()
    docs_path.write_text(
        replace_generated_block(
            docs,
            render_docs_markdown(summary),
            DOCS_BEGIN_MARKER,
            DOCS_END_MARKER,
        )
    )
    readme = readme_path.read_text()
    readme_path.write_text(
        replace_generated_block(
            readme,
            render_readme_markdown(summary),
            README_BEGIN_MARKER,
            README_END_MARKER,
        )
    )


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _compare_file(path: Path, expected: str) -> bool:
    try:
        actual = path.read_text()
    except OSError:
        actual = None
    if actual == expected:
        return True
    print(f"{path} is stale. Regenerate benchmark docs.", file=sys.stderr)
    return False


def check_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path | None,
    docs_path: Path,
    readme_path: Path,
    check_summary_file: bool,
) -> bool:
    ok = True
    if check_summary_file and summary_path is not None:
        ok = _compare_file(summary_path, summary_text(summary)) and ok

    expected_docs = replace_generated_block(
        docs_path.read_text(),
        render_docs_markdown(summary),
        DOCS_BEGIN_MARKER,
        DOCS_END_MARKER,
    )
    ok = _compare_file(docs_path, expected_docs) and ok

    expected_readme = replace_generated_block(
        readme_path.read_text(),
        render_readme_markdown(summary),
        README_BEGIN_MARKER,
        README_END_MARKER,
    )
    ok = _compare_file(readme_path, expected_readme) and ok
    return ok


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path, help="Directory of raw benchmark JSON files.")
    source.add_argument("--summary", type=Path, help="Existing compact benchmark JSON artifact.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write",
        action="store_true",
        help="Write the benchmark artifact and generated Markdown blocks.",
    )
    mode.add_argument("--check", action="store_true", help="Check that generated outputs are fresh.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("benchmarks/latest_summary.json"),
        help="Path for the compact benchmark JSON artifact.",
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=Path("docs/benchmarks.md"),
        help="Path to the benchmark docs page.",
    )
    parser.add_argument(
        "--readme-path",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.write and args.summary is not None:
        print("--write requires --results, not --summary.", file=sys.stderr)
        return 2
    if args.write and args.results is None:
        print("--write requires --results.", file=sys.stderr)
        return 2

    if args.results is not None:
        generated_at = None
        if args.check and args.summary_path.exists():
            generated_at = _load_summary(args.summary_path).get("generated_at")
        summary = build_summary(args.results, generated_at=generated_at)
        check_summary_file = True
    elif args.summary is not None:
        summary = _load_summary(args.summary)
        check_summary_file = False
    elif args.check:
        summary = _load_summary(args.summary_path)
        check_summary_file = False
    else:
        print("Provide --results for --write.", file=sys.stderr)
        return 2

    if args.write:
        write_outputs(
            summary,
            summary_path=args.summary_path,
            docs_path=args.docs_path,
            readme_path=args.readme_path,
        )
        return 0

    ok = check_outputs(
        summary,
        summary_path=args.summary_path,
        docs_path=args.docs_path,
        readme_path=args.readme_path,
        check_summary_file=check_summary_file,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
