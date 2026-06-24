"""Generate the committed benchmark summary and docs table.

This script consumes raw benchmark JSON files written by ``benchmarks/benchmark.ipynb``.
The raw files are intentionally not committed; the compact summary JSON and generated
Markdown block are committed instead.
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


BEGIN_MARKER = "<!-- BEGIN GENERATED BENCHMARK SUMMARY -->"
END_MARKER = "<!-- END GENERATED BENCHMARK SUMMARY -->"

DEFAULT_SELECTION = {
    "solver": "ruge_stuben",
    "cycle_type": "V",
    "coarse_solver": "pinv",
    "smoother": "jacobi",
    "dtype": "f64",
    "tol": 1e-8,
    "maxiter_vcycle": 250,
    "maxiter_solv": 500,
    "vmap_k": 64,
}

DYNAMIC_SELECTION_KEYS = ("tol", "maxiter_vcycle", "maxiter_solv")

SCENARIOS = (
    {
        "scenario": "Single RHS",
        "mode": "single",
        "method": "amjax",
        "label": "AMJax",
        "baseline": "pyamg",
    },
    {
        "scenario": "Single RHS",
        "mode": "single",
        "method": "amjax_pcg",
        "label": "AMJax + PCG",
        "baseline": "pyamg_pcg",
    },
    {
        "scenario": "Batched RHS (vmap)",
        "mode": "vmap",
        "method": "amjax",
        "label": "AMJax",
        "baseline": "pyamg",
    },
    {
        "scenario": "Batched RHS (vmap)",
        "mode": "vmap",
        "method": "amjax_pcg",
        "label": "AMJax + PCG",
        "baseline": "pyamg_pcg",
    },
)


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

    maxiter_vcycle = config.get("maxiter_vcycle", config.get("maxiter_cycle"))
    normalized = {
        "path": str(path),
        "solver": config.get("solver"),
        "coarse_solver": config.get("coarse_solver"),
        "dtype": config.get("dtype"),
        "tol": _as_float(config.get("tol")),
        "maxiter_vcycle": _as_int(maxiter_vcycle),
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
    """Resolve run-dependent selection fields from available results.

    The report-era benchmark run used ``tol=1e-8`` and ``maxiter_cycle=250``.
    The current config may use different values. Prefer the documented defaults
    when present, otherwise select the available value from the current run.
    """
    selection = copy.deepcopy(DEFAULT_SELECTION)
    dynamic = set(DYNAMIC_SELECTION_KEYS)

    for key in DYNAMIC_SELECTION_KEYS:
        candidates = {
            result[key]
            for result in results
            if result.get(key) is not None
            and _matches_partial_selection(result, selection, skip=dynamic)
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
    return selection


def _find_result(
    results: list[dict[str, Any]],
    *,
    mode: str,
    method: str,
    device: str,
    grid_size: int,
) -> dict[str, Any] | None:
    for result in results:
        if (
            result["mode"] == mode
            and result["method"] == method
            and result["device"] == device
            and result["grid_size"] == grid_size
        ):
            return result
    return None


def _find_baseline(
    results: list[dict[str, Any]],
    *,
    mode: str,
    method: str,
    grid_size: int,
) -> dict[str, Any] | None:
    baseline = _find_result(
        results, mode=mode, method=method, device="cpu", grid_size=grid_size
    )
    if baseline is not None:
        return baseline
    for result in results:
        if (
            result["mode"] == mode
            and result["method"] == method
            and result["grid_size"] == grid_size
        ):
            return result
    return None


def _speedup(baseline: dict[str, Any] | None, target: dict[str, Any] | None) -> float | None:
    if baseline is None or target is None:
        return None
    if target["time_seconds"] == 0:
        return None
    return baseline["time_seconds"] / target["time_seconds"]


def _scenario_row(
    filtered: list[dict[str, Any]],
    spec: dict[str, str],
) -> tuple[dict[str, Any], list[str]]:
    mode = spec["mode"]
    method = spec["method"]
    baseline_method = spec["baseline"]
    warnings = []

    baseline_grids = {
        row["grid_size"]
        for row in filtered
        if row["mode"] == mode and row["method"] == baseline_method
    }
    target_grids = {
        row["grid_size"] for row in filtered if row["mode"] == mode and row["method"] == method
    }
    common_grids = sorted(baseline_grids & target_grids)

    if not common_grids:
        warnings.append(
            f"Missing benchmark pair for {spec['scenario']} / {spec['label']}."
        )
        return {
            "scenario": spec["scenario"],
            "method": spec["label"],
            "grid_size": None,
            "cpu_speedup": None,
            "gpu_speedup": None,
            "baseline": baseline_method,
            "baseline_time_seconds": None,
            "cpu_time_seconds": None,
            "gpu_time_seconds": None,
            "cpu_relative_residual": None,
            "gpu_relative_residual": None,
        }, warnings

    grid_size = common_grids[-1]
    baseline = _find_baseline(
        filtered, mode=mode, method=baseline_method, grid_size=grid_size
    )
    cpu_target = _find_result(
        filtered, mode=mode, method=method, device="cpu", grid_size=grid_size
    )
    gpu_target = _find_result(
        filtered, mode=mode, method=method, device="gpu", grid_size=grid_size
    )

    if baseline is None:
        warnings.append(
            f"Missing CPU baseline for {spec['scenario']} / {spec['label']} at n={grid_size}."
        )
    if cpu_target is None:
        warnings.append(
            f"Missing CPU AMJax result for {spec['scenario']} / {spec['label']} at n={grid_size}."
        )
    if gpu_target is None:
        warnings.append(
            f"Missing GPU AMJax result for {spec['scenario']} / {spec['label']} at n={grid_size}."
        )

    return {
        "scenario": spec["scenario"],
        "method": spec["label"],
        "grid_size": grid_size,
        "cpu_speedup": _speedup(baseline, cpu_target),
        "gpu_speedup": _speedup(baseline, gpu_target),
        "baseline": baseline_method,
        "baseline_time_seconds": None if baseline is None else baseline["time_seconds"],
        "cpu_time_seconds": None if cpu_target is None else cpu_target["time_seconds"],
        "gpu_time_seconds": None if gpu_target is None else gpu_target["time_seconds"],
        "cpu_relative_residual": None
        if cpu_target is None
        else cpu_target["relative_residual"],
        "gpu_relative_residual": None
        if gpu_target is None
        else gpu_target["relative_residual"],
    }, warnings


def build_summary(
    results_dir: Path,
    *,
    generated_at: str | None = None,
    config_path: Path = Path("benchmarks/config.yaml"),
) -> dict[str, Any]:
    """Build the compact benchmark summary from raw result JSON files."""
    results, result_file_count = load_results(results_dir)
    if not results:
        raise RuntimeError(f"No valid benchmark results found in {results_dir}.")

    selection = resolve_selection(results)
    filtered = [result for result in results if _matches_selection(result, selection)]
    if not filtered:
        raise RuntimeError(
            "No benchmark results match the default documentation selection. "
            "Check solver, cycle, coarse solver, smoother, dtype, and iteration settings."
        )

    rows = []
    warnings = []
    for spec in SCENARIOS:
        row, row_warnings = _scenario_row(filtered, spec)
        rows.append(row)
        warnings.extend(row_warnings)

    return {
        "schema_version": 1,
        "generated_at": generated_at or _now_iso(),
        "config_hash": _config_hash(config_path),
        "source": {
            "results_dir": str(results_dir),
            "result_file_count": result_file_count,
        },
        "selection": selection,
        "recommendations": {
            "poisson_default": "AMJax + PCG",
            "hierarchy": "ruge_stuben",
            "cycle_type": "V",
            "coarse_solver": "pinv",
            "smoother": "jacobi",
            "dtype_accuracy": "f64",
            "dtype_speed": "f32",
            "batch_size": selection["vmap_k"],
        },
        "rows": rows,
        "warnings": warnings,
    }


def summary_text(summary: dict[str, Any]) -> str:
    """Return the canonical JSON representation for the committed summary."""
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


def _fmt_grid(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the generated Markdown block for ``docs/benchmarks.md``."""
    selection = summary["selection"]
    recs = summary["recommendations"]
    rows = summary["rows"]

    lines = [
        "## Recommendations",
        "",
        (
            f"- For 2D Poisson problems, start with **{recs['poisson_default']}** "
            f"using `{recs['hierarchy']}`, a `{recs['cycle_type']}`-cycle, "
            f"`{recs['coarse_solver']}` coarse solve, and `{recs['smoother']}` smoothing."
        ),
        (
            f"- Use `{recs['dtype_accuracy']}` for tight residuals; use "
            f"`{recs['dtype_speed']}` only when speed matters more than final accuracy."
        ),
        (
            "- Batch multiple right-hand sides with `jax.vmap`; the report-backed "
            f"default is `k={recs['batch_size']}` when memory allows."
        ),
        (
            "- Treat Pairwise as a preconditioner option rather than the default "
            "standalone large-system solver."
        ),
        "",
        "## Latest generated summary",
        "",
        (
            f"Generated at `{summary['generated_at']}` from "
            f"{summary['source']['result_file_count']} raw result files."
        ),
        "",
        (
            "Slice: "
            f"`solver={selection['solver']}`, "
            f"`cycle={selection['cycle_type']}`, "
            f"`coarse_solver={selection['coarse_solver']}`, "
            f"`smoother={selection['smoother']}`, "
            f"`dtype={selection['dtype']}`, "
            f"`vmap_k={selection['vmap_k']}`."
        ),
        "",
        "| Scenario | Method | Grid n | CPU speedup | GPU speedup | Residual |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for row in rows:
        residual = row.get("gpu_relative_residual")
        if residual is None:
            residual = row.get("cpu_relative_residual")
        lines.append(
            "| {scenario} | {method} | {grid} | {cpu} | {gpu} | {residual} |".format(
                scenario=row["scenario"],
                method=row["method"],
                grid=_fmt_grid(row["grid_size"]),
                cpu=_fmt_speedup(row["cpu_speedup"]),
                gpu=_fmt_speedup(row["gpu_speedup"]),
                residual=_fmt_residual(residual),
            )
        )

    lines.extend(
        [
            "",
            (
                "Speedups use the PyAMG counterpart on CPU as the baseline. "
                "Benchmark timings exclude hierarchy setup, device transfer, and JIT "
                "compilation warm-up."
            ),
        ]
    )

    if summary.get("warnings"):
        lines.extend(["", "!!! warning \"Incomplete benchmark pairs\""])
        for warning in summary["warnings"]:
            lines.append(f"    - {warning}")

    return "\n".join(lines).rstrip() + "\n"


def replace_generated_block(markdown: str, generated: str) -> str:
    """Replace the generated benchmark block in a Markdown document."""
    if BEGIN_MARKER not in markdown or END_MARKER not in markdown:
        raise RuntimeError("Benchmark generated block markers are missing.")
    before, rest = markdown.split(BEGIN_MARKER, 1)
    _, after = rest.split(END_MARKER, 1)
    return f"{before}{BEGIN_MARKER}\n{generated}{END_MARKER}{after}"


def write_outputs(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    docs_path: Path,
) -> None:
    summary_path.write_text(summary_text(summary))
    docs = docs_path.read_text()
    docs_path.write_text(replace_generated_block(docs, render_markdown(summary)))


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
    check_summary_file: bool,
) -> bool:
    ok = True
    if check_summary_file and summary_path is not None:
        ok = _compare_file(summary_path, summary_text(summary)) and ok
    expected_docs = replace_generated_block(docs_path.read_text(), render_markdown(summary))
    ok = _compare_file(docs_path, expected_docs) and ok
    return ok


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--results", type=Path, help="Directory of raw benchmark JSON files.")
    source.add_argument("--summary", type=Path, help="Existing compact summary JSON.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Write summary JSON and docs block.")
    mode.add_argument("--check", action="store_true", help="Check that generated outputs are fresh.")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("benchmarks/latest_summary.json"),
        help="Path for the compact summary JSON.",
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=Path("docs/benchmarks.md"),
        help="Path to the benchmark docs page.",
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
        write_outputs(summary, summary_path=args.summary_path, docs_path=args.docs_path)
        return 0

    ok = check_outputs(
        summary,
        summary_path=args.summary_path,
        docs_path=args.docs_path,
        check_summary_file=check_summary_file,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
