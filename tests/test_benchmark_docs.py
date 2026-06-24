import json

from benchmarks.update_benchmark_docs import (
    build_summary,
    main,
    render_markdown,
    replace_generated_block,
    write_outputs,
)


def _write_result(
    path,
    *,
    method,
    mode,
    device,
    time,
    residual=1e-9,
    grid_size=100,
    tol=1e-10,
    maxiter_vcycle=100,
):
    payload = {
        "config": {
            "solver": "ruge_stuben",
            "coarse_solver": "pinv",
            "dtype": "f64",
            "tol": tol,
            "maxiter_vcycle": maxiter_vcycle,
            "maxiter_solv": 500,
            "vmap_k": 64,
            "cycle_type": "V",
            "grid_size": grid_size,
            "method": method,
            "smoother": "jacobi",
            "mode": mode,
            "device": device,
        },
        "time": time,
        "residual": residual,
    }
    path.write_text(json.dumps(payload))


def _write_complete_results(results_dir):
    for mode in ("single", "vmap"):
        _write_result(
            results_dir / f"pyamg_{mode}_cpu.json",
            method="pyamg",
            mode=mode,
            device="cpu",
            time=10.0,
        )
        _write_result(
            results_dir / f"pyamg_pcg_{mode}_cpu.json",
            method="pyamg_pcg",
            mode=mode,
            device="cpu",
            time=8.0,
        )
        _write_result(
            results_dir / f"amjax_{mode}_cpu.json",
            method="amjax",
            mode=mode,
            device="cpu",
            time=5.0,
        )
        _write_result(
            results_dir / f"amjax_{mode}_gpu.json",
            method="amjax",
            mode=mode,
            device="gpu",
            time=1.0,
        )
        _write_result(
            results_dir / f"amjax_pcg_{mode}_cpu.json",
            method="amjax_pcg",
            mode=mode,
            device="cpu",
            time=4.0,
        )
        _write_result(
            results_dir / f"amjax_pcg_{mode}_gpu.json",
            method="amjax_pcg",
            mode=mode,
            device="gpu",
            time=0.5,
        )


def test_build_summary_computes_speedups(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_complete_results(results_dir)

    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")

    single_amjax = summary["rows"][0]
    single_pcg = summary["rows"][1]
    assert single_amjax["cpu_speedup"] == 2.0
    assert single_amjax["gpu_speedup"] == 10.0
    assert single_pcg["cpu_speedup"] == 2.0
    assert single_pcg["gpu_speedup"] == 16.0
    assert summary["recommendations"]["poisson_default"] == "AMJax + PCG"


def test_build_summary_accepts_report_era_maxiter_cycle_key(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    payload = {
        "config": {
            "solver": "ruge_stuben",
            "coarse_solver": "pinv",
            "dtype": "f64",
            "tol": 1e-8,
            "maxiter_cycle": 250,
            "maxiter_solv": 500,
            "vmap_k": 64,
            "cycle_type": "V",
            "grid_size": 500,
            "method": "amjax",
            "smoother": "jacobi",
            "mode": "single",
            "device": "gpu",
        },
        "time": 1.0,
        "residual": 1e-9,
    }
    (results_dir / "old.json").write_text(json.dumps(payload))
    baseline = json.loads(json.dumps(payload))
    baseline["config"]["method"] = "pyamg"
    baseline["config"]["device"] = "cpu"
    baseline["time"] = 10.0
    (results_dir / "old_baseline.json").write_text(json.dumps(baseline))

    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")

    assert summary["selection"]["tol"] == 1e-8
    assert summary["selection"]["maxiter_vcycle"] == 250
    assert summary["rows"][0]["gpu_speedup"] == 10.0


def test_render_markdown_is_deterministic(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_complete_results(results_dir)
    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")

    rendered = render_markdown(summary)

    assert "## Recommendations" in rendered
    assert "AMJax + PCG" in rendered
    assert "10.0x" in rendered
    assert "16.0x" in rendered


def test_check_mode_from_summary(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_complete_results(results_dir)
    docs_path = tmp_path / "benchmarks.md"
    summary_path = tmp_path / "latest_summary.json"
    docs_path.write_text(
        "# Benchmarks\n\n"
        "<!-- BEGIN GENERATED BENCHMARK SUMMARY -->\n"
        "placeholder\n"
        "<!-- END GENERATED BENCHMARK SUMMARY -->\n"
    )
    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")
    write_outputs(summary, summary_path=summary_path, docs_path=docs_path)

    assert main(
        [
            "--summary",
            str(summary_path),
            "--docs-path",
            str(docs_path),
            "--check",
        ]
    ) == 0


def test_replace_generated_block_requires_markers():
    try:
        replace_generated_block("no markers", "generated")
    except RuntimeError as exc:
        assert "markers" in str(exc)
    else:
        raise AssertionError("Expected missing marker error")
