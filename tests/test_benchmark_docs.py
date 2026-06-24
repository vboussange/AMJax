import json

from benchmarks.update_benchmark_docs import (
    DOCS_BEGIN_MARKER,
    DOCS_END_MARKER,
    README_BEGIN_MARKER,
    README_END_MARKER,
    build_summary,
    main,
    render_docs_markdown,
    render_readme_markdown,
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
    dtype="f64",
    solver="smoothed_aggregation",
):
    payload = {
        "config": {
            "solver": solver,
            "coarse_solver": "pinv",
            "dtype": dtype,
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
        "n_iter": 13,
    }
    path.write_text(json.dumps(payload))


def _write_complete_results(results_dir, *, grid_size=100):
    for mode in ("single", "vmap"):
        _write_result(
            results_dir / f"pyamg_{mode}_cpu.json",
            method="pyamg",
            mode=mode,
            device="cpu",
            time=10.0,
            grid_size=grid_size,
        )
        _write_result(
            results_dir / f"pyamg_pcg_{mode}_cpu.json",
            method="pyamg_pcg",
            mode=mode,
            device="cpu",
            time=8.0,
            grid_size=grid_size,
        )
        _write_result(
            results_dir / f"amjax_{mode}_gpu.json",
            method="amjax",
            mode=mode,
            device="gpu",
            time=1.0,
            grid_size=grid_size,
        )
        _write_result(
            results_dir / f"amjax_pcg_{mode}_gpu.json",
            method="amjax_pcg",
            mode=mode,
            device="gpu",
            time=0.5,
            grid_size=grid_size,
        )
        _write_result(
            results_dir / f"amjax_{mode}_gpu_f32.json",
            method="amjax",
            mode=mode,
            device="gpu",
            time=0.25,
            grid_size=grid_size,
            dtype="f32",
            residual=1e-4,
        )
        _write_result(
            results_dir / f"amjax_pcg_{mode}_gpu_f32.json",
            method="amjax_pcg",
            mode=mode,
            device="gpu",
            time=0.2,
            grid_size=grid_size,
            dtype="f32",
            residual=1e-4,
        )


def test_build_summary_computes_concrete_speedups(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_complete_results(results_dir)

    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")

    single_amjax = summary["headline_rows"][0]
    single_pcg = summary["headline_rows"][1]
    assert summary["schema_version"] == 2
    assert single_amjax["speedup"] == 10.0
    assert single_pcg["speedup"] == 16.0
    assert summary["speedup_tables"]["headline_single_gpu"][1][
        "amjax_f32_over_f64"
    ] == 4.0
    assert summary["recommendations"]["poisson_default"] == "AMJax + PCG"


def test_render_markdown_is_deterministic(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_complete_results(results_dir)
    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")

    docs = render_docs_markdown(summary)
    readme = render_readme_markdown(summary)

    assert "## Speedup Ratios" in docs
    assert "Headline Smoothed Aggregation Numbers" in docs
    assert "PyAMG / AMJax f64" in docs
    assert "Benchmark slice: solve $A X = B$" in readme
    assert "$X, B \\in \\mathbb{R}^{N \\times k}$" in readme
    assert "NVIDIA A100 80GB" in readme
    assert "16.0x" in readme
    assert "the report" not in readme.lower()
    assert "the report" not in docs.lower()


def test_check_mode_from_summary(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _write_complete_results(results_dir)
    docs_path = tmp_path / "benchmarks.md"
    readme_path = tmp_path / "README.md"
    summary_path = tmp_path / "latest_summary.json"
    docs_path.write_text(
        "# Benchmarks\n\n"
        f"{DOCS_BEGIN_MARKER}\n"
        "placeholder\n"
        f"{DOCS_END_MARKER}\n"
    )
    readme_path.write_text(
        "# README\n\n"
        f"{README_BEGIN_MARKER}\n"
        "placeholder\n"
        f"{README_END_MARKER}\n"
    )
    summary = build_summary(results_dir, generated_at="2026-06-24T00:00:00+00:00")
    write_outputs(
        summary,
        summary_path=summary_path,
        docs_path=docs_path,
        readme_path=readme_path,
    )

    assert (
        main(
            [
                "--summary",
                str(summary_path),
                "--docs-path",
                str(docs_path),
                "--readme-path",
                str(readme_path),
                "--check",
            ]
        )
        == 0
    )


def test_replace_generated_block_requires_markers():
    try:
        replace_generated_block("no markers", "generated", DOCS_BEGIN_MARKER, DOCS_END_MARKER)
    except RuntimeError as exc:
        assert "markers" in str(exc)
    else:
        raise AssertionError("Expected missing marker error")
