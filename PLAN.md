# Minimal Documentation Migration and Benchmark Update Plan

## Goal

Move AMJax documentation into this repository with a small Zensical site, keep the
content low-maintenance by linking to PyAMG where possible, and make benchmark
numbers in the docs update automatically after a new benchmark run.

The implementation should avoid copying the old Jekyll site, avoid committing
large raw benchmark dumps, and avoid hand-editing benchmark numbers.

## Current State

- The package code lives under `amjax/`.
- The benchmark notebook is `benchmarks/benchmark.ipynb`.
- Benchmark parameters live in `benchmarks/config.yaml`.
- Benchmark result helpers in `benchmarks/plots.py` write one JSON file per
  configuration into the root-level `results/` directory.
- `results/` is not currently tracked.
- `README.md` contains stale hard-coded benchmark numbers and examples that use
  `AMJAXSolver`, while the public entry point is `amjax.MultilevelSolver`.
- There is no in-repo `docs/` directory and no docs deployment workflow.

## Guiding Decisions

- Use Zensical, not MkDocs, Jekyll, or Sphinx.
- Keep docs minimal:
  - AMJax owns installation, quickstart, JAX interoperability, and API surface.
  - PyAMG owns AMG theory, hierarchy factories, smoother/coarse-solver options,
    and algorithm details. AMJax docs should link there instead of duplicating it.
- Use `amjax.MultilevelSolver` as the canonical documented entry point.
- Publish docs at `https://vboussange.github.io/AMJax/`.
- Treat benchmark numbers as generated content:
  - Raw benchmark JSON files are inputs.
  - A compact committed summary artifact is the stable source used by docs.
  - Markdown benchmark tables are generated from that summary.
- Use Fanny Missillier's report as the source for qualitative recommendation
  copy, but avoid freezing report-era numerical values by hand when the same
  values can be regenerated from benchmark output.

## Implementation Checklist

- [x] Work in a new branch: `codex/docs-benchmark-automation`.
- [x] Stage 1: Added a minimal Zensical documentation site.
- [x] Stage 2: Added documentation dependencies and updated the documentation
  project URL.
- [x] Stage 3: Updated README examples, recommendation, limitations, and
  roadmap.
- [x] Stage 4: Added a generated compact benchmark summary contract and
  artifact at `benchmarks/latest_summary.json`.
- [x] Stage 5: Added `benchmarks/update_benchmark_docs.py` with `--write` and
  `--check` modes.
- [x] Stage 6: Wired the benchmark notebook to refresh benchmark docs after a
  benchmark run.
- [x] Stage 6 CLI wrapper: added `benchmarks/run_full_benchmark.sh` to execute
  the notebook with the full config and refresh generated benchmark docs.
- [x] Stage 7: Added the GitHub Pages docs workflow.
- [x] Stage 8: Added lightweight validation tests for the benchmark docs
  generator.
- [x] Stage 9: Generated the first summary/docs block from report-era raw
  benchmark results from the old docs repository.
- [ ] Stage 9 full notebook rerun: not completed; no full local/GPU benchmark
  rerun was performed.
- [ ] Stage 9 commit step: not completed; changes are left unstaged and
  uncommitted for review.
- [ ] Stage 10 optional future improvements: intentionally not implemented.
- [x] Acceptance: `uv run --group docs python -m zensical build --strict` succeeds.
- [x] Acceptance: `uv run python -m pytest tests/ -v` succeeds.
- [x] Acceptance: generated benchmark docs are fresh relative to
  `benchmarks/latest_summary.json`.
- [ ] Acceptance: GitHub Pages deployment on `main` not observed yet; this can
  only complete after merge/push with Pages configured to GitHub Actions.

## Stage 1: Add Minimal Zensical Documentation

### Files to add

- `zensical.toml`
- `docs/index.md`
- `docs/quickstart.md`
- `docs/api.md`
- `docs/benchmarks.md`

### `zensical.toml`

Configure:

- `[project].site_name = "AMJax"`
- `[project].site_url = "https://vboussange.github.io/AMJax/"`
- `[project].repo_url = "https://github.com/vboussange/AMJax"`
- `[project].docs_dir = "docs"`
- `[project].site_dir = "site"`
- strict builds via `uv run --group docs python -m zensical build --strict`
- `project.theme.features` includes `navigation.sections` and `content.code.copy`
- `project.plugins.mkdocstrings` with the Python handler
- `[project].nav`:
  - Overview: `index.md`
  - Quickstart: `quickstart.md`
  - API reference: `api.md`
  - Benchmarks: `benchmarks.md`

Keep configuration conservative. Do not add custom CSS, custom JS, Chart.js,
large generated JSON, notebooks-as-pages, or old Jekyll layouts.

### Page content

- `docs/index.md`
  - One-paragraph description of AMJax.
  - Installation command.
  - Link to quickstart.
  - Link to PyAMG classical, aggregation, and reference documentation.
  - Concise AMJax-specific compatibility notes:
    - hierarchy setup happens in Python/PyAMG;
    - repeated solves and preconditioner application happen in JAX;
    - arrays are converted to JAX sparse/dense representations;
    - JAX tracing requires static cycle/smoother/coarse-solver choices.

- `docs/quickstart.md`
  - Show the minimal pattern:
    - build a PyAMG hierarchy;
    - convert it with `MultilevelSolver.from_pyamg`;
    - call `solve`;
    - use `aspreconditioner` with a JAX Krylov solver;
    - use `jax.jit`, `jax.vmap`, and `jax.grad`.
  - Keep examples short enough to be maintainable.

- `docs/api.md`
  - Use `mkdocstrings` to render:
    - `amjax.MultilevelSolver`
    - `amjax.change_smoothers`
    - `amjax.jacobi`
    - `amjax.inverse_diagonal`
  - Do not generate internal/private API pages.

- `docs/benchmarks.md`
  - Keep prose short.
  - Include the Colab/notebook link.
  - Start with a short "Recommendations" header based on the report:
    - default to AMJax + PCG for 2D Poisson solves when accuracy and runtime
      both matter;
    - use `f64` for tight convergence targets and `f32` only when speed matters
      more than residual accuracy;
    - prefer batched solves with `jax.vmap` when solving many right-hand sides,
      with larger batches such as `k=64` generally using the GPU more
      effectively than smaller batches in the report;
    - prefer Ruge-Stüben for the headline Poisson recommendation; treat
      Smoothed Aggregation and Root Node as reliable alternatives; avoid
      Pairwise as a standalone large-system solver unless used as a
      preconditioner.
  - Include one generated benchmark summary table bounded by explicit markers:

    ```markdown
    <!-- BEGIN GENERATED BENCHMARK SUMMARY -->
    ...
    <!-- END GENERATED BENCHMARK SUMMARY -->
    ```

  - Add a note that the recommendation numbers and table are generated from the
    latest committed benchmark summary and should not be edited by hand.

### Report-grounded recommendation source

Use the semester report at:

- `/Users/victorboussange/Library/Containers/com.apple.mail/Data/Library/Mail Downloads/699C46D4-5AE9-4BBE-A49A-39290214F1D4/Semester_Project_Report_Fanny_Missillier.pdf`

The relevant findings to preserve in docs are:

- On 2D Poisson benchmarks, AMJax consistently outperforms PyAMG in the tested
  GPU setup, and AMJax + PCG gives the best runtime/convergence trade-off.
- The report states the runtime ordering as:

  ```text
  AMJax+PCG < AMJax << PyAMG+PCG < PyAMG
  ```

- Ruge-Stüben is the most efficient hierarchy in the convergence study,
  requiring 12-14 cycles across tested grid sizes; Smoothed Aggregation and
  Root Node also converge reliably but require more cycles.
- Pairwise is slow as a standalone method and fails to converge for large
  systems in standalone AMJax at `n >= 500`, but AMJax + PCG stabilizes it.
- `f64` reaches tight residuals below `1e-8`; `f32` is roughly one order of
  magnitude faster on GPU in the report but does not reach the same tolerance
  and exceeds `1e-3` residual at `n = 500`.
- Batched `vmap` solves preserve the speedups, and `k = 64` is consistently
  slightly more efficient than `k = 32`; batching helps most on small problems
  where fixed launch/dispatch costs are a larger fraction of runtime.
- Reported benchmark timings exclude hierarchy setup, device transfer, and JIT
  compilation warm-up. Keep this caveat near the benchmark recommendations.

## Stage 2: Add Docs Dependencies

Update `pyproject.toml`.

Preferred structure:

```toml
[dependency-groups]
docs = [
    "mkdocstrings-python",
    "zensical",
]
```

Update project URLs:

```toml
[project.urls]
Documentation = "https://vboussange.github.io/AMJax/"
Source = "https://github.com/vboussange/AMJax"
```

Do not add documentation dependencies to the runtime `dependencies` list.

## Stage 3: Add README Recommendation, Limitations, and Roadmap Sections

Update `README.md` so it no longer carries manually maintained benchmark
tables, but still gives readers a concise report-grounded recommendation.

Required README changes:

- Fix examples to import and use `MultilevelSolver`, not `AMJAXSolver`.
- Replace the benchmark table with a short "Recommendation" header that says:
  - for 2D Poisson problems, start with Ruge-Stüben + V-cycle + `pinv` coarse
    solve + Jacobi smoothing;
  - use AMJax + PCG when runtime and convergence both matter;
  - use `f64` for tight residuals and `f32` only for speed-first workloads;
  - batch multiple right-hand sides with `jax.vmap`, with `k=64` as the
    report-backed default when memory allows.
- Keep this README recommendation qualitative. Do not include detailed speedup
  numbers in README unless they are inside a generated block produced by the
  same benchmark-docs generator used for `docs/benchmarks.md`.
- Link to:
  - the docs benchmark page;
  - `benchmarks/latest_summary.json`;
  - `benchmarks/benchmark.ipynb`.
- Keep installation and basic usage examples in README.
- Add a "Limitations" section based on the report:
  - AMJax currently delegates hierarchy construction to PyAMG, so hierarchy
    setup is Python/CPU-side and not differentiable through the hierarchy;
  - a fully native JAX hierarchy is blocked by sparse-sparse Galerkin products
    (`P.T @ A @ P`) whose sparsity pattern is not known at JIT trace time;
  - benchmark speedups combine solver implementation, GPU execution, JIT
    compilation, and batching, so they are practical end-to-end comparisons
    rather than solver-only hardware-controlled comparisons;
  - precise GPU memory accounting is not yet reported;
  - Pairwise should not be presented as the default standalone large-system
    solver.
- Add a "Roadmap" section based on the report:
  - add more smoothers and coarse-grid solvers where they fit JAX's static
    compilation model;
  - investigate native JAX hierarchy construction for the Pairwise case, whose
    binary prolongator gives a predictable sparsity pattern;
  - add rigorous GPU memory profiling, for example with `gpu_tracker` or
    Scalene;
  - explore complex matrices and additional Krylov/preconditioner combinations
    such as FGMRES only if use cases justify the maintenance cost.

This avoids duplicating generated numbers in both README and docs. If a README
benchmark table is later desired, it should use the same generated-block
mechanism as `docs/benchmarks.md`.

## Stage 4: Define Benchmark Summary Contract

Add a compact committed summary file:

- `benchmarks/latest_summary.json`

This file is generated from raw `results/*.json` files after a benchmark run.
It is small, deterministic, and reviewed like source code.

### Summary schema

Use schema versioning from the first implementation.

```json
{
  "schema_version": 1,
  "generated_at": "ISO-8601 timestamp",
  "config_hash": "sha256 of benchmarks/config.yaml",
  "source": {
    "results_dir": "results",
    "result_file_count": 0
  },
  "selection": {
    "solver": "ruge_stuben",
    "cycle_type": "V",
    "coarse_solver": "pinv",
    "smoother": "jacobi",
    "dtype": "f64",
    "tol": 1e-10,
    "maxiter_vcycle": 100,
    "maxiter_solv": 500,
    "vmap_k": 64
  },
  "recommendations": {
    "poisson_default": "AMJax + PCG",
    "hierarchy": "ruge_stuben",
    "dtype_accuracy": "f64",
    "dtype_speed": "f32",
    "batch_size": 64
  },
  "rows": [
    {
      "scenario": "single",
      "method": "amjax",
      "grid_size": 1000,
      "cpu_speedup": null,
      "gpu_speedup": 16.0,
      "baseline": "pyamg",
      "time_seconds": 0.0,
      "baseline_time_seconds": 0.0,
      "relative_residual": 0.0
    }
  ]
}
```

The exact numbers above are placeholders. The generator must compute them from
the raw result files.

### Metric policy

The first generated table should preserve only a small set of headline metrics:

- single right-hand side, AMJax vs PyAMG;
- single right-hand side, AMJax + PCG vs PyAMG + PCG;
- batched right-hand sides with `jax.vmap`, AMJax vs PyAMG loop;
- batched right-hand sides with `jax.vmap`, AMJax + PCG vs PyAMG + PCG.

Default benchmark slice for the headline table:

- solver: `ruge_stuben`
- cycle: `V`
- smoother: `jacobi`
- coarse solver: `pinv`
- dtype: `f64`
- tolerance and iteration limits: resolved from the available benchmark run,
  preferring the configured defaults when present, so report-era results and
  new local runs can both be summarized
- grid size: largest common grid size available across the required methods
- batch size: configured `vmap_k`

If a required method/device pair is missing, render `n/a` and include a
generator warning. Do not silently reuse stale values.

## Stage 5: Add Benchmark Docs Generator

Add a script:

- `benchmarks/update_benchmark_docs.py`

### Responsibilities

The script must:

- read raw JSON files from `results/` by default;
- parse each result's `config`, `time`, and `residual` fields;
- normalize configs into comparable keys;
- choose the benchmark slice defined in Stage 4;
- compute speedups as `baseline_time / amjax_time`;
- write `benchmarks/latest_summary.json`;
- update only the generated recommendation/table block in `docs/benchmarks.md`;
- support `--check` mode for CI.

### CLI

Required commands:

```bash
uv run python benchmarks/update_benchmark_docs.py --results results --write
uv run python benchmarks/update_benchmark_docs.py --results results --check
uv run python benchmarks/update_benchmark_docs.py --summary benchmarks/latest_summary.json --check
```

Behavior:

- `--write`:
  - recompute `benchmarks/latest_summary.json` from raw results;
  - rewrite the generated block in `docs/benchmarks.md`.

- `--check` with `--results`:
  - recompute expected output from raw results;
  - fail if `benchmarks/latest_summary.json` or `docs/benchmarks.md` differs.

- `--check` with `--summary`:
  - render expected Markdown from the committed summary;
  - fail if `docs/benchmarks.md` differs.
  - This is the mode used by normal docs CI, because raw `results/` may not be
    present in pull requests.

### Markdown output

The generated block should include:

- date/time generated;
- benchmark configuration slice;
- a short recommendation callout whose qualitative claims come from the report
  and whose numerical values come from `benchmarks/latest_summary.json`;
- a compact table with scenario, method, CPU speedup, GPU speedup, residual;
- a short note that JAX timings exclude compilation if that is true in the raw
  benchmark implementation.

Avoid charts and JavaScript in v1.

## Stage 6: Wire Automatic Updates Into Benchmark Runs

The docs update should happen whenever a benchmark run is completed through the
official benchmark entry points.

### Notebook

Update the final cell of `benchmarks/benchmark.ipynb` to run:

```python
!uv run python benchmarks/update_benchmark_docs.py --results results --write
```

For Colab, ensure the command runs from the repository root.

### CLI wrapper

The scripted benchmark entry point is:

- `benchmarks/run_full_benchmark.sh`

The runner:

- executes the configured benchmark matrix through a temporary notebook copy;
- write raw JSON files into `results/`;
- call `benchmarks/update_benchmark_docs.py --results results --write` at the
  end;
- exit non-zero if any selected headline metric cannot be summarized.

This makes the automatic update part of the benchmark workflow rather than a
separate manual docs step.

## Stage 7: Add Docs Deployment Workflow

Add:

- `.github/workflows/docs.yml`

Workflow requirements:

- Run on pull requests that touch:
  - `docs/**`
  - `zensical.toml`
  - `pyproject.toml`
  - `README.md`
  - `benchmarks/latest_summary.json`
  - `benchmarks/update_benchmark_docs.py`
- Run on pushes to `main`.
- Build docs on every run.
- Deploy only on push to `main`.

Use GitHub Pages artifact deployment:

- `actions/checkout`
- `astral-sh/setup-uv`
- `actions/configure-pages`
- `actions/upload-pages-artifact`
- `actions/deploy-pages`

Commands:

```bash
uv run --group docs python benchmarks/update_benchmark_docs.py --summary benchmarks/latest_summary.json --check
uv run --group docs python -m zensical build --strict
```

Notes:

- Full benchmarks should not run in normal docs CI.
- GPU benchmark runs require suitable hardware and should not be assumed on
  GitHub-hosted runners.
- Repository Pages settings must use "GitHub Actions" as the Pages source.

## Stage 8: Add Validation Tests

Add lightweight tests that do not run the full benchmark matrix.

Recommended tests:

- A unit test for parsing one or two synthetic benchmark result JSON files.
- A unit test for speedup calculation.
- A unit test for recommendation selection and wording from a synthetic summary.
- A unit test for deterministic Markdown rendering.
- A `--check` smoke test using a tiny temporary `results/` directory.

These can live under:

- `tests/test_benchmark_docs.py`

The tests should use small synthetic JSON fixtures, not real benchmark output.

## Stage 9: First Benchmark Refresh Procedure

After the docs generator is implemented:

1. Run or rerun the benchmark notebook, or run:

   ```bash
   benchmarks/run_full_benchmark.sh
   ```

2. Confirm raw JSON files exist under `results/`.
3. Run:

   ```bash
   uv run python benchmarks/update_benchmark_docs.py --results results --write
   ```

4. Review:

   ```bash
   git diff -- benchmarks/latest_summary.json docs/benchmarks.md
   ```

5. Build docs:

   ```bash
   uv run --group docs python -m zensical build --strict
   ```

6. Run tests:

   ```bash
   uv run python -m pytest tests/ -v
   ```

7. Commit the docs, generator, summary, and workflow changes.

## Stage 10: Optional Future Improvements

These are intentionally out of scope for the first migration:

- A self-hosted GPU runner that runs benchmark refreshes on demand.
- A `workflow_dispatch` benchmark workflow that commits updated
  `benchmarks/latest_summary.json` and `docs/benchmarks.md`.
- Historical benchmark trend pages.
- Interactive benchmark charts.
- Per-release benchmark archives.
- Notebook execution inside docs.

Only add these if the project needs them. They increase maintenance cost.

## Acceptance Criteria

The migration is complete when:

- `uv run --group docs python -m zensical build --strict` succeeds.
- The docs site contains only minimal AMJax-specific pages.
- API docs render the public AMJax entry points.
- README no longer contains manually maintained benchmark numbers.
- README contains a short qualitative recommendation plus limitations and
  roadmap sections grounded in the semester report.
- `benchmarks/latest_summary.json` exists and is generated, not hand-written.
- `docs/benchmarks.md` contains a generated recommendation and benchmark block.
- Running the benchmark update script after a new benchmark run refreshes both
  the summary JSON and docs table.
- CI fails if the committed benchmark docs are stale relative to
  `benchmarks/latest_summary.json`.
- GitHub Pages deploys from the official Pages Actions workflow on `main`.

## Non-Goals

- Migrating the old Jekyll site as-is.
- Preserving the old `fannymissillier.github.io/AMJax-docs` URL.
- Committing every raw benchmark result JSON file.
- Duplicating PyAMG reference documentation.
- Maintaining an interactive benchmark dashboard in the first docs migration.
