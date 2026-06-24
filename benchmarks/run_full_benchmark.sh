#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  benchmarks/run_full_benchmark.sh [--config benchmarks/config.yaml]

Runs the full AMJax benchmark notebook from the repository root.

The script:
  1. creates a temporary copy of benchmarks/benchmark.ipynb;
  2. skips the Colab-only setup cell;
  3. points the notebook at the requested config file;
  4. executes the notebook with nbconvert;
  5. regenerates benchmarks/latest_summary.json and docs/benchmarks.md.

Raw per-configuration JSON files are written to the root-level results/
directory, which is intentionally ignored by git.
USAGE
}

CONFIG_FILE="benchmarks/config.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      if [[ $# -lt 2 ]]; then
        echo "error: --config requires a path" >&2
        exit 2
      fi
      CONFIG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "error: config file not found: ${CONFIG_FILE}" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/amjax-benchmark.XXXXXX")"
trap 'rm -rf "${TMP_DIR}"' EXIT

PATCHED_NOTEBOOK="${TMP_DIR}/benchmark.full.ipynb"

echo "Repository: ${REPO_ROOT}"
echo "Config:     ${CONFIG_FILE}"
echo "Results:    ${REPO_ROOT}/results"

uv run python - "${CONFIG_FILE}" "${PATCHED_NOTEBOOK}" <<'PY'
import json
import sys
from pathlib import Path

config_file = sys.argv[1]
output_path = Path(sys.argv[2])
notebook_path = Path("benchmarks/benchmark.ipynb")

nb = json.loads(notebook_path.read_text())

for cell in nb["cells"]:
    source = "".join(cell.get("source", []))
    if "# -- Colab setup --" in source:
        cell["source"] = [
            "# Colab setup skipped by benchmarks/run_full_benchmark.sh\n",
            "print('Skipping Colab-only setup cell.')\n",
        ]
        cell["outputs"] = []
        cell["execution_count"] = None
    elif "CONFIG_FILE =" in source and cell.get("cell_type") == "code":
        cell["source"] = [f'CONFIG_FILE = "{config_file}"\n']
        cell["outputs"] = []
        cell["execution_count"] = None

output_path.write_text(json.dumps(nb, indent=1))
PY

echo "Executing benchmark notebook. This may take a long time for the full matrix."

uv run --with jupyter --with nbconvert --with pyyaml \
  python -m jupyter nbconvert \
  --to notebook \
  --execute "${PATCHED_NOTEBOOK}" \
  --output benchmark.executed.ipynb \
  --output-dir "${TMP_DIR}" \
  --ExecutePreprocessor.timeout=-1

echo "Refreshing generated benchmark docs."
uv run python benchmarks/update_benchmark_docs.py --results results --write
uv run python benchmarks/update_benchmark_docs.py --summary benchmarks/latest_summary.json --check

echo "Done."
echo "Review generated changes with:"
echo "  git diff -- benchmarks/latest_summary.json docs/benchmarks.md"
