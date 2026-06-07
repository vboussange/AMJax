"""Plotting utilities for JAX multigrid solver benchmarks."""

import json as _json
import numpy as np
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
    print(f"Results saved: {out}")


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
