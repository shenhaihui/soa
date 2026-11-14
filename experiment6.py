"""Experiment 6: empty-start 14-shift Case and six-setting sensitivity analysis."""
import json
from pathlib import Path

from case_study import case_sensitivity_row
from experiment_common import OUTPUT_ROOT, case_settings, run_point
from numerical import write_csv

# With BASELINE_ONLY=False, RHOS can split the six settings between two devices.
RHOS = (0.3, 0.0)
BASELINE_ONLY = False
OUTPUT = OUTPUT_ROOT / "experiment6"


def run(output: Path = OUTPUT, use_parallel: bool = True,
        pilot: bool = False, rhos: tuple[float, ...] = RHOS,
        baseline_only: bool = BASELINE_ONLY) -> list[dict]:
    summaries = []
    points = case_settings(pilot, (0.3,))[:1] if baseline_only else case_settings(pilot, rhos)
    for name, p in points:
        summary = run_point(output / name, p, "case", "test" if pilot else "formal", use_parallel)
        summaries.append(summary)
        rows = []
        for saved_name, saved_p in case_settings(pilot):
            path = output / saved_name
            if (path / "run_complete.json").is_file() and (path / "summary.json").is_file():
                saved = json.loads((path / "summary.json").read_text())
                rows.append(case_sensitivity_row(saved, saved_p["rho"], saved_p["sigma2"]))
        rows.sort(key=lambda row: (row["sigma2"], row["rho"]))
        write_csv(output / "sensitivity_table.csv", rows)
    return summaries


if __name__ == "__main__":
    run()
