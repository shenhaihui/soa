"""Experiment 6: empty-start 14-shift Case, baseline plus six demand settings."""
import json
from pathlib import Path

from case_study import case_sensitivity_row
from experiment_revision import OUTPUT_ROOT, case_settings, run_point
from numerical import write_csv

# Use (0.3,) or (0.0,) to split the six settings between two devices.
RHOS = (0.3, 0.0)


def run(output: Path = OUTPUT_ROOT / "experiment6", use_parallel: bool = True,
        pilot: bool = False, rhos: tuple[float, ...] = RHOS) -> list[dict]:
    summaries = []
    for name, p in case_settings(pilot, rhos):
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
