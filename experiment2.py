"""Experiment 2: optimize separately for four initial patient counts."""
from pathlib import Path

from experiment_revision import OUTPUT_ROOT, run_point, single_params
from numerical import write_csv


def run(output: Path = OUTPUT_ROOT / "experiment2", use_parallel: bool = True, pilot: bool = False) -> list[dict]:
    summaries, rows = [], []
    for count in (30, 60, 90, 120):
        p = {**single_params(pilot), "initial_patient_count": count}
        summary = run_point(output / f"initialization{count}", p, "single", "test" if pilot else "formal", use_parallel)
        summaries.append(summary)
        rows.extend({"initialization": count, **row} for row in summary["benchmarks"].values())
        write_csv(output / "benchmark_table.csv", rows)
    return summaries


if __name__ == "__main__":
    run()
