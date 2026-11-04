"""Experiment 3: cost and demand-variance sensitivity, initialization 60/120."""
from pathlib import Path

from experiment_revision import OUTPUT_ROOT, run_point, single_params
from numerical import write_csv


def settings(pilot: bool = False) -> list[tuple[str, dict]]:
    return [(f"initialization{count}_h{h:g}_var{variance:g}",
             {**single_params(pilot), "initial_patient_count": count,
              "h": h, "a": 2 * h, "sigma": variance ** 0.5})
            for count in (60, 120) for h in (1.5, 20.0) for variance in (1.0, 3.5)]


def run(output: Path = OUTPUT_ROOT / "experiment3", use_parallel: bool = True, pilot: bool = False) -> list[dict]:
    summaries, rows = [], []
    for name, p in settings(pilot):
        # The two baseline settings are exactly the corresponding Experiment 2 runs.
        path = (output.parent / "experiment2" / f"initialization{p['initial_patient_count']}"
                if p["h"] == 1.5 and p["sigma"] == 1.0 else output / name)
        summary = run_point(path, p, "single", "test" if pilot else "formal", use_parallel)
        summaries.append(summary)
        rows.extend({"initialization": p["initial_patient_count"], "h": p["h"], "a": p["a"],
                     "variance": p["sigma"] ** 2, **row} for row in summary["benchmarks"].values())
        write_csv(output / "benchmark_table.csv", rows)
    return summaries


if __name__ == "__main__":
    run()
