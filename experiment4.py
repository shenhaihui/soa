"""Experiment 4: Gamma arrivals, lognormal service, and free balking at Q > 20."""
from pathlib import Path

from experiment_revision import OUTPUT_ROOT, run_point, single_params
from numerical import write_csv


def settings(pilot: bool = False) -> list[tuple[str, dict]]:
    base = {**single_params(pilot),
            "arrival_distribution": "gamma", "arrival_scv": 2.0,
            "service_distribution": "lognormal", "service_scv": 2.0,
            "initial_residual_distribution": "lognormal", "initial_residual_scv": 2.0,
            "initial_residual_protocol": "artificial same-as-service residual; not equilibrium carryover",
            "balk_threshold": 20, "balk_penalty": 0.0,
            "qed_lookup_only": False}
    return [(f"initialization{count}", {**base, "initial_patient_count": count}) for count in (60, 120)]


def run(output: Path = OUTPUT_ROOT / "experiment4", use_parallel: bool = True, pilot: bool = False) -> list[dict]:
    summaries, rows = [], []
    for name, p in settings(pilot):
        summary = run_point(output / name, p, "single", "test" if pilot else "formal", use_parallel)
        summaries.append(summary)
        rows.extend({"initialization": p["initial_patient_count"], **row} for row in summary["benchmarks"].values())
        write_csv(output / "benchmark_table.csv", rows)
    return summaries


if __name__ == "__main__":
    run()
