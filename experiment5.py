"""Experiment 5: retain the fitted bases, then build exact Direct and a 1000-path lookup."""
import csv
from pathlib import Path

import revision_two_shift_cond_mean as direct
import revision_two_shift_lookup as lookup
import revision_two_shift_online_time as online
from experiment_revision import OUTPUT_ROOT, run_point, write_latex
from numerical import write_csv, write_json
from two_shift import two_params

RUN_ONLINE_TIMING = True
METHODS = ("SEQ_REF", "COND_MEAN", "BIN_LOOKUP")


def run(output: Path = OUTPUT_ROOT / "experiment5", use_parallel: bool = True,
        pilot: bool = False, time_online: bool = RUN_ONLINE_TIMING) -> dict:
    p = two_params(pilot)
    training = output / "training"
    exact = output / "exact_direct"
    expanded = output / "lookup_1000"
    run_point(training, p, "two", "test" if pilot else "formal", use_parallel)
    direct.WORKERS = lookup.WORKERS = online.WORKERS = p["workers"]
    direct.run(training, exact, use_parallel)
    summary = lookup.run(exact, expanded, use_parallel, extra_scenarios=2 if pilot else 900)
    timing_path = output / "opt_online_timing"
    if time_online:
        timed = online.run(expanded, timing_path, use_parallel)
    else:
        timed = None
    # Publish only the three manuscript policies; intermediate files stay separate.
    summary = {**summary, "benchmarks": {label: summary["benchmarks"][label] for label in METHODS}}
    times = summary["timings_sec"]
    direct_times = summary["cond_mean_recalculation"]["timings_sec"]
    lookup_times = summary["lookup_expansion"]["timings_sec"]
    bench = []
    for label in METHODS:
        offline = times["search_seq"] if label == "SEQ_REF" else times["search_aux"]
        if label == "BIN_LOOKUP":
            offline += lookup_times["lookup_training"]
        online_time = (timed["timings_sec"]["online_total"] if timed else None) if label == "SEQ_REF" else (
            direct_times["conditional_n2"] if label == "COND_MEAN" else 0.0)
        bench.append({**summary["benchmarks"][label], "offline_search_sec": offline,
                      "online_search_sec": online_time})
    write_csv(output / "benchmark_table.csv", bench)
    for filename in ("scenario_costs.csv", "policy_decisions.csv", "lookup.csv"):
        with (expanded / filename).open(newline="", encoding="utf-8") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("benchmark", "BIN_LOOKUP") in METHODS]
        write_csv(output / filename, rows)
    summary["opt_online_timing"] = timed
    write_json(output / "summary.json", summary)
    write_latex(output, summary, "two")
    return summary


if __name__ == "__main__":
    run()
