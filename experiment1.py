"""Experiment 1: empty-start single shift, including isolated policy timings."""
from pathlib import Path

from experiment_revision import OUTPUT_ROOT, run_point, single_params
from numerical import write_csv


def run(output: Path = OUTPUT_ROOT / "experiment1", use_parallel: bool = True, pilot: bool = False) -> dict:
    p = single_params(pilot)
    summary = run_point(output, p, "single", "test" if pilot else "formal", use_parallel)
    write_csv(output / "policy_timings.csv", [
        {"benchmark": label, "offline_search_sec": times["offline_search"],
         "online_search_sec": times["online_search"], "evaluation_sec": times["evaluation"],
         "search_scenarios": p["search_scenarios"], "eval_scenarios": p["eval_scenarios"],
         "replications": p["replications"]}
        for label, times in summary["policy_timings_sec"].items()])
    return summary


if __name__ == "__main__":
    run()
