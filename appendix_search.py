"""Three outer coordinate-search rounds using the Experiment 6 baseline."""
from __future__ import annotations

import ast
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from case_study import build_case_dataset, init_case_worker, jso_search
from experiment_common import OUTPUT_ROOT
from figures import save_round_gain
from numerical import write_csv, write_json

ROUNDS = 3
WORKERS = max(1, (os.cpu_count() or 2) // 2)
CASE_OUTPUT = OUTPUT_ROOT / "experiment6" / "rho0.3_sigma20.118"
OUTPUT = OUTPUT_ROOT / "appendix_search"


def load_case_inputs(case_output: Path = CASE_OUTPUT) -> dict:
    summary = json.loads((case_output / "summary.json").read_text(encoding="utf-8"))
    p = dict(summary["config"])
    ends = [row for row in summary["jso_outer_trace"] if row.get("coordinate_end")]
    if len(ends) != int(p["horizon"]):
        raise ValueError("The baseline must contain one full outer search round")
    search = build_case_dataset(p, p["search_scenarios"], p["search_replications"],
                                p["search_seed"], "search")
    return {
        "p": p, "search": search,
        "n1_start": list(map(int, summary["n1_search_starts"]["JSO"])),
        "main_n1": list(map(int, ast.literal_eval(summary["benchmarks"]["JSO"]["n1_vec"]))),
        "main_objective": float(ends[-1]["objective_after"]),
    }


def run_round_gain(data: dict, use_parallel: bool = True) -> tuple[list[dict], list[dict], float]:
    horizon = int(data["p"]["horizon"])
    p = {**data["p"], "outer_limit": ROUNDS * horizon, "workers": WORKERS}
    started = time.perf_counter()
    init_case_worker(p, data["search"])
    executor = ProcessPoolExecutor(max_workers=WORKERS, initializer=init_case_worker,
                                   initargs=(p, data["search"])) if use_parallel else None
    try:
        _n1, _results, trace = jso_search(p, data["search"], data["n1_start"],
                                         p["search_replications"], executor)
    finally:
        if executor is not None:
            executor.shutdown()
    ends = [row for row in trace if row.get("coordinate_end")]
    assert len(ends) == ROUNDS * horizon
    assert all(row["objective_after"] <= row["objective_before"] + 1e-9 for row in ends)
    first_round = ends[horizon - 1]
    if first_round["n1_vec"] != data["main_n1"] or not math.isclose(
            first_round["objective_after"], data["main_objective"], rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError("Appendix first round does not reproduce the saved Case search")
    initial = float(ends[0]["objective_before"])
    rows, rounds = [], []
    previous_round_objective = initial
    for coordinate, row in enumerate(ends, 1):
        before, after = float(row["objective_before"]), float(row["objective_after"])
        rows.append({
            "round": (coordinate - 1) // horizon + 1, "coordinate": coordinate,
            "shift": int(row["shift"]) + 1, "n1_before": int(row["n1_before"]),
            "n1_after": int(row["n1_after"]), "moves": int(row["moves"]),
            "objective_before": before, "objective_after": after, "gain": before - after,
            "gain_rate": (before - after) / before if before else 0.0,
            "cumulative_gain": initial - after, "elapsed_sec": float(row["elapsed_sec"]),
        })
        if coordinate % horizon == 0:
            gain = previous_round_objective - after
            rounds.append({
                "round": len(rounds) + 1, "coordinate": coordinate,
                "n1_vec": str(list(map(int, row["n1_vec"]))), "search_objective": after,
                "gain_vs_previous_round": gain,
                "gain_rate_vs_previous_round": gain / previous_round_objective if previous_round_objective else 0.0,
                "cumulative_gain": initial - after, "search_elapsed_sec": float(row["elapsed_sec"]),
            })
            previous_round_objective = after
    return rows, rounds, time.perf_counter() - started


def run(case_output: Path = CASE_OUTPUT, output: Path = OUTPUT,
        use_parallel: bool = True) -> dict:
    data = load_case_inputs(case_output)
    coordinates, rounds, runtime = run_round_gain(data, use_parallel)
    write_csv(output / "coordinate_gain.csv", coordinates)
    write_csv(output / "round_gain.csv", rounds)
    save_round_gain(coordinates, output / "coordinate_gain.png", int(data["p"]["horizon"]))
    lines = [r"\begin{tabular}{crrr}", r"\hline",
             r"Outer rounds & Average total cost & Improvement (\%) & Cumulative time (s) \\", r"\hline"]
    for row in rounds:
        improvement = "--" if row["round"] == 1 else f"{100 * row['gain_rate_vs_previous_round']:.4f}"
        lines.append(f"{row['round']} & {row['search_objective']:,.2f} & {improvement} & "
                     f"{row['search_elapsed_sec']:,.2f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "round_gain.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "case_output": str(case_output), "config": data["p"], "n1_start": data["n1_start"],
        "rounds": ROUNDS, "workers": WORKERS if use_parallel else 1,
        "round_gain": {"search_time_sec": runtime, "results": rounds},
        "cost_basis": "Average total cost on the search scenarios, not independent evaluation",
        "timing_basis": "Per-round cumulative times exclude the initial objective evaluation; total search time includes it",
    }
    write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    run()
