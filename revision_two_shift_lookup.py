"""Expand BIN_LOOKUP with joint rate paths, retaining N1 and all other policies."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiment_revision import ROOT, write_latex
from numerical import build_rate_lookup, mean_ci, rate_lookup_n2, write_csv, write_json
from revision_two_shift_cond_mean import OUTPUT_ROOT as SOURCE_OUTPUT, load_inputs, read_csv
from two_shift import (
    METHODS, evaluate_replication, init_worker, joint_rate, paired_cost_rows,
    path_task, progress_map, seed, task_result, worker_context,
)

OUTPUT_ROOT = ROOT / "output" / "experiment5" / "lookup_1000"
EXTRA_SCENARIOS = 900
WORKERS = max(1, (os.cpu_count() or 2) // 2)


def load_search_paths(source: Path, p: dict, n1: list[int]) -> list[dict]:
    # Each original full path occurs once for each target shift. Reuse it only once.
    groups = [{int(r["anchor_id"]): r for r in read_csv(source / "lookup_points.csv")
               if r["source"] == "search" and int(r["shift"]) == shift} for shift in range(2)]
    if any(set(group) != set(range(p["M"])) for group in groups):
        raise ValueError("The original N1 search paths are incomplete")
    paths = []
    for sid in range(p["M"]):
        row, other = groups[0][sid], groups[1][sid]
        if any(row[key] != other[key] for key in ("lambda_1", "lambda_2", "n2_1", "n2_2", "objective_value")):
            raise ValueError("The two records of a reused path disagree")
        if [int(group[sid]["n1"]) for group in groups] != n1:
            raise ValueError("Reused surge decisions have a different base staffing")
        paths.append({"source": "search", "scenario_id": sid,
                      "lambda_1": float(row["lambda_1"]), "lambda_2": float(row["lambda_2"]),
                      "n2_1": int(row["n2_1"]), "n2_2": int(row["n2_2"]),
                      "objective_value": float(row["objective_value"]), "solver_stage": int(row["solver_stage"])})
    return paths


def additional_tasks(p: dict, n1: list[int], extra: int) -> list[tuple]:
    return [(p, n1, joint_rate(p, np.random.default_rng(seed(p, 80, sid, stream=10))), 81, sid, 0)
            for sid in range(p["M"], p["M"] + extra)]


def evaluate_lookup_scenario(task: tuple) -> dict:
    p, n1, rates, sid, tables = task
    ctx = worker_context(p)
    n2 = [rate_lookup_n2(tables[i], rates[i]) for i in range(2)]
    results = [evaluate_replication(p, "BIN_LOOKUP", n1, n2, rates, sid, rep, ctx)
               for rep in range(p["eval_replications"])]
    row = {"benchmark": "BIN_LOOKUP", "scenario_id": sid, "lambda_1": rates[0], "lambda_2": rates[1],
           "n1_vec": str(n1),
           **{key: float(np.mean([r["row"][key] for r in results])) for key in results[0]["row"]}}
    return task_result(ctx, row=row, decisions=[r for result in results for r in result["decisions"]])


def run(source: Path = SOURCE_OUTPUT, output: Path = OUTPUT_ROOT, use_parallel: bool = True,
        extra_scenarios: int = EXTRA_SCENARIOS) -> dict:
    if source.resolve() == output.resolve() or extra_scenarios < 1:
        raise ValueError("Use a separate output directory and a positive number of additional paths")
    summary, old_rows, bench, rates = load_inputs(source)
    if "cond_mean_recalculation" not in summary:
        raise ValueError("Run revision_two_shift_cond_mean.py first; retain the corrected COND_MEAN")
    p = {**summary["config"], "workers": WORKERS}
    n1 = list(summary["search"]["aux"]["n1"])
    paths = load_search_paths(source, p, n1)
    retained = ("search_results.json", "search_trace.csv", "conditional_mean_points.csv", "conditional_path_points.csv")
    inputs = ("summary.json", "scenario_costs.csv", "benchmark_table.csv", "lookup_points.csv",
              "policy_decisions.csv", "policy_information.json", "paired_comparison.csv",
              "approximation_gap_decomposition.csv", *retained)
    signature = {"extra_scenarios": extra_scenarios,
                 "input_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in inputs},
                 "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                   (Path(__file__).name, "revision_two_shift_cond_mean.py", "two_shift.py",
                                    "simulation.py", "numerical.py", "formulas.py", "experiment_revision.py")}}
    required = ("summary.json", "benchmark_table.csv", "scenario_costs.csv", "policy_decisions.csv",
                "lookup.csv", "lookup_points.csv", "pathwise_points.csv", "decision_queries.csv",
                "paired_comparison.csv", "approximation_gap_decomposition.csv", "policy_information.json",
                "benchmark_table.tex", *retained)
    config = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not config.exists() or json.loads(config.read_text()) != signature:
            raise ValueError(f"Existing lookup output has different inputs/source: {output}")
        if (output / "run_complete.json").exists() and all((output / name).is_file() for name in required):
            print(f"[Expanded lookup] SKIP completed {output}", flush=True)
            return json.loads((output / "summary.json").read_text())
        print("[Expanded lookup] Recomputing this unfinished lookup extension", flush=True)
    write_json(config, signature)
    (output / "run_complete.json").unlink(missing_ok=True)
    started, timings = time.perf_counter(), {}
    init_worker()
    executor = ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) if use_parallel else None
    print(f"[Expanded lookup] N1={n1} retained; reuse {len(paths)} paths, add {extra_scenarios}; "
          f"bins={p['lookup_bins']}, workers={WORKERS if use_parallel else 1}", flush=True)
    try:
        begin = time.perf_counter()
        tasks = additional_tasks(p, n1, extra_scenarios)
        results = progress_map(path_task, tasks, executor, "additional joint paths")
        for task, result in zip(tasks, results):
            _, _, path, stage, sid, _ = task
            paths.append({"source": "additional_joint", "scenario_id": sid,
                          "lambda_1": path[0], "lambda_2": path[1],
                          "n2_1": result["n2"][0], "n2_2": result["n2"][1],
                          "objective_value": p["c1"] * sum(n1) + result["cost"], "solver_stage": stage})
        queries = [row for result in results for row in result["queries"]]
        points = [{**row, "shift": shift, "lambda_fixed": row[f"lambda_{shift + 1}"],
                   "n1": n1[shift], "n2": row[f"n2_{shift + 1}"]} for shift in range(2) for row in paths]
        tables = [build_rate_lookup([r for r in points if r["shift"] == shift], p["lookup_bins"]) for shift in range(2)]
        write_csv(output / "pathwise_points.csv", paths)
        write_csv(output / "lookup_points.csv", points)
        write_csv(output / "lookup.csv", [{"shift": i, "n1": n1[i], **row} for i, table in enumerate(tables) for row in table])
        timings["lookup_training"] = time.perf_counter() - begin
        begin = time.perf_counter()
        results = progress_map(evaluate_lookup_scenario, [(p, n1, path, sid, tables) for sid, path in rates.items()],
                               executor, "expanded lookup evaluation")
        timings["evaluation"] = time.perf_counter() - begin
    finally:
        if executor is not None:
            executor.shutdown()
    new_rows = [r["row"] for r in results]
    rows = [r for r in old_rows if r["benchmark"] != "BIN_LOOKUP"] + new_rows
    rows.sort(key=lambda r: (int(r["scenario_id"]), METHODS.index(r["benchmark"])))
    decisions = [r for r in read_csv(source / "policy_decisions.csv") if r["benchmark"] != "BIN_LOOKUP"]
    decisions.extend(r for result in results for r in result["decisions"])
    decisions.sort(key=lambda r: (int(r["scenario_id"]), METHODS.index(r["benchmark"]), int(r["replication_id"]), int(r["shift"])))
    mean, low, high = mean_ci([r["total_cost"] for r in new_rows])
    benchmark = {"benchmark": "BIN_LOOKUP", "n1_vec": str(n1), "mean_total_cost": mean,
                 "ci95_lower": low, "ci95_upper": high,
                 **{key: float(np.mean([r[key.removeprefix("mean_")] for r in new_rows]))
                    for key in summary["benchmarks"]["BIN_LOOKUP"] if key.startswith("mean_") and key != "mean_total_cost"}}
    write_csv(output / "benchmark_table.csv", [benchmark if r["benchmark"] == "BIN_LOOKUP" else r for r in bench])
    write_csv(output / "scenario_costs.csv", rows)
    write_csv(output / "policy_decisions.csv", decisions)
    write_csv(output / "decision_queries.csv", queries + [q for r in results for q in r["queries"]])
    costs = [{**r, "scenario_id": int(r["scenario_id"]), "total_cost": float(r["total_cost"])} for r in rows]
    for name in ("paired_comparison.csv", "approximation_gap_decomposition.csv"):
        paired = []
        for old in read_csv(source / name):
            if "BIN_LOOKUP" not in (old["left"], old["right"]):
                paired.append(old)
                continue
            row = paired_cost_rows(costs, ((old["gap"], old["left"], old["right"]),))[0]
            if "mean_saving" in old:
                row.update(mean_saving=-row["mean_cost_difference"], saving_ci_lower=-row["ci95_upper"], saving_ci_upper=-row["ci95_lower"])
            paired.append(row)
        write_csv(output / name, paired)
    for name in retained:
        shutil.copy2(source / name, output / name)
    info = json.loads((source / "policy_information.json").read_text())
    info["BIN_LOOKUP"] = (f"Retain auxiliary N1 and {p['M']} original search paths; add {extra_scenarios} independent joint paths. "
                          f"Optimize a full N2 vector on each new path; per shift, bin all {len(paths)} target decisions into "
                          f"{p['lookup_bins']} equal-count groups, average raw N2, and round half-up. No conditional expansion.")
    info["seeds"] += " Expanded lookup draws joint rates at stage 80 and solves paths at stage 81; evaluation retains stage 61 and original IDs/rates."
    info["paired_statistics"] = ("Scenario-paired differences are left minus right. Only BIN_LOOKUP is recomputed. "
                                 "The legacy coarsening_gap compares independently trained lookup and exact COND_MEAN, not pure binning error.")
    write_json(output / "policy_information.json", info)
    summary["benchmarks"]["BIN_LOOKUP"] = benchmark
    summary["lookup_expansion"] = {"source_result": str(source), "retained_n1": n1, "reused_paths": p["M"],
                                   "additional_path_solves": extra_scenarios, "paths_per_shift": len(paths), "point_count": len(points),
                                   "requested_bins": p["lookup_bins"], "merged_intervals": [len(t) for t in tables],
                                   "workers": WORKERS if use_parallel else 1, "timings_sec": {**timings, "total": time.perf_counter() - started},
                                   "timing_scope": "Only this lookup extension; top-level timings/offline_training describe the original run.",
                                   "retained_policies": [m for m in METHODS if m != "BIN_LOOKUP"]}
    write_latex(output, summary, "two")
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    print(f"[Expanded lookup] END cost={mean:.6f}, elapsed={time.perf_counter() - started:.1f}s; {output}", flush=True)
    return summary


if __name__ == "__main__":
    run()
