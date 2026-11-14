"""Direct and Lookup deployment policies for the two-shift experiment."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from experiment_common import OUTPUT_ROOT, ROOT, write_latex
from numerical import build_rate_lookup, mean_ci, rate_lookup_n2, write_csv, write_json
from two_shift import (
    METHODS, conditional_rate, evaluate_replication, init_worker, joint_rate,
    paired_cost_rows, path_task, progress_map, seed, task_result, train_two, worker_context,
)

TRAINING_OUTPUT = OUTPUT_ROOT / "experiment5" / "training"
DIRECT_OUTPUT = OUTPUT_ROOT / "experiment5" / "exact_direct"
LOOKUP_OUTPUT = OUTPUT_ROOT / "experiment5" / "lookup_1000"
EXTRA_SCENARIOS = 900
WORKERS = max(1, (os.cpu_count() or 2) // 2)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_inputs(source: Path) -> tuple[dict, list[dict], list[dict], dict[int, list[float]]]:
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    rows = read_csv(source / "scenario_costs.csv")
    bench = read_csv(source / "benchmark_table.csv")
    p, n1 = summary["config"], summary["search"]["aux"]["n1"]
    if p["initialization"] != "empty" or p["K"] < 1:
        raise ValueError("Use the completed empty-start two-shift experiment with positive K")
    rates = {int(r["scenario_id"]): [float(r["lambda_1"]), float(r["lambda_2"])]
             for r in rows if r["benchmark"] == "SEQ_REF"}
    available = {r["benchmark"] for r in bench}
    if len(rates) != p["eval_scenarios"] or "SEQ_REF" not in available or not available <= set(METHODS):
        raise ValueError("The source must contain the completed sequential evaluation")
    for method in available:
        group = [r for r in rows if r["benchmark"] == method]
        if len(group) != len(rates) or {int(r["scenario_id"]) for r in group} != set(rates):
            raise ValueError("Source policies have different evaluation scenario IDs")
        for row in group:
            if [float(row["lambda_1"]), float(row["lambda_2"])] != rates[int(row["scenario_id"])]:
                raise ValueError("Source policies have different evaluation rates")
            if method != "SEQ_REF" and ast.literal_eval(row["n1_vec"]) != n1:
                raise ValueError("Source auxiliary policies have inconsistent N1")
    return summary, rows, bench, dict(sorted(rates.items()))


def conditional_tasks(p: dict, n1: list[int], rates: dict[int, list[float]], shift: int) -> list[tuple]:
    tasks = []
    for sid in sorted(rates):
        known = rates[sid][shift]
        for k in range(p["K"]):
            # Never reuse the other component of the realized evaluation path, even for k=0.
            other = conditional_rate(p, shift, known, np.random.default_rng(seed(p, 70 + shift, sid, k, 0, 10)))
            path = [known, other] if shift == 0 else [other, known]
            tasks.append((p, n1, path, 72 + shift, sid, k))
    return tasks


def evaluate_exact_scenario(task: tuple) -> dict:
    p, n1, rates, sid, n2 = task
    ctx = worker_context(p)
    results = [evaluate_replication(p, "COND_MEAN", n1, n2, rates, sid, rep, ctx)
               for rep in range(p["eval_replications"])]
    row = {"benchmark": "COND_MEAN", "scenario_id": sid, "lambda_1": rates[0], "lambda_2": rates[1],
           "n1_vec": str(n1),
           **{key: float(np.mean([result["row"][key] for result in results])) for key in results[0]["row"]}}
    return task_result(ctx, row=row, decisions=[r for result in results for r in result["decisions"]])


def save_comparisons(output: Path, rows: list[dict]) -> None:
    costs = [{**r, "scenario_id": int(r["scenario_id"]), "total_cost": float(r["total_cost"])} for r in rows]
    available = {r["benchmark"] for r in costs}
    comparisons = (("offline_policy_gap", "COND_MEAN", "SEQ_REF"),
                   ("coarsening_gap", "BIN_LOOKUP", "COND_MEAN"),
                   ("total_gap", "BIN_LOOKUP", "SEQ_REF"))
    active = tuple(row for row in comparisons if row[1] in available and row[2] in available)
    values = paired_cost_rows(costs, active)
    write_csv(output / "approximation_gap_decomposition.csv", values)
    write_csv(output / "paired_comparison.csv", [
        {**r, "mean_saving": -r["mean_cost_difference"], "saving_ci_lower": -r["ci95_upper"],
         "saving_ci_upper": -r["ci95_lower"]} for r in values])


def run_direct(source: Path = TRAINING_OUTPUT, output: Path = DIRECT_OUTPUT, use_parallel: bool = True) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Keep the original results; choose a separate output directory")
    summary, old_rows, bench, rates = load_inputs(source)
    p = {**summary["config"], "workers": WORKERS}
    n1 = list(summary["search"]["aux"]["n1"])
    retained = ("search_paths.csv", "search_results.json", "search_trace.csv", "online_timing.csv")
    inputs = ("summary.json", "scenario_costs.csv", "benchmark_table.csv", "policy_decisions.csv",
              "policy_information.json", *retained)
    signature = {
        "input_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in inputs},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in (Path(__file__).name, "two_shift.py", "simulation.py", "numerical.py", "formulas.py")},
    }
    config_path = output / "run_config.json"
    required = ("summary.json", "benchmark_table.csv", "scenario_costs.csv", "policy_decisions.csv",
                "conditional_mean_points.csv", "conditional_path_points.csv", "decision_queries.csv",
                "paired_comparison.csv", "approximation_gap_decomposition.csv", "policy_information.json",
                "benchmark_table.tex", *retained)
    if output.exists() and any(output.iterdir()):
        if not config_path.exists() or json.loads(config_path.read_text()) != signature:
            raise ValueError(f"Existing COND_MEAN output has different inputs/source: {output}")
        if (output / "run_complete.json").exists() and all((output / name).is_file() for name in required):
            print(f"[Exact COND_MEAN] SKIP completed {output}", flush=True)
            return json.loads((output / "summary.json").read_text())
        print("[Exact COND_MEAN] Recomputing this unfinished correction", flush=True)
    write_json(config_path, signature)
    (output / "run_complete.json").unlink(missing_ok=True)
    started, timings, points, anchors, queries = time.perf_counter(), {}, [], [], []
    fixed = {sid: [0, 0] for sid in rates}
    init_worker()
    executor = ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) if use_parallel else None
    print(f"[Exact COND_MEAN] N1={n1} retained; {2 * len(rates) * p['K']} conditional paths, "
          f"evaluation={len(rates)}x{p['eval_replications']}, workers={WORKERS if use_parallel else 1}", flush=True)
    try:
        begin = time.perf_counter()
        for shift in range(2):
            tasks = conditional_tasks(p, n1, rates, shift)
            results = progress_map(path_task, tasks, executor, f"exact conditional shift {shift + 1}")
            values = {sid: [] for sid in rates}
            for task, result in zip(tasks, results):
                _, _, path, stage, sid, k = task
                values[sid].append(result["n2"][shift])
                points.append({"scenario_id": sid, "shift": shift, "conditional_id": k,
                               "lambda_fixed": path[shift], "lambda_1": path[0], "lambda_2": path[1],
                               "n1": n1[shift], "n2": result["n2"][shift], "n2_1": result["n2"][0],
                               "n2_2": result["n2"][1], "objective_value": p["c1"] * sum(n1) + result["cost"],
                               "solver_stage": stage})
                queries.extend(result["queries"])
            for sid in rates:
                mean = float(np.mean(values[sid]))
                fixed[sid][shift] = int(math.floor(mean + 0.5))
                anchors.append({"scenario_id": sid, "shift": shift, "lambda_fixed": rates[sid][shift],
                                "n1": n1[shift], "source_count": len(values[sid]), "n2_mean": mean,
                                "n2": fixed[sid][shift]})
        timings["conditional_n2"] = time.perf_counter() - begin
        write_csv(output / "conditional_path_points.csv", points)
        write_csv(output / "conditional_mean_points.csv", anchors)
        begin = time.perf_counter()
        results = progress_map(evaluate_exact_scenario, [(p, n1, path, sid, fixed[sid]) for sid, path in rates.items()],
                               executor, "exact COND_MEAN evaluation")
        timings["evaluation"] = time.perf_counter() - begin
    finally:
        if executor is not None:
            executor.shutdown()
    new_rows = [result["row"] for result in results]
    rows = [r for r in old_rows if r["benchmark"] != "COND_MEAN"] + new_rows
    rows.sort(key=lambda r: (int(r["scenario_id"]), METHODS.index(r["benchmark"])))
    decisions = [r for r in read_csv(source / "policy_decisions.csv") if r["benchmark"] != "COND_MEAN"]
    decisions.extend(r for result in results for r in result["decisions"])
    decisions.sort(key=lambda r: (int(r["scenario_id"]), METHODS.index(r["benchmark"]), int(r["replication_id"]), int(r["shift"])))
    mean, low, high = mean_ci([r["total_cost"] for r in new_rows])
    benchmark = {"benchmark": "COND_MEAN", "n1_vec": str(n1), "mean_total_cost": mean,
                 "ci95_lower": low, "ci95_upper": high,
                 **{key: float(np.mean([r[key.removeprefix("mean_")] for r in new_rows]))
                    for key in summary["benchmarks"]["SEQ_REF"] if key.startswith("mean_") and key != "mean_total_cost"}}
    bench = [r for r in bench if r["benchmark"] != "COND_MEAN"] + [benchmark]
    bench.sort(key=lambda r: METHODS.index(r["benchmark"]))
    write_csv(output / "benchmark_table.csv", bench)
    write_csv(output / "scenario_costs.csv", rows)
    write_csv(output / "policy_decisions.csv", decisions)
    write_csv(output / "decision_queries.csv", queries)
    save_comparisons(output, rows)
    for name in retained:
        shutil.copy2(source / name, output / name)
    info = json.loads((source / "policy_information.json").read_text())
    info.update({
        "COND_MEAN": "Retain auxiliary N1. At each exact evaluation rate, independently draw K other-shift rates, "
                     "solve K pathwise N2 vectors, average the target coordinate, and round half-up. "
                     "No grid or lookup query. One fixed decision per scenario/shift across execution replications.",
        "seeds": "Original policies keep their original streams. Exact COND_MEAN uses conditional-rate stages 70/71, "
                 "path-solver stages 72/73, and the original execution stage 61 with saved evaluation scenario IDs. "
                 "All K other-shift draws are independent of the realized other-shift rate and execution samples.",
        "paired_statistics": "Left minus right, with scenario-paired CIs. Direct averaging includes auxiliary N1 "
                             "and conditional-decision approximation. No training-rate grid is evaluated.",
    })
    write_json(output / "policy_information.json", info)
    summary["benchmarks"]["COND_MEAN"] = benchmark
    summary["cond_mean_recalculation"] = {
        "original_result": str(source), "retained_n1": n1, "scenario_count": len(rates), "paths_per_rate": p["K"],
        "conditional_path_count": len(points), "workers": WORKERS if use_parallel else 1,
        "timings_sec": {**timings, "total": time.perf_counter() - started},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "timing_scope": "Only exact-rate conditional averaging and its evaluation; top-level timings describe base searches and OPT evaluation.",
        "retained_policies": [r["benchmark"] for r in bench if r["benchmark"] != "COND_MEAN"],
        **signature,
    }
    write_latex(output, summary, "two")
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    print(f"[Exact COND_MEAN] END cost={mean:.6f}, elapsed={time.perf_counter() - started:.1f}s; {output}", flush=True)
    return summary


def load_search_paths(source: Path, p: dict, n1: list[int]) -> list[dict]:
    rows = sorted(read_csv(source / "search_paths.csv"), key=lambda r: int(r["scenario_id"]))
    if len(rows) != p["M"] or [int(r["scenario_id"]) for r in rows] != list(range(p["M"])):
        raise ValueError("The original N1 search paths are incomplete")
    paths = []
    for row in rows:
        if ast.literal_eval(row["n1_vec"]) != n1:
            raise ValueError("Reused surge decisions have a different base staffing")
        paths.append({"source": "search", "scenario_id": int(row["scenario_id"]),
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


def run_lookup(source: Path = DIRECT_OUTPUT, output: Path = LOOKUP_OUTPUT, use_parallel: bool = True,
        extra_scenarios: int = EXTRA_SCENARIOS) -> dict:
    if source.resolve() == output.resolve() or extra_scenarios < 1:
        raise ValueError("Use a separate output directory and a positive number of additional paths")
    summary, old_rows, bench, rates = load_inputs(source)
    if "cond_mean_recalculation" not in summary:
        raise ValueError("Run run_direct first; retain the exact-rate Direct result")
    p = {**summary["config"], "workers": WORKERS}
    n1 = list(summary["search"]["aux"]["n1"])
    paths = load_search_paths(source, p, n1)
    retained = ("search_paths.csv", "search_results.json", "search_trace.csv", "online_timing.csv",
                "conditional_mean_points.csv", "conditional_path_points.csv")
    inputs = ("summary.json", "scenario_costs.csv", "benchmark_table.csv",
              "policy_decisions.csv", "policy_information.json", "paired_comparison.csv",
              "approximation_gap_decomposition.csv", *retained)
    signature = {"extra_scenarios": extra_scenarios,
                 "input_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in inputs},
                 "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                                   (Path(__file__).name, "two_shift.py",
                                    "simulation.py", "numerical.py", "formulas.py", "experiment_common.py")}}
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
                    for key in summary["benchmarks"]["SEQ_REF"] if key.startswith("mean_") and key != "mean_total_cost"}}
    bench = [r for r in bench if r["benchmark"] != "BIN_LOOKUP"] + [benchmark]
    bench.sort(key=lambda r: METHODS.index(r["benchmark"]))
    write_csv(output / "benchmark_table.csv", bench)
    write_csv(output / "scenario_costs.csv", rows)
    write_csv(output / "policy_decisions.csv", decisions)
    write_csv(output / "decision_queries.csv", queries + [q for r in results for q in r["queries"]])
    save_comparisons(output, rows)
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
                                   "timing_scope": "Only the additional joint paths, lookup construction and evaluation; top-level timings describe base searches and OPT evaluation.",
                                   "retained_policies": [m for m in METHODS if m != "BIN_LOOKUP"]}
    write_latex(output, summary, "two")
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    print(f"[Expanded lookup] END cost={mean:.6f}, elapsed={time.perf_counter() - started:.1f}s; {output}", flush=True)
    return summary


def run_pipeline(output: Path, p: dict, use_parallel: bool = True) -> dict:
    global WORKERS
    WORKERS = p["workers"]
    training, exact, expanded = (output / name for name in ("training", "exact_direct", "lookup_1000"))
    train_two(training, p, use_parallel)
    run_direct(training, exact, use_parallel)
    summary = run_lookup(exact, expanded, use_parallel, p["lookup_extra_paths"])
    times = summary["timings_sec"]
    direct_times = summary["cond_mean_recalculation"]["timings_sec"]
    lookup_times = summary["lookup_expansion"]["timings_sec"]
    bench = []
    for label in METHODS:
        offline = times["search_seq"] if label == "SEQ_REF" else times["search_aux"]
        if label == "BIN_LOOKUP":
            offline += lookup_times["lookup_training"]
        online = times["online_total"] if label == "SEQ_REF" else (
            direct_times["conditional_n2"] if label == "COND_MEAN" else 0.0)
        bench.append({**summary["benchmarks"][label], "offline_search_sec": offline, "online_search_sec": online})
    write_csv(output / "benchmark_table.csv", bench)
    final_rows = read_csv(expanded / "scenario_costs.csv")
    write_csv(output / "scenario_costs.csv", final_rows)
    write_csv(output / "policy_decisions.csv", read_csv(expanded / "policy_decisions.csv"))
    save_comparisons(output, final_rows)
    for name in ("lookup.csv", "lookup_points.csv", "search_paths.csv", "search_results.json", "search_trace.csv", "policy_information.json", "online_timing.csv"):
        shutil.copy2(expanded / name, output / name)
    write_json(output / "summary.json", summary)
    write_latex(output, summary, "two")
    return summary
