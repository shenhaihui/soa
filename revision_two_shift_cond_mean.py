"""Recompute only COND_MEAN at the exact, saved evaluation rates."""
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

from experiment_revision import OUTPUT_ROOT as REVISION_OUTPUT, ROOT, write_latex
from numerical import mean_ci, write_csv, write_json
from two_shift import (
    METHODS, conditional_rate, evaluate_replication, init_worker, paired_cost_rows,
    path_task, progress_map, seed, task_result, worker_context,
)

SOURCE_OUTPUT = REVISION_OUTPUT / "experiment5" / "training"
OUTPUT_ROOT = ROOT / "output" / "experiment5" / "exact_direct"
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
             for r in rows if r["benchmark"] == "COND_MEAN"}
    if len(rates) != p["eval_scenarios"] or set(r["benchmark"] for r in bench) != set(METHODS):
        raise ValueError("The source must contain all four completed evaluation policies")
    for method in METHODS:
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


def refresh_comparisons(rows: list[dict], templates: list[dict]) -> list[dict]:
    costs = [{**r, "scenario_id": int(r["scenario_id"]), "total_cost": float(r["total_cost"])} for r in rows]
    result = []
    for old in templates:
        if "COND_MEAN" not in (old["left"], old["right"]):
            result.append(old)
            continue
        row = paired_cost_rows(costs, ((old["gap"], old["left"], old["right"]),))[0]
        if "mean_saving" in old:
            row.update(mean_saving=-row["mean_cost_difference"], saving_ci_lower=-row["ci95_upper"],
                       saving_ci_upper=-row["ci95_lower"])
        result.append(row)
    return result


def run(source: Path = SOURCE_OUTPUT, output: Path = OUTPUT_ROOT, use_parallel: bool = True) -> dict:
    if source.resolve() == output.resolve():
        raise ValueError("Keep the original results; choose a separate output directory")
    summary, old_rows, bench, rates = load_inputs(source)
    p = {**summary["config"], "workers": WORKERS}
    n1 = list(summary["search"]["aux"]["n1"])
    retained = ("lookup.csv", "lookup_points.csv", "search_results.json", "search_trace.csv")
    inputs = ("summary.json", "scenario_costs.csv", "benchmark_table.csv", "policy_decisions.csv",
              "policy_information.json", "paired_comparison.csv", "approximation_gap_decomposition.csv", *retained)
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
                    for key in summary["benchmarks"]["COND_MEAN"] if key.startswith("mean_") and key != "mean_total_cost"}}
    bench = [benchmark if r["benchmark"] == "COND_MEAN" else r for r in bench]
    write_csv(output / "benchmark_table.csv", bench)
    write_csv(output / "scenario_costs.csv", rows)
    write_csv(output / "policy_decisions.csv", decisions)
    write_csv(output / "decision_queries.csv", queries)
    for name in ("paired_comparison.csv", "approximation_gap_decomposition.csv"):
        write_csv(output / name, refresh_comparisons(rows, read_csv(source / name)))
    for name in retained:
        shutil.copy2(source / name, output / name)
    info = json.loads((source / "policy_information.json").read_text())
    info.update({
        "COND_MEAN": "Retain auxiliary N1. At each exact evaluation rate, independently draw K other-shift rates, "
                     "solve K pathwise N2 vectors, average the target coordinate, and round half-up. "
                     "No grid or lookup query. One fixed decision per scenario/shift across execution replications.",
        "BIN_LOOKUP": "Original trained coarse lookup and all its evaluation results retained unchanged.",
        "seeds": "Original policies keep their original streams. Exact COND_MEAN uses conditional-rate stages 70/71, "
                 "path-solver stages 72/73, and the original execution stage 61 with saved evaluation scenario IDs. "
                 "All K other-shift draws are independent of the realized other-shift rate and execution samples.",
        "paired_statistics": "Left minus right, with scenario-paired CIs. The legacy coarsening_gap field now compares "
                             "the retained coarse lookup with exact-rate COND_MEAN using different conditional samples; "
                             "it is not pure binning error. The other three policy outcomes are unchanged.",
    })
    write_json(output / "policy_information.json", info)
    summary["benchmarks"]["COND_MEAN"] = benchmark
    summary["cond_mean_recalculation"] = {
        "original_result": str(source), "retained_n1": n1, "scenario_count": len(rates), "paths_per_rate": p["K"],
        "conditional_path_count": len(points), "workers": WORKERS if use_parallel else 1,
        "timings_sec": {**timings, "total": time.perf_counter() - started},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "timing_scope": "Only this COND_MEAN correction; top-level timings/offline_training refer to the original run.",
        "retained_policies": [method for method in METHODS if method != "COND_MEAN"],
        **signature,
    }
    write_latex(output, summary, "two")
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    print(f"[Exact COND_MEAN] END cost={mean:.6f}, elapsed={time.perf_counter() - started:.1f}s; {output}", flush=True)
    return summary


if __name__ == "__main__":
    run()
