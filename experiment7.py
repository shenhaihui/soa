"""Experiment 7: fit a 75-patient baseline, then evaluate continuous 52-week paths."""
import ast
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from case_study import (
    aggregate_case, build_case_dataset, case_initial_state, lookup_n2,
    n1_vectors_for_params, run_case_path,
)
from experiment_common import OUTPUT_ROOT, case_settings, run_point
from formulas import adjusted_n2, eta_star
from numerical import mean_ci, write_csv, write_json

CASE_INPUT = OUTPUT_ROOT / "experiment6_initialization75" / "rho0.3_sigma20.118"
OUTPUT = OUTPUT_ROOT / "experiment7_initialization75"
INITIAL_PATIENT_COUNT = 75
WEEKS = 52
TRAJECTORIES = 1000
WORKERS = max(1, (os.cpu_count() or 2) // 2)
POLICIES = ("1,NV", "1,SO", "2,QED", "JSO")
_ROLLING = None


def prepare_baseline(source: Path, use_parallel: bool, workers: int) -> None:
    if (source / "run_complete.json").is_file():
        print(f"[Rolling] Reuse completed 75-patient baseline: {source}", flush=True)
        return
    _, p = case_settings()[0]
    p.update(initialization="fixed_count", initial_patient_count=INITIAL_PATIENT_COUNT, workers=workers)
    run_point(source, p, "case", "formal", use_parallel)


def load_case_inputs(source: Path) -> tuple[dict, dict, list[dict]]:
    if not (source / "run_complete.json").is_file():
        raise FileNotFoundError("Complete the separate 75-patient baseline before rolling evaluation")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if summary["config"].get("initial_patient_count", 0) <= 0:
        raise ValueError("Use a nonempty-initialization baseline, not the empty-start Experiment 6")
    bases = {label: ast.literal_eval(summary["benchmarks"][label]["n1_vec"]) for label in POLICIES}
    bins = summary["config"]["lhs_bin_counts"][0]
    with (source / f"jso_lookup_{bins}bins.csv").open(newline="", encoding="utf-8") as handle:
        lookup = list(csv.DictReader(handle))
    return summary, bases, lookup


def init_rolling_worker(p: dict, dataset: list[dict], label: str, n1: list[int], lookup: list[dict]) -> None:
    global _ROLLING
    eta_star.cache_clear()
    n1_qed = n1_vectors_for_params(p)[1] if label == "2,QED" else None
    _ROLLING = (p, dataset, label, n1, n1_qed, lookup)


def trajectory_task(sid: int) -> dict:
    p, dataset, label, n1, n1_qed, lookup = _ROLLING
    scenario = dataset[sid]
    seed = scenario["trials"][0]["seed"]
    initial = case_initial_state(p, seed)
    if label == "2,QED":
        rule = lambda i, lam, state: adjusted_n2(lam, n1[i], n1_qed[i], len(state),
                                                p["h"], p["mu"], p["theta"], p["a"], p["c2"], p["xi2"])
    elif label == "JSO":
        rule = lambda i, lam, _state: lookup_n2(lookup, i, lam)
    else:
        rule = lambda _i, _lam, _state: 0
    metrics, n2 = run_case_path(p, initial, n1, rule, scenario["lambda_vec"], seed)
    weeks = len(n2) // len(n1)
    phase_means = np.asarray(n2).reshape(weeks, len(n1)).mean(axis=0)
    return {"benchmark": label, "scenario_id": sid, "total_lambda": sum(scenario["lambda_vec"]),
            **metrics, "weekly_total_cost": metrics["total_cost"] / weeks,
            "n2_sum": sum(n2) / weeks,
            **{f"n2_{i + 1}": float(value) for i, value in enumerate(phase_means)}}


def paired_rows(rows: list[dict], weeks: int) -> list[dict]:
    by = {label: {int(row["scenario_id"]): row for row in rows if row["benchmark"] == label}
          for label in POLICIES}
    result = []
    for baseline in POLICIES[:-1]:
        left, right = by[baseline], by["JSO"]
        assert left.keys() == right.keys()
        ids = sorted(left)
        assert all(left[sid]["total_lambda"] == right[sid]["total_lambda"] for sid in ids)
        differences = [left[sid]["total_cost"] - right[sid]["total_cost"] for sid in ids]
        mean, low, high = mean_ci(differences)
        result.append({"baseline": baseline, "comparison": "JSO", "mean_cost_saving": mean,
                       "ci95_lower": low, "ci95_upper": high,
                       "mean_weekly_saving": mean / weeks,
                       "weekly_ci95_lower": low / weeks, "weekly_ci95_upper": high / weeks,
                       "saving_pct": 100 * mean / np.mean([left[sid]["total_cost"] for sid in ids]),
                       "win_rate_pct": 100 * float(np.mean(np.asarray(differences) > 0)),
                       "trajectory_count": len(ids)})
    return result


def run(output: Path = OUTPUT, case_input: Path | None = None, use_parallel: bool = True,
        trajectories: int = TRAJECTORIES, weeks: int = WEEKS, workers: int = WORKERS) -> dict:
    baseline_started = time.perf_counter()
    if case_input is None:
        case_input = CASE_INPUT
        prepare_baseline(case_input, use_parallel, workers)
    baseline_preparation_sec = time.perf_counter() - baseline_started
    trained, bases, lookup = load_case_inputs(case_input)
    p = {**trained["config"], "seed_namespace": 2026111107,
         "eval_scenarios": trajectories, "eval_replications": 1, "workers": workers}
    if trajectories < 1 or weeks < 1 or weeks * p["horizon"] > 1024:
        raise ValueError("Use positive budgets and at most 1024 shifts per trajectory")
    source = Path(__file__).resolve().parent
    bins = p["lhs_bin_counts"][0]
    signature = {"config": {k: v for k, v in p.items() if k != "workers"}, "weeks": weeks,
                 "training": {name: hashlib.sha256((case_input / name).read_bytes()).hexdigest()
                              for name in ("summary.json", f"jso_lookup_{bins}bins.csv")},
                 "source": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                            for name in ("experiment7.py", "case_study.py", "numerical.py", "simulation.py", "formulas.py")}}
    signature = json.loads(json.dumps(signature))
    config = output / "run_config.json"
    required = ("summary.json", "benchmark_table.csv", "scenario_costs.csv", "paired_comparison.csv", "demand_paths.json")
    if output.exists() and any(output.iterdir()):
        if not config.is_file() or json.loads(config.read_text()) != signature:
            raise ValueError("Existing Experiment 7 output has different inputs/source; use a new directory")
        if (output / "run_complete.json").is_file() and all((output / name).is_file() for name in required):
            print(f"[Rolling] SKIP completed {output}", flush=True)
            return json.loads((output / "summary.json").read_text())
    write_json(config, signature)
    (output / "run_complete.json").unlink(missing_ok=True)
    started, timings, rows, bench = time.perf_counter(), {}, [], []
    # One AR(1) path spans every week; only the deterministic weekly means repeat.
    long_params = {**p, "lambda_means": list(p["lambda_means"]) * weeks}
    dataset = build_case_dataset(long_params, trajectories, 1, 1, "rolling_eval")
    write_json(output / "demand_paths.json", dataset)
    for label in POLICIES:
        begin = time.perf_counter()
        print(f"[Rolling] START {label}: {trajectories} trajectories x {weeks} weeks", flush=True)
        args = (p, dataset, label, bases[label], lookup)
        if use_parallel:
            with ProcessPoolExecutor(max_workers=workers, initializer=init_rolling_worker, initargs=args) as executor:
                group = []
                for row in executor.map(trajectory_task, range(trajectories)):
                    group.append(row)
                    if len(group) % 100 == 0:
                        print(f"[Rolling] {label}: {len(group)}/{trajectories}, "
                              f"elapsed={time.perf_counter() - begin:.1f}s", flush=True)
        else:
            init_rolling_worker(*args)
            group = [trajectory_task(sid) for sid in range(trajectories)]
        timings[label] = time.perf_counter() - begin
        rows.extend(group)
        benchmark = {"benchmark": label, **aggregate_case(group, bases[label], trained["benchmarks"][label]["search_time_sec"])}
        for key in ("total_cost", "base_cost", "surge_cost", "waiting_cost", "abandonment_cost"):
            benchmark[f"mean_weekly_{key}"] = benchmark[f"mean_{key}"] / weeks
        benchmark.update(weekly_ci95_lower=benchmark["ci95_lower"] / weeks,
                         weekly_ci95_upper=benchmark["ci95_upper"] / weeks,
                         mean_weekly_n2_sum=float(np.mean([row["n2_sum"] for row in group])))
        bench.append(benchmark)
        write_csv(output / "benchmark_table.csv", bench)
        write_csv(output / "scenario_costs.csv", rows)
        print(f"[Rolling] END {label}, elapsed={timings[label]:.1f}s", flush=True)
    comparisons = paired_rows(rows, weeks)
    write_csv(output / "paired_comparison.csv", comparisons)
    summary = {"config": p, "weeks": weeks, "trajectories": trajectories,
               "case_input": str(case_input), "benchmarks": {row["benchmark"]: row for row in bench},
               "baseline_preparation_sec": baseline_preparation_sec,
               "paired_comparisons": comparisons, "evaluation_time_sec": timings,
               "actual_workers": workers if use_parallel else 1,
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
               "total_runtime_sec": time.perf_counter() - started,
               "semantics": "One DES execution per demand trajectory. Initial patients and execution streams "
                            "are common across policies; each policy has fresh workers/caches. Patient state "
                            "and AR(1) demand continue across weeks. The fitted 14-shift bases and lookup repeat "
                            "without reoptimization. Total costs cover all weeks; weekly columns divide by weeks. "
                            "Confidence intervals and savings are trajectory-paired; offline search times are "
                            "retained from the separate nonempty baseline. Baseline preparation wall time is "
                            "reported separately and excluded from total_runtime_sec for rolling evaluation."}
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    return summary


if __name__ == "__main__":
    run()
