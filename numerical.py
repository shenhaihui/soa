from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from formulas import eta_star, newsvendor_n1, qed_n1, qed_n2
from simulation import build_model_stream, build_shift_stream_crn, service_samples, simulate_shift

_WORKER_P: dict[str, Any] | None = None
_WORKER_DATASET: list[dict[str, Any]] | None = None
_WORKER_POOL: list[dict[str, Any]] | None = None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(data.mean())
    if data.size == 1:
        return mean, mean, mean
    margin = float(student_t.ppf(0.975, data.size - 1) * data.std(ddof=1) / math.sqrt(data.size))
    return mean, mean - margin, mean + margin


def paired_comparison(rows: list[dict], policies: tuple[str, ...]) -> list[dict]:
    by_policy = {label: {int(r["scenario_id"]): r for r in rows if r["benchmark"] == label}
                 for label in policies}
    result = []
    for baseline, comparison in combinations(policies, 2):
        left, right = by_policy[baseline], by_policy[comparison]
        assert left.keys() == right.keys()
        ids = sorted(left)
        assert all(float(left[s]["lambda_fixed"]) == float(right[s]["lambda_fixed"]) for s in ids)
        differences = [float(left[s]["total_cost"]) - float(right[s]["total_cost"]) for s in ids]
        mean, low, high = mean_ci(differences)
        base_mean = float(np.mean([float(left[s]["total_cost"]) for s in ids]))
        result.append({"baseline": baseline, "comparison": comparison,
                       "mean_saving": mean, "ci95_lower": low, "ci95_upper": high,
                       "saving_rate": mean / base_mean,
                       "win_rate": float(np.mean(np.asarray(differences) > 0)),
                       "scenario_count": len(ids)})
    return result


def sample_lambda(rng: np.random.Generator, lambda_mean: float, alpha: float, sigma: float) -> float:
    while True:
        val = float(lambda_mean + rng.normal(0.0, sigma) * lambda_mean ** alpha)
        if val > 0.0:
            return val


def namespaced_seed(namespace: int, stage: int, scenario: int, replication: int, stream: int) -> int:
    fields = (namespace, stage, scenario, replication, stream)
    widths = (32, 16, 40, 32, 8)
    seed = 0
    for value, width in zip(fields, widths):
        if not isinstance(value, int) or not 0 <= value < 2 ** width:
            raise ValueError(f"Seed field {value} outside its {width}-bit range")
        seed = (seed << width) | value
    return seed


def build_rate_lookup(points: list[dict[str, Any]], bins: int) -> list[dict[str, Any]]:
    if not 1 <= int(bins) <= len(points):
        raise ValueError("Lookup bins must be between 1 and the number of training points")
    ordered = sorted(points, key=lambda r: (float(r["lambda_fixed"]), str(r.get("source", "")), int(r["scenario_id"])))
    rates = [float(r["lambda_fixed"]) for r in ordered]
    if any(not math.isfinite(x) or x <= 0 for x in rates):
        raise ValueError("Lookup training rates must be finite and positive")
    if any(int(r["n2"]) != r["n2"] or r["n2"] < 0 for r in ordered):
        raise ValueError("Lookup decisions must be nonnegative integers")
    # Move quantile cuts past ties so equal rates can never receive conflicting bins.
    ends = set()
    for chunk in np.array_split(np.arange(len(ordered)), int(bins)):
        end = int(chunk[-1]) + 1
        while end < len(rates) and rates[end - 1] == rates[end]:
            end += 1
        ends.add(end)
    rows, start, lower = [], 0, 0.0
    for end in sorted(ends):
        values = ordered[start:end]
        upper = float("inf") if end == len(rates) else rates[end - 1] + (rates[end] - rates[end - 1]) / 2.0
        if end < len(rates) and upper >= rates[end]:
            upper = rates[end - 1]
        mean = float(np.mean([r["n2"] for r in values]))
        decision = int(math.floor(mean + 0.5))
        if rows and rows[-1]["n2"] == decision:
            previous = rows[-1]
            count = previous["source_count"] + len(values)
            previous["n2_mean"] = (previous["n2_mean"] * previous["source_count"] + mean * len(values)) / count
            previous.update(lambda_upper=upper, source_count=count)
        else:
            rows.append({"bin_id": len(rows), "lambda_lower": lower, "lambda_upper": upper,
                         "source_count": len(values), "n2_mean": mean, "n2": decision})
        start, lower = end, upper
    return rows


def rate_lookup_n2(lookup: list[dict[str, Any]], lam: float) -> int:
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("Lookup query must be a finite positive rate")
    for row in lookup:
        if float(row["lambda_lower"]) < lam <= float(row["lambda_upper"]):
            return int(row["n2"])
    raise ValueError("Lookup intervals do not cover the query")


def numerical_params() -> dict[str, Any]:
    return {
        "warmup_shifts": 10,
        "lambda_mean": 100.0,
        "alpha": 0.75,
        "sigma": 1.0,
        "mu": 1.0,
        "theta": 0.1,
        "c1": 1.0,
        "c2": 1.5,
        "h": 1.5,
        "a": 3.0,
        "search_scenarios": 1000,
        "eval_scenarios": 1000,
        "replications": 100,
        "pool_size": 3000,
        "warmup_seed": 100_000_000,
        "search_seed": 500_000_000,
        "online_n2_seed": 1_000_000_000,
        "eval_seed": 1_500_000_000,
        "search_upper_multiplier": 2,
        "n2_search_upper_floor": 20,
        "workers": max(1, (os.cpu_count() or 2) // 2),
    }


def warmup_sample(args: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    sample_id, p = args
    sample_seed = int(p["warmup_seed"]) + 97 * int(sample_id)
    rng = np.random.default_rng(sample_seed)
    n1 = qed_n1(p["lambda_mean"], p["c1"], p["c2"], p["h"], p["mu"], p["theta"], p["a"], p["sigma"], p["alpha"])
    state: list[dict[str, Any]] = []
    for shift in range(int(p["warmup_shifts"])):
        lam = sample_lambda(rng, p["lambda_mean"], p["alpha"], p["sigma"])
        n2 = qed_n2(lam, n1, p["h"], p["mu"], p["theta"], p["a"], p["c2"])
        base_seed = sample_seed + 10_000 * int(shift)
        stream = build_shift_stream_crn(
            lam, p["mu"], p["theta"],
            base_seed + 1,
            base_seed + 2,
            base_seed + 3,
        )
        state = simulate_shift(state, n1, n2, lam, p["mu"], p["theta"], p["c1"], p["c2"], p["h"], p["a"], base_seed + 4, stream, balk_threshold=p.get("balk_threshold"), balk_penalty=p.get("balk_penalty"))["final_state"]
    return {"sample_id": int(sample_id), "patients_state": state}


def parallel_map(fn: Any, tasks: list[Any], executor: Any | None) -> list[Any]:
    if executor is None or len(tasks) < 8:
        return [fn(task) for task in tasks]
    return list(executor.map(fn, tasks))


def build_warmup_pool(p: dict[str, Any], output: Path, executor: Any | None) -> list[dict[str, Any]]:
    pool = parallel_map(warmup_sample, [(i, p) for i in range(int(p["pool_size"]))], executor)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(pool, indent=2), encoding="utf-8")
    return pool


def build_dataset(p: dict[str, Any], pool_size: int, count: int, seed: int, tag: str, lambda_values: list[float] | None = None) -> list[dict[str, Any]]:
    if p.get("initialization", "warmup") not in ("warmup", "fixed_count", "count_mixture"):
        raise ValueError("Unknown initialization mode")
    rows = []
    for sid in range(int(count)):
        trials = []
        for rep in range(int(p["replications"])):
            base_seed = (namespaced_seed(int(p["seed_namespace"]), int(seed), sid, rep, 0)
                         if "seed_namespace" in p else int(seed) + 100_000 * int(sid) + 10 * int(rep))
            pool_rng = np.random.default_rng(base_seed + 4)
            trials.append({
                "pool_index": int(pool_rng.integers(0, pool_size)) if p.get("initialization", "warmup") == "warmup" else None,
                "arrival_seed": base_seed + 1,
                "service_seed": base_seed + 2,
                "patience_seed": base_seed + 3,
            })
        if lambda_values is None:
            lambda_seed = (namespaced_seed(int(p["seed_namespace"]), int(seed), sid, 0, 5)
                           if "seed_namespace" in p else int(seed) + 200_000_000 + int(sid))
            lambda_rng = np.random.default_rng(lambda_seed)
            lam = sample_lambda(lambda_rng, p["lambda_mean"], p["alpha"], p["sigma"])
        else:
            lam = float(lambda_values[sid])
        rows.append({"dataset_tag": tag, "scenario_id": sid, "lambda_fixed": lam, "trials": trials})
    return rows


def eval_total_task(task: tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], int]) -> dict[str, Any]:
    p, scenario, pool, total_servers = task
    vals = []
    initialization = p.get("initialization", "warmup")
    if initialization == "count_mixture":
        if p.get("record_terminal_counts", False):
            raise ValueError("Raw terminal counts require a single initial-state setting")
        counts = p["initial_patient_counts"]
        if not counts:
            raise ValueError("initial_patient_counts must not be empty")
        # Average costs/performance before optimizing, not separately optimized decisions.
        results = [eval_total_task(({**p, "initialization": "fixed_count", "initial_patient_count": count},
                                    scenario, pool, total_servers)) for count in counts]
        identifiers = {"scenario_id", "lambda_fixed", "total_servers"}
        return {key: value if key in identifiers else float(np.mean([r[key] for r in results]))
                for key, value in results[0].items()}
    if initialization not in ("warmup", "fixed_count"):
        raise ValueError("Unknown initialization mode")
    if initialization == "fixed_count":
        initial_count = int(p["initial_patient_count"])
        if initial_count < 0 or initial_count != p["initial_patient_count"]:
            raise ValueError("initial_patient_count must be a nonnegative integer")
    for trial in scenario["trials"]:
        if initialization == "fixed_count":
            # Reuse the unused pool-selection stream for initial residual services.
            rng = np.random.default_rng(int(trial["patience_seed"]) + 1)
            residual = p.get("initial_residual_distribution", "exponential")
            if p.get("service_distribution", "exponential") != "exponential" and "initial_residual_distribution" not in p:
                raise ValueError("Non-exponential service requires an explicit initial residual assumption")
            state = [{"has_started_service": True, "remaining_service_time": float(remaining)}
                     for remaining in service_samples(rng, p["mu"], residual, p.get("initial_residual_scv", 1.0), initial_count)]
        else:
            state = pool[int(trial["pool_index"])]["patients_state"]
        stream = build_model_stream(p, scenario["lambda_fixed"], trial["arrival_seed"], trial["service_seed"], trial["patience_seed"])
        vals.append(simulate_shift(state, total_servers, 0, scenario["lambda_fixed"], p["mu"], p["theta"], p["c1"], p["c2"], p["h"], p["a"], trial["service_seed"], stream, strict_stream=p.get("strict_stream", False), balk_threshold=p.get("balk_threshold"), balk_penalty=p.get("balk_penalty"), quadratic_queue_cost=p.get("quadratic_queue_cost")))
    keys = ("queue_area", "abandon_count", "arrival_count", "patient_count", "lwbs_risk_count", "avg_queue", "avg_wait_time", "lwbs_rate")
    if p.get("balk_threshold") is not None:
        keys += ("balk_count", "admitted_arrival_count", "balk_rate")
    if p.get("quadratic_queue_cost") is not None:
        keys += ("queue_squared_area",)
    out = {"scenario_id": int(scenario["scenario_id"]), "lambda_fixed": float(scenario["lambda_fixed"]), "total_servers": int(total_servers)}
    for key in keys:
        out[key] = float(np.mean([v[key] for v in vals]))
    if p.get("record_terminal_counts", False):
        out["terminal_counts"] = []
        for rep, value in enumerate(vals):
            terminal = value["final_state"]
            started = sum(bool(row["has_started_service"]) for row in terminal)
            out["terminal_counts"].append({"replication_id": rep, "terminal_count": len(terminal),
                                           "terminal_started_count": started,
                                           "terminal_unstarted_count": len(terminal) - started})
    return out


def init_perf_worker(p: dict[str, Any], dataset: list[dict[str, Any]], pool: list[dict[str, Any]]) -> None:
    global _WORKER_P, _WORKER_DATASET, _WORKER_POOL
    _WORKER_P = p
    _WORKER_DATASET = dataset
    _WORKER_POOL = pool


def eval_total_worker_task(task: tuple[int, int]) -> dict[str, Any]:
    sid, total = task
    if _WORKER_P is None or _WORKER_DATASET is None or _WORKER_POOL is None:
        raise RuntimeError("worker context is not initialized")
    return eval_total_task((_WORKER_P, _WORKER_DATASET[int(sid)], _WORKER_POOL, int(total)))


def ensure_perf(p: dict[str, Any], dataset: list[dict[str, Any]], pool: list[dict[str, Any]], cache: dict[tuple[int, int], dict[str, Any]], requests: list[tuple[int, int]], executor: Any | None) -> None:
    unique = list(dict.fromkeys((int(sid), int(total)) for sid, total in requests))
    missing = [(sid, total) for sid, total in unique if (sid, total) not in cache]
    if executor is None or len(missing) < 8:
        results = [eval_total_task((p, dataset[sid], pool, total)) for sid, total in missing]
    else:
        results = executor.map(eval_total_worker_task, missing)
    for result in results:
        cache[(int(result["scenario_id"]), int(result["total_servers"]))] = result


def waiting_cost(p: dict[str, Any], perf: dict[str, Any]) -> float:
    cost = p["h"] * perf["queue_area"]
    if p.get("quadratic_queue_cost") is not None:
        cost += p["quadratic_queue_cost"] * perf["queue_squared_area"]
    return float(cost)


def recourse_cost(p: dict[str, Any], perf: dict[str, Any], n2: int) -> float:
    if p.get("balk_penalty") is not None:
        return float(p["c2"] * int(n2) + waiting_cost(p, perf) + p["a"] * perf["abandon_count"] + p["balk_penalty"] * perf.get("balk_count", 0))
    return float(p["c2"] * int(n2) + waiting_cost(p, perf) + p["a"] * (perf["abandon_count"] + perf.get("balk_count", 0)))


def best_n2_batch(p: dict[str, Any], dataset: list[dict[str, Any]], n1: int, pool: list[dict[str, Any]], cache: dict[tuple[int, int], dict[str, Any]], executor: Any | None, memo: dict[tuple[int, int], int]) -> dict[int, int]:
    # Reuse performance within a policy; keep memo local to its cost/domain.
    states = []
    out = {}
    for s in dataset:
        sid = int(s["scenario_id"])
        key = (sid, int(n1))
        if key in memo:
            out[sid] = memo[key]
            continue
        current = qed_n2(s["lambda_fixed"], n1, p["h"], p["mu"], p["theta"], p["a"], p["c2"])
        upper = int(p["n2_upper"]) if "n2_upper" in p else max(int(p["n2_search_upper_floor"]), int(p["search_upper_multiplier"]) * max(1, int(current)))
        if not 0 <= current <= upper:
            raise ValueError(f"N2 start {current} outside [0, {upper}] for scenario {sid}, N1={n1}")
        states.append({"sid": sid, "current": int(current), "upper": upper})
    while states:
        requests = []
        for item in states:
            cands = sorted({max(0, item["current"] - 1), item["current"], min(item["upper"], item["current"] + 1)})
            item["cands"] = cands
            requests.extend((item["sid"], n1 + n2) for n2 in cands)
        ensure_perf(p, dataset, pool, cache, requests, executor)
        next_states = []
        for item in states:
            sid = item["sid"]
            costs = {n2: recourse_cost(p, cache[(sid, n1 + n2)], n2) for n2 in item["cands"]}
            best = min(item["cands"], key=lambda n2: (costs[n2], n2))
            if best == item["current"]:
                memo[(sid, int(n1))] = int(best)
                out[sid] = int(best)
            else:
                item["current"] = int(best)
                next_states.append(item)
        states = next_states
    return out


def objective_1so(p: dict[str, Any], n1: int, dataset: list[dict[str, Any]], pool: list[dict[str, Any]], cache: dict[tuple[int, int], dict[str, Any]], executor: Any | None) -> float:
    ensure_perf(p, dataset, pool, cache, [(s["scenario_id"], n1) for s in dataset], executor)
    vals = [recourse_cost(p, cache[(s["scenario_id"], n1)], 0) for s in dataset]
    return float(p["c1"] * int(n1) + np.mean(vals))


def objective_2so(p: dict[str, Any], n1: int, dataset: list[dict[str, Any]], pool: list[dict[str, Any]], cache: dict[tuple[int, int], dict[str, Any]], executor: Any | None, n2_memo: dict[tuple[int, int], int]) -> float:
    n2_by_sid = best_n2_batch(p, dataset, n1, pool, cache, executor, n2_memo)
    vals = []
    for s in dataset:
        sid = int(s["scenario_id"])
        n2 = n2_by_sid[sid]
        vals.append(recourse_cost(p, cache[(sid, n1 + n2)], n2))
    return float(p["c1"] * int(n1) + np.mean(vals))


def hill_climb(start: int, objective: Any, upper_multiplier: int, upper: int | None = None, lower: int = 1) -> dict[str, Any]:
    lo, hi = int(lower), max(1, int(upper_multiplier) * int(start)) if upper is None else int(upper)
    if upper is not None and not lo <= int(start) <= hi:
        raise ValueError(f"N1 start {start} outside [{lo}, {hi}]")
    current = min(hi, max(lo, int(start)))
    trace = []
    while True:
        cur = objective(current)
        left = objective(current - 1) if current > lo else None
        right = objective(current + 1) if current < hi else None
        trace.append({"n1": current, "objective": cur, "left_objective": left, "right_objective": right})
        best, best_val = current, cur
        if left is not None and left < best_val - 1e-12:
            best, best_val = current - 1, left
        if right is not None and right < best_val - 1e-12:
            best, best_val = current + 1, right
        if best == current:
            return {"best_n1": int(current), "best_objective": float(cur), "trace": trace}
        current = best


def scenario_row(p: dict[str, Any], label: str, perf: dict[str, Any], n1: int, n2: int) -> dict[str, Any]:
    base = p["c1"] * int(n1)
    surge = p["c2"] * int(n2)
    wait = waiting_cost(p, perf)
    abandon = p["a"] * perf["abandon_count"]
    balk = (p["a"] if p.get("balk_penalty") is None else p["balk_penalty"]) * perf.get("balk_count", 0)
    qos = wait + abandon + balk
    row = {
        "benchmark": label, "scenario_id": int(perf["scenario_id"]), "lambda_fixed": perf["lambda_fixed"], "n1": int(n1), "n2": int(n2),
        "base_cost": base, "surge_cost": surge, "qos_cost": qos, "staff_cost": base + surge, "waiting_cost": wait,
        "abandonment_cost": abandon, "recourse_cost": surge + qos, "queue_area": perf["queue_area"], "abandon_count": perf["abandon_count"],
        "arrival_count": perf["arrival_count"], "patient_count": perf["patient_count"], "lwbs_risk_count": perf["lwbs_risk_count"], "avg_queue": perf["avg_queue"],
        "avg_wait_time": perf["avg_wait_time"], "lwbs_rate": perf["lwbs_rate"], "total_cost": base + surge + qos,
    }
    if p.get("balk_threshold") is not None:
        row.update(balk_cost=balk, balk_count=perf["balk_count"], balk_rate=perf["balk_rate"],
                   admitted_arrival_count=perf["admitted_arrival_count"])
    if p.get("quadratic_queue_cost") is not None:
        row.update(queue_squared_area=perf["queue_squared_area"],
                   linear_waiting_cost=p["h"] * perf["queue_area"],
                   quadratic_waiting_cost=p["quadratic_queue_cost"] * perf["queue_squared_area"])
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [float(r["total_cost"]) for r in rows]
    mean, low, high = mean_ci(costs)
    result = {
        "N1": rows[0]["n1"], "AvgN2": float(np.mean([r["n2"] for r in rows])), "MeanTotalCost": mean, "CILow": low, "CIHigh": high,
        "MeanBaseCost": float(np.mean([r["base_cost"] for r in rows])), "MeanSurgeCost": float(np.mean([r["surge_cost"] for r in rows])),
        "MeanQoSCost": float(np.mean([r["qos_cost"] for r in rows])), "MeanStaffCost": float(np.mean([r["staff_cost"] for r in rows])),
        "MeanWaitingCost": float(np.mean([r["waiting_cost"] for r in rows])), "MeanAbandonmentCost": float(np.mean([r["abandonment_cost"] for r in rows])),
        "MeanAvgQueue": float(np.mean([r["avg_queue"] for r in rows])), "MeanAvgWaitTime": float(np.mean([r["avg_wait_time"] for r in rows])),
        "MeanLWBSRate": float(np.mean([r["lwbs_rate"] for r in rows])), "MeanQueueArea": float(np.mean([r["queue_area"] for r in rows])),
        "MeanAbandonCount": float(np.mean([r["abandon_count"] for r in rows])), "ScenarioCount": len(rows),
    }
    if "balk_count" in rows[0]:
        result.update(MeanBalkCount=float(np.mean([r["balk_count"] for r in rows])),
                      MeanBalkCost=float(np.mean([r["balk_cost"] for r in rows])),
                      MeanBalkRate=float(np.mean([r["balk_rate"] for r in rows])),
                      MeanAdmittedArrivalCount=float(np.mean([r["admitted_arrival_count"] for r in rows])))
    if "queue_squared_area" in rows[0]:
        result.update(MeanQueueSquaredArea=float(np.mean([r["queue_squared_area"] for r in rows])),
                      MeanLinearWaitingCost=float(np.mean([r["linear_waiting_cost"] for r in rows])),
                      MeanQuadraticWaitingCost=float(np.mean([r["quadratic_waiting_cost"] for r in rows])))
    return result


def write_terminal_diagnostics(output_root: Path, rows: list[dict[str, Any]]) -> None:
    summary = []
    for label in dict.fromkeys(row["benchmark"] for row in rows):
        group = [row for row in rows if row["benchmark"] == label]
        for metric in ("terminal_count", "terminal_started_count", "terminal_unstarted_count"):
            values = np.asarray([row[metric] for row in group])
            quantiles = np.percentile(values, [5, 25, 50, 75, 95])
            summary.append({"benchmark": label, "metric": metric, "level": "scenario_replication",
                            "sample_count": len(values), "mean": float(values.mean()),
                            **dict(zip(("p5", "p25", "median", "p75", "p95"), map(float, quantiles))),
                            "min": int(values.min()), "max": int(values.max())})
    write_csv(output_root / "terminal_counts.csv", rows)
    write_csv(output_root / "terminal_count_summary.csv", summary)


def run_single(output_root: Path, use_parallel: bool = True, params: dict[str, Any] | None = None, environment: tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    p = dict(params or numerical_params())
    lookup_enabled = bool(p.get("include_so_lookup", False))
    qed_lookup_only = bool(p.get("qed_lookup_only", False))
    if qed_lookup_only and not lookup_enabled:
        raise ValueError("qed_lookup_only requires include_so_lookup")
    direct_label = "2,SO-Direct" if lookup_enabled else "2,SO"
    so_label = "2,SO-Lookup" if qed_lookup_only else direct_label
    # Raw terminal states are needed only for final evaluation, not every search candidate.
    search_p = {**p, "record_terminal_counts": False} if p.get("record_terminal_counts", False) else p
    parallel_enabled = bool(use_parallel)
    output_root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    start = time.perf_counter()
    needs_warmup = p.get("initialization", "warmup") == "warmup"
    warmup_executor = ProcessPoolExecutor(max_workers=p.get("workers")) if parallel_enabled and environment is None and needs_warmup else None
    try:
        if environment is None:
            pool = build_warmup_pool(p, output_root / "warmup_pool.json", warmup_executor) if needs_warmup else []
            search = build_dataset(p, len(pool), p["search_scenarios"], p["search_seed"], "search")
            evl = build_dataset(p, len(pool), p["eval_scenarios"], p["eval_seed"], "eval")
            online = [] if qed_lookup_only else build_dataset(p, len(pool), p["eval_scenarios"], p["online_n2_seed"], "online_n2", [s["lambda_fixed"] for s in evl])
        else:
            pool, search, online, evl = environment
        timings["environment"] = time.perf_counter() - start
    finally:
        if warmup_executor is not None:
            warmup_executor.shutdown()
    n1_nv = None if qed_lookup_only else newsvendor_n1(p["lambda_mean"], p["c1"], p["h"], p["mu"], p["theta"], p["a"], p["sigma"], p["alpha"])
    n1_qed = qed_n1(p["lambda_mean"], p["c1"], p["c2"], p["h"], p["mu"], p["theta"], p["a"], p["sigma"], p["alpha"])
    n2_memo: dict[tuple[int, int], int] = {}
    one = None
    search_jobs = ([] if qed_lookup_only else [("1so", n1_nv)]) + [("2so", n1_qed)]
    for method, initial in search_jobs:
        # Each policy starts with empty performance/formula caches and fresh workers.
        search_cache: dict[tuple[int, int], dict[str, Any]] = {}
        eta_star.cache_clear()
        print(f"[Single] START offline {method}", flush=True)
        start = time.perf_counter()
        search_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_perf_worker, initargs=(search_p, search, pool)) if parallel_enabled else None
        try:
            objective = (lambda n: objective_1so(search_p, n, search, pool, search_cache, search_executor)) if method == "1so" else (
                lambda n: objective_2so(search_p, n, search, pool, search_cache, search_executor, n2_memo))
            result = hill_climb(initial, objective, p["search_upper_multiplier"], p.get("n1_upper"))
        finally:
            if search_executor is not None:
                search_executor.shutdown()
        timings[f"search_{method}"] = time.perf_counter() - start
        if method == "1so":
            one = result
        else:
            two = result
        print(f"[Single] END offline {method}, elapsed={timings[f'search_{method}']:.1f}s", flush=True)
    lookup = []
    if lookup_enabled:
        start = time.perf_counter()
        points = [{"source": "search", "scenario_id": int(s["scenario_id"]), "lambda_fixed": s["lambda_fixed"],
                   "n1": two["best_n1"], "n2": n2_memo[(int(s["scenario_id"]), two["best_n1"])]} for s in search]
        extra_count = int(p.get("lookup_scenarios", len(search))) - len(search)
        if extra_count < 0:
            raise ValueError("lookup_scenarios cannot be smaller than search_scenarios")
        if extra_count:
            extra = build_dataset(p, len(pool), extra_count, p["lookup_seed"], "lookup_extra")
            extra_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_perf_worker, initargs=(search_p, extra, pool)) if parallel_enabled else None
            try:
                extra_n2 = best_n2_batch(search_p, extra, two["best_n1"], pool, {}, extra_executor, {})
            finally:
                if extra_executor is not None:
                    extra_executor.shutdown()
            points.extend({"source": "lookup_extra", "scenario_id": int(s["scenario_id"]), "lambda_fixed": s["lambda_fixed"],
                           "n1": two["best_n1"], "n2": extra_n2[int(s["scenario_id"])]} for s in extra)
        timings["lookup_training"] = time.perf_counter() - start
        start = time.perf_counter()
        lookup = build_rate_lookup(points, int(p["lookup_bins"]))
        timings["lookup_construction"] = time.perf_counter() - start
        write_csv(output_root / "lookup_points.csv", points)
        write_csv(output_root / "lookup.csv", [{"n1": two["best_n1"], **row} for row in lookup])
    all_rows: list[dict[str, Any]] = []
    terminal_rows: list[dict[str, Any]] = []
    online_n2: dict[str, dict[int, int]] = {}
    online_times: dict[str, float] = {}
    online_jobs = [] if qed_lookup_only else [(direct_label, int(two["best_n1"]), "online_direct_n2")]
    if not qed_lookup_only and p.get("include_qed_n2_search", False):
        online_jobs.append(("QED-N1+SO-N2", n1_qed, "online_qed_n2"))
    for label, n1, timing_key in online_jobs:
        eta_star.cache_clear()
        print(f"[Single] START online {label}", flush=True)
        start = time.perf_counter()
        online_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_perf_worker, initargs=(search_p, online, pool)) if parallel_enabled else None
        try:
            online_n2[label] = best_n2_batch(search_p, online, n1, pool, {}, online_executor, {})
        finally:
            if online_executor is not None:
                online_executor.shutdown()
        timings[timing_key] = online_times[label] = time.perf_counter() - start
        print(f"[Single] END online {label}, elapsed={online_times[label]:.1f}s", flush=True)
    if lookup_enabled and not qed_lookup_only:
        timings["online_decisions"] = sum(online_times.values())
    if qed_lookup_only:
        policies = [("2,QED", n1_qed), ("2,SO-Lookup", two["best_n1"])]
    else:
        policies = [("1,NV", n1_nv)] if p.get("include_1nv", True) else []
        policies += [("1,SO", one["best_n1"]), ("2,QED", n1_qed)]
        if p.get("include_qed_n2_search", False):
            policies.append(("QED-N1+SO-N2", n1_qed))
        policies.append((direct_label, two["best_n1"]))
        if lookup_enabled:
            policies.append(("2,SO-Lookup", two["best_n1"]))
    evaluation_times: dict[str, float] = {}
    start = time.perf_counter()
    for label, n1 in policies:
        eta_star.cache_clear()
        eval_cache: dict[tuple[int, int], dict[str, Any]] = {}
        policy_start = time.perf_counter()
        eval_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_perf_worker, initargs=(p, evl, pool)) if parallel_enabled else None
        try:
            requests = []
            n2_by_sid = dict(online_n2.get(label, {}))
            if label == "2,SO-Lookup":
                n2_by_sid = {int(s["scenario_id"]): rate_lookup_n2(lookup, float(s["lambda_fixed"])) for s in evl}
            for s in evl:
                n2 = 0 if label in {"1,NV", "1,SO"} else (qed_n2(s["lambda_fixed"], n1, p["h"], p["mu"], p["theta"], p["a"], p["c2"]) if label == "2,QED" else n2_by_sid[int(s["scenario_id"])])
                n2_by_sid[s["scenario_id"]] = n2
                requests.append((s["scenario_id"], n1 + n2))
            ensure_perf(p, evl, pool, eval_cache, requests, eval_executor)
            for s in evl:
                n2 = n2_by_sid[s["scenario_id"]]
                perf = eval_cache[(s["scenario_id"], n1 + n2)]
                all_rows.append(scenario_row(p, label, perf, n1, n2))
                if p.get("record_terminal_counts", False):
                    terminal_rows.extend({"benchmark": label, "scenario_id": int(s["scenario_id"]),
                                          "lambda_fixed": s["lambda_fixed"], "n1": n1, "n2": n2,
                                          **counts} for counts in perf["terminal_counts"])
        finally:
            if eval_executor is not None:
                eval_executor.shutdown()
        evaluation_times[label] = time.perf_counter() - policy_start
    timings["evaluation"] = time.perf_counter() - start
    order = tuple(label for label, _n1 in policies)
    search_times = {"1,NV": 0.0, "1,SO": timings.get("search_1so", 0.0), "2,QED": 0.0, "QED-N1+SO-N2": 0.0, direct_label: timings["search_2so"]}
    if lookup_enabled:
        search_times["2,SO-Lookup"] = timings["search_2so"] + timings["lookup_training"] + timings["lookup_construction"]
    bench = []
    for label in order:
        row = {"Benchmark": label, "SearchTimeSec": search_times[label], **aggregate([r for r in all_rows if r["benchmark"] == label])}
        bench.append(row)
    start = time.perf_counter()
    write_csv(output_root / "benchmark_table.csv", bench)
    write_csv(output_root / "scenario_costs.csv", sorted(all_rows, key=lambda r: (order.index(r["benchmark"]), r["scenario_id"])))
    decomp = write_diagnostics(output_root, all_rows, order, so_label)
    if terminal_rows:
        write_terminal_diagnostics(output_root, terminal_rows)
    if lookup_enabled:
        write_csv(output_root / "paired_comparison.csv", paired_comparison(all_rows, order))
        write_csv(output_root / "policy_decisions.csv", [{key: row[key] for key in ("benchmark", "scenario_id", "lambda_fixed", "n1", "n2")} for row in all_rows])
    from figures import save_cost_decomposition, save_qos_ecdf

    save_cost_decomposition(decomp, output_root / "cost_decomposition.png")
    save_qos_ecdf(all_rows, output_root / "qos_ecdf.png")
    timings["outputs_and_figures"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - t0
    searches = {so_label: two} if one is None else {"1,SO": one, so_label: two}
    summary = {"config": p, "pool_size": len(pool), "paths": {"summary_json": str(output_root / "summary.json")}, "timings_sec": timings, "search": searches, "benchmarks": {r["Benchmark"]: r for r in bench}}
    summary["policy_timings_sec"] = {
        label: {"offline_search": search_times[label], "online_search": online_times.get(label, 0.0),
                "evaluation": evaluation_times[label]} for label in order
    }
    summary["timing_scope"] = {
        "cache_reuse": "Within each policy and stage only; separate performance caches, N2 memos and worker pools.",
        "elapsed_time": "Wall-clock seconds, including worker startup/dispatch/collection/shutdown; evaluation excludes online search.",
        "common_data": "Policies share scenario data and random seeds (CRN), not simulated performance results.",
        "direct_and_lookup": "Deployment variants of the same fitted SO policy; both are charged the full base-policy training time. Lookup also includes extra training and table construction.",
        "zero_search_time": "Zero means no simulation optimization in that phase, not a timed formula or lookup query.",
    }
    if p.get("balk_threshold") is not None:
        summary["balking"] = {
            "rule": "Reject a new arrival iff the live pre-arrival waiting count is strictly greater than balk_threshold.",
            "queue": "Includes interrupted patients waiting to resume; excludes active service. Initial patients never balk.",
            "cost": "balk_penalty per balk (defaults to a); a per post-admission abandonment. Reported separately and included in QoS and total cost.",
            "balk_penalty": p["a"] if p.get("balk_penalty") is None else p["balk_penalty"],
            "rates": "balk_rate = balks / all new arrivals; lwbs_rate = (balks + abandons) / initial unstarted patients plus all new arrivals. Replication ratios are macro-averaged.",
            "crn": "Rejected arrivals retain their position in the pre-generated service/patience streams.",
            "formula_policies": "NV/QED formulas are unchanged; SO searches and all evaluations include balking.",
        }
    if lookup_enabled:
        summary["lookup"] = {"n1": two["best_n1"], "training_points": len(points), "requested_bins": p["lookup_bins"],
                             "merged_bins": len(lookup), "intervals": "(lower, upper]; final (lower, +infinity); cover (0, +infinity)",
                             "rounding": "floor(mean + 0.5)", "monotonic_postprocessing": False}
    if ("n1_upper" in p and "n2_upper" in p) or p.get("record_search_bounds", False):
        bounds = []

        def record(stage: str, label: str, values: list[int], lower: int, upper: int | list[int]) -> None:
            uppers = [upper] * len(values) if isinstance(upper, int) else upper
            hits = sum(v == hi for v, hi in zip(values, uppers))
            bounds.append({"stage": stage, "benchmark": label, "lower": lower, "upper": upper if isinstance(upper, int) else "per_decision",
                           "decision_count": len(values), "lower_hit_count": sum(v == lower for v in values),
                           "upper_hit_count": hits, "zero_count": values.count(0),
                           "upper_hit_rate": hits / len(values)})
            if p.get("record_search_bounds", False):
                bounds[-1].update(upper_min=min(uppers), upper_max=max(uppers))

        def n2_uppers(data: list[dict[str, Any]], keys: list[tuple[int, int]]) -> int | list[int]:
            if "n2_upper" in p:
                return int(p["n2_upper"])
            return [max(int(p["n2_search_upper_floor"]), int(p["search_upper_multiplier"]) * max(1, qed_n2(data[sid]["lambda_fixed"], n1, p["h"], p["mu"], p["theta"], p["a"], p["c2"]))) for sid, n1 in keys]

        for label, result, initial in (("1,SO", one, n1_nv), (so_label, two, n1_qed)):
            if result is not None:
                record("n1_search_final", label, [result["best_n1"]], 1, int(p.get("n1_upper", p["search_upper_multiplier"] * initial)))
        record("n2_search_all_candidates", so_label, list(n2_memo.values()), 0, n2_uppers(search, list(n2_memo)))
        final_keys = [key for key in n2_memo if key[1] == two["best_n1"]]
        record("n2_search_final_n1", so_label, [n2_memo[key] for key in final_keys], 0, n2_uppers(search, final_keys))
        for label, values in online_n2.items():
            n1 = two["best_n1"] if label == direct_label else n1_qed
            record("n2_online", label, list(values.values()), 0, n2_uppers(online, [(sid, n1) for sid in values]))
        summary["boundary_diagnostics"] = bounds
        write_csv(output_root / "boundary_diagnostics.csv", bounds)
    write_json(output_root / "summary.json", summary)
    return summary


def write_diagnostics(output_root: Path, rows: list[dict[str, Any]], order: tuple[str, ...], so_label: str = "2,SO") -> list[dict[str, Any]]:
    decomp = []
    diag = []
    quant = []
    lambdas = sorted({r["lambda_fixed"] for r in rows})
    targets = {q: float(np.percentile(lambdas, q)) for q in (10, 50, 90)}
    for label in order:
        group = [r for r in rows if r["benchmark"] == label]
        total = float(np.mean([r["total_cost"] for r in group]))
        staff = float(np.mean([r["staff_cost"] for r in group]))
        qos = float(np.mean([r["qos_cost"] for r in group]))
        decomp.append({"benchmark": label, "mean_total_cost": total, "mean_base_cost": float(np.mean([r["base_cost"] for r in group])), "mean_surge_cost": float(np.mean([r["surge_cost"] for r in group])), "mean_waiting_cost": float(np.mean([r["waiting_cost"] for r in group])), "mean_abandonment_cost": float(np.mean([r["abandonment_cost"] for r in group])), "mean_staff_cost": staff, "mean_qos_cost": qos, "staff_cost_share": staff / total, "qos_cost_share": qos / total})
        if "balk_cost" in group[0]:
            decomp[-1]["mean_balk_cost"] = float(np.mean([r["balk_cost"] for r in group]))
        diag.append({"benchmark": label, "n2_p95": float(np.percentile([r["n2"] for r in group], 95)), "n2_max": max(r["n2"] for r in group), "cost_cv": float(np.std([r["total_cost"] for r in group]) / total), "queue_p95": float(np.percentile([r["avg_queue"] for r in group], 95)), "wait_p95": float(np.percentile([r["avg_wait_time"] for r in group], 95)), "lwbs_p95": float(np.percentile([r["lwbs_rate"] for r in group], 95)), "scenario_count": len(group)})
    for q, target in targets.items():
        chosen = min(lambdas, key=lambda x: (abs(x - target), x))
        for row in rows:
            if abs(row["lambda_fixed"] - chosen) <= 1e-9:
                quant.append({"demand_quantile": q / 100.0, "lambda_fixed": chosen, **row})
    paired = {int(row["scenario_id"]): {} for row in rows if row["benchmark"] == "2,QED"}
    for row in rows:
        if row["benchmark"] in {"2,QED", so_label}:
            paired[int(row["scenario_id"])][row["benchmark"]] = row
    scenarios = sorted(paired.values(), key=lambda pair: pair["2,QED"]["lambda_fixed"])
    quartiles = []
    groups = [*zip(("Lowest 25%", "Second 25%", "Third 25%", "Highest 25%"), np.array_split(scenarios, 4)), ("Overall", scenarios)]
    for label, group in groups:
        savings = [pair["2,QED"]["total_cost"] - pair[so_label]["total_cost"] for pair in group]
        wins = sum(x > 0.0 for x in savings)
        quartiles.append({"demand_quartile": label, "scenario_count": len(group), "2so_win_count": wins, "2so_win_rate_pct": round(100.0 * wins / len(group), 1), "mean_cost_saving": round(float(np.mean(savings)), 2), "avg_n2_2qed": round(float(np.mean([pair["2,QED"]["n2"] for pair in group])), 2), "avg_n2_2so": round(float(np.mean([pair[so_label]["n2"] for pair in group])), 2)})
    write_csv(output_root / "cost_decomposition.csv", decomp)
    write_csv(output_root / "diagnostics_table.csv", diag)
    write_csv(output_root / "demand_quantile_table.csv", quant)
    write_csv(output_root / "demand_quartile_comparison.csv", quartiles)
    return decomp


def sensitivity_specs() -> list[tuple[str, str, tuple[float, ...]]]:
    return [
        ("h_a", "h", (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)),
        ("c2", "c2", (1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8)),
        ("alpha", "alpha", (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)),
        ("x_variance", "x_variance", (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)),
    ]


def run_sensitivity(output_root: Path, use_parallel: bool = True, params: dict[str, Any] | None = None, specs: list[tuple[str, str, tuple[float, ...]]] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    rows = []
    base = dict(params or numerical_params())
    sweeps = sensitivity_specs() if specs is None else specs
    executor = ProcessPoolExecutor(max_workers=base.get("workers")) if use_parallel else None
    try:
        shared_pool = build_warmup_pool(base, output_root / "shared" / "warmup_pool.json", executor)
        shared_search = build_dataset(base, len(shared_pool), base["search_scenarios"], base["search_seed"], "search")
        shared_eval = build_dataset(base, len(shared_pool), base["eval_scenarios"], base["eval_seed"], "eval")
        shared_online = build_dataset(base, len(shared_pool), base["eval_scenarios"], base["online_n2_seed"], "online_n2", [s["lambda_fixed"] for s in shared_eval])
        shared_env = (shared_pool, shared_search, shared_online, shared_eval)
        if executor is not None:
            executor.shutdown()
            executor = None
        for name, param, values in sweeps:
            for value in values:
                p = dict(base)
                if name == "h_a":
                    p["h"], p["a"] = float(value), 2.0 * float(value)
                elif name == "c2":
                    p["c2"] = float(value)
                elif name == "alpha":
                    p["alpha"] = float(value)
                else:
                    p["sigma"] = math.sqrt(float(value))
                point = output_root / name / str(value).replace(".", "p")
                summary = run_single(point, use_parallel, p, shared_env if name in {"h_a", "c2"} else None)
                for label, agg in summary["benchmarks"].items():
                    rows.append({
                        "sweep": name, "parameter": param, "value": float(value), "benchmark": label,
                        "mean_total_cost": agg["MeanTotalCost"], "mean_base_cost": agg["MeanBaseCost"],
                        "mean_surge_cost": agg["MeanSurgeCost"], "mean_waiting_cost": agg["MeanWaitingCost"],
                        "mean_abandonment_cost": agg["MeanAbandonmentCost"], "mean_staff_cost": agg["MeanStaffCost"],
                        "mean_qos_cost": agg["MeanQoSCost"], "mean_avg_queue": agg["MeanAvgQueue"],
                        "mean_avg_wait_time": agg["MeanAvgWaitTime"], "mean_lwbs_rate": agg["MeanLWBSRate"],
                        "ci95_lower": agg["CILow"], "ci95_upper": agg["CIHigh"],
                    })
        variance_table = []
        for value in sorted({row["value"] for row in rows if row["sweep"] == "x_variance"}):
            costs = {row["benchmark"]: row["mean_total_cost"] for row in rows if row["sweep"] == "x_variance" and row["value"] == value}
            two_so = costs["2,SO"]
            variance_table.append({"x_variance": value, "cost_1nv": round(costs["1,NV"], 3), "cost_1so": round(costs["1,SO"], 3), "cost_2qed": round(costs["2,QED"], 3), "cost_2so": round(two_so, 3), "saving_vs_1nv_pct": round(100.0 * (costs["1,NV"] - two_so) / costs["1,NV"], 2), "saving_vs_1so_pct": round(100.0 * (costs["1,SO"] - two_so) / costs["1,SO"], 2), "saving_vs_2qed_pct": round(100.0 * (costs["2,QED"] - two_so) / costs["2,QED"], 2)})
        write_csv(output_root / "sensitivity_table.csv", rows)
        write_csv(output_root / "variance_comparison_table.csv", variance_table)
        from figures import save_sensitivity

        save_sensitivity(rows, output_root)
        summary = {"base_config": base, "sweeps": [{"name": n, "parameter": p, "values": v} for n, p, v in sweeps], "paths": {"sensitivity_table_csv": str(output_root / "sensitivity_table.csv"), "variance_comparison_table_csv": str(output_root / "variance_comparison_table.csv")}, "total_runtime_seconds": time.perf_counter() - t0}
        write_json(output_root / "summary.json", summary)
        return summary
    finally:
        if executor is not None:
            executor.shutdown()


USE_PARALLEL = True
OUTPUT_ROOT = Path(__file__).resolve().parent / "output" / "numerical"
RUN_SINGLE = True
RUN_SENSITIVITY = True


def run() -> None:
    if RUN_SINGLE:
        run_single(OUTPUT_ROOT / "single", USE_PARALLEL)
    if RUN_SENSITIVITY:
        run_sensitivity(OUTPUT_ROOT / "sensitivity", USE_PARALLEL)


if __name__ == "__main__":
    run()
