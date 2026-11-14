from __future__ import annotations

import math
import json
import time
import os
import csv
import hashlib
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, qmc

from formulas import adjusted_n1_vec, adjusted_n2, eta_star, newsvendor_n1, qed_n1, qed_n2, round_half_up
from numerical import mean_ci, namespaced_seed, parallel_map, write_csv, write_json
from simulation import build_model_stream, simulate_shift

_CASE_P: dict[str, Any] | None = None
_CASE_SCENARIOS: dict[int, dict[str, Any]] | None = None
_CASE_WEEK_CACHE: OrderedDict[tuple[Any, ...], float] | None = None
_CASE_PREFIX_CACHE: OrderedDict[tuple[Any, ...], tuple[list[dict[str, Any]], float, float]] | None = None

def init_case_worker(p: dict[str, Any], dataset: list[dict[str, Any]]) -> None:
    global _CASE_P, _CASE_SCENARIOS, _CASE_WEEK_CACHE, _CASE_PREFIX_CACHE
    eta_star.cache_clear()
    _CASE_P = p
    _CASE_SCENARIOS = {int(s["scenario_id"]): s for s in dataset}
    _CASE_WEEK_CACHE = OrderedDict()
    _CASE_PREFIX_CACHE = OrderedDict()


def case_worker_context() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if _CASE_P is None or _CASE_SCENARIOS is None:
        raise RuntimeError("case worker context is not initialized")
    return _CASE_P, _CASE_SCENARIOS


def case_parallel_map(fn: Any, tasks: list[Any], executor: Any | None) -> list[Any]:
    if executor is None:
        return [fn(task) for task in tasks]
    return list(executor.map(fn, tasks))


def lru_put(cache: OrderedDict[Any, Any], key: Any, value: Any, limit: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(limit):
        cache.popitem(last=False)


def case_params() -> dict[str, Any]:
    return {
        "lambda_means": (
            141.019, 89.462, 207.385, 97.058, 188.769, 97.769, 186.942,
            93.711, 185.208, 95.189, 175.173, 96.692, 147.058, 94.115,
        ),
        "horizon": 14, "alpha": 0.769, "rho": 0.3, "sigma2": 0.118,
        "mu": 1.4, "theta": 12.0 / 27.5, "c1": 540.0, "c2": 810.0,
        "h": 810.0, "a": 1620.0, "xi1": 5.0, "xi2": 1.0,
        "search_scenarios": 100, "search_replications": 10,
        "eval_scenarios": 100, "eval_replications": 100,
        "outer_limit": 14, "inner_limit": 28,
        "seed_namespace": 2026102811, "search_seed": 1,
        "eval_recourse_seed": 3, "eval_seed": 2,
        "lhs_scenarios": 3900, "lhs_replications": 10,
        "lhs_seed": 5, "lhs_stream_base": 4,
        "lhs_bin_counts": (20,), "lhs_batch_size": 50,
        "n2_search_upper_floor": 20,
        "initial_patient_count": 0,
        "workers": max(1, (os.cpu_count() or 2) // 2),
    }


def lambda_path(p: dict[str, Any], seed: int) -> list[float]:
    rng = np.random.default_rng(int(seed))
    means = tuple(float(x) for x in p["lambda_means"])
    sd = math.sqrt(float(p["sigma2"]))
    eps_sd = math.sqrt((1.0 - float(p["rho"]) ** 2) * float(p["sigma2"]))
    while True:
        x = float(rng.normal(0.0, sd))
        vals = []
        ok = True
        for i, mean in enumerate(means):
            if i > 0:
                x = float(p["rho"]) * x + float(rng.normal(0.0, eps_sd))
            lam = mean + x * mean ** float(p["alpha"])
            vals.append(float(lam))
            ok = ok and lam > 0.0
        if ok:
            return vals


def n1_vectors_for_params(p: dict[str, Any]) -> tuple[list[int], list[int], list[int]]:
    # Arrival-rate X already includes mu**(1-alpha); staffing uses Hu's X.
    sigma = math.sqrt(float(p["sigma2"])) / float(p["mu"]) ** (1.0 - float(p["alpha"]))
    n1_nv = [newsvendor_n1(lam, p["c1"], p["h"], p["mu"], p["theta"], p["a"], sigma, p["alpha"]) for lam in p["lambda_means"]]
    n1_qed = [qed_n1(lam, p["c1"], p["c2"], p["h"], p["mu"], p["theta"], p["a"], sigma, p["alpha"]) for lam in p["lambda_means"]]
    return n1_nv, n1_qed, adjusted_n1_vec(tuple(p["lambda_means"]), n1_qed, p["mu"], p["theta"], p["xi1"])


def case_initial_state(p: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    count = int(p.get("initial_patient_count", 0))
    if count < 0 or count != p.get("initial_patient_count", 0):
        raise ValueError("initial_patient_count must be a nonnegative integer")
    rng = np.random.default_rng(int(seed) + 4)
    return [{"has_started_service": True, "remaining_service_time": float(remaining)}
            for remaining in rng.exponential(1.0 / float(p["mu"]), size=count)]


def case_shift(p: dict[str, Any], state: list[dict[str, Any]], n1: int, n2: int, lam: float, base: int, shift: int) -> dict[str, Any]:
    if not 0 <= shift < 1024:
        raise ValueError("Case stream layout reserves 1024 shifts per replication")
    seed = base + 256 * shift
    stream = build_model_stream(p, lam, seed + 1, seed + 2, seed + 3)
    return simulate_shift(state, n1, n2, lam, p["mu"], p["theta"], p["c1"], p["c2"], p["h"], p["a"], seed, stream, strict_stream=True)


def run_case_path(p: dict[str, Any], initial_state: list[dict[str, Any]], n1_vec: list[int], n2_rule: Any, lambda_vec: list[float], seed: int) -> tuple[dict[str, Any], list[int]]:
    state = [dict(row) for row in initial_state]
    n2_vec: list[int] = []
    totals = {k: 0.0 for k in ("total_cost", "base_cost", "surge_cost", "waiting_cost", "abandonment_cost", "queue_area", "abandon_count", "arrival_count", "wait_over_10min_count")}
    totals["wait_over_10min_count"] = float(sum(bool(row.get("wait_over_10min_counted")) for row in initial_state))
    patient_count = float(len(state))
    risk_count = float(sum(1 for row in state if not row.get("has_started_service")))
    for shift, lam in enumerate(lambda_vec):
        phase = shift % len(n1_vec)
        n2 = max(0, int(n2_rule(phase, float(lam), state)))
        n2_vec.append(n2)
        result = case_shift(p, state, n1_vec[phase], n2, float(lam), seed, shift)
        for key in totals:
            totals[key] += float(result[key])
        patient_count += float(result["arrival_count"])
        risk_count += float(result["arrival_count"])
        state = result["final_state"]
    metrics = dict(totals)
    metrics.update({
        "patient_count": patient_count, "lwbs_risk_count": risk_count,
        "avg_queue": totals["queue_area"] / max(1, len(lambda_vec)),
        "avg_wait_time": totals["queue_area"] / patient_count if patient_count > 0 else 0.0,
        "wait_over_10min_rate": totals["wait_over_10min_count"] / patient_count if patient_count > 0 else 0.0,
        "lwbs_rate": totals["abandon_count"] / risk_count if risk_count > 0 else 0.0,
    })
    return metrics, n2_vec


def build_case_dataset(p: dict[str, Any], count: int, reps: int, seed: int, tag: str, lambda_paths: list[list[float]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for sid in range(int(count)):
        trials = []
        for rep in range(int(reps)):
            base = namespaced_seed(int(p["seed_namespace"]), seed, sid, rep * 1024, 0)
            trials.append({"replication_id": rep, "seed": base})
        rate_seed = namespaced_seed(int(p["seed_namespace"]), seed, sid, 0, 5)
        path = lambda_path(p, rate_seed) if lambda_paths is None else list(map(float, lambda_paths[sid]))
        rows.append({"dataset_tag": tag, "scenario_id": sid, "lambda_vec": path, "trials": trials})
    return rows


def lhs_paths(p: dict[str, Any], count: int, seed: int) -> list[list[float]]:
    z_rows = norm.ppf(np.clip(qmc.LatinHypercube(d=int(p["horizon"]), seed=seed).random(count), 1e-12, 1 - 1e-12))
    sd = math.sqrt(float(p["sigma2"]))
    innovation_sd = math.sqrt((1.0 - float(p["rho"]) ** 2) * float(p["sigma2"]))
    paths = []
    for z in z_rows:
        x, path = sd * float(z[0]), []
        for shift, mean in enumerate(map(float, p["lambda_means"])):
            if shift:
                x = float(p["rho"]) * x + innovation_sd * float(z[shift])
            path.append(mean + x * mean ** float(p["alpha"]))
        if min(path) <= 0.0:
            raise ValueError("LHS produced a nonpositive arrival rate")
        paths.append(path)
    return paths


def build_lhs_dataset(p: dict[str, Any]) -> list[dict[str, Any]]:
    seed = namespaced_seed(int(p["seed_namespace"]), p["lhs_seed"], 0, 0, 5)
    paths = lhs_paths(p, p["lhs_scenarios"], seed)
    rows = build_case_dataset(p, p["lhs_scenarios"], p["lhs_replications"], p["lhs_stream_base"], "lookup_lhs", paths)
    for offset, row in enumerate(rows):
        row["scenario_id"] = 100 + offset
    return rows


def point_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(row["source"]), "scenario_id": int(row["scenario_id"]), "shift": int(row["shift"]),
        "lambda_fixed": float(row["lambda_fixed"]), "n1_shift": int(row["n1_shift"]),
        "n2_pathwise": int(row["n2_pathwise"]), "objective_value": float(row["objective_value"]),
        "replications_used": int(row["replications_used"]),
    }


def load_lhs_checkpoint(path: Path, dataset: list[dict[str, Any]], n1_vec: list[int], reps: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [point_row(row) for row in csv.DictReader(handle)]
    scenarios = {int(s["scenario_id"]): s for s in dataset}
    counts: dict[int, int] = {}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        sid, shift = row["scenario_id"], row["shift"]
        if sid not in scenarios or not 0 <= shift < int(len(n1_vec)) or (sid, shift) in seen:
            raise ValueError("LHS checkpoint belongs to another experiment")
        seen.add((sid, shift))
        valid = row["n1_shift"] == n1_vec[shift] and row["replications_used"] == reps
        valid = valid and math.isclose(row["lambda_fixed"], scenarios[sid]["lambda_vec"][shift], abs_tol=1e-12)
        if not valid:
            raise ValueError("LHS checkpoint belongs to another experiment")
        counts[sid] = counts.get(sid, 0) + 1
    complete = {sid for sid, count in counts.items() if count == len(n1_vec)}
    return [row for row in rows if row["scenario_id"] in complete]


def save_lhs_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, rows)
    temporary.replace(path)


def path_cost(p: dict[str, Any], scenario: dict[str, Any], n1_vec: list[int], n2_vec: list[int], reps: int, week_cache: OrderedDict[Any, Any], prefix_cache: OrderedDict[Any, Any]) -> float:
    total_staff = tuple(int(n1) + int(n2) for n1, n2 in zip(n1_vec, n2_vec))
    trials = scenario["trials"][: int(reps)]
    context = (tuple(map(float, scenario["lambda_vec"])), float(p["mu"]), float(p["theta"]), float(p["h"]), float(p["a"]))
    context += tuple(p.get(key) for key in ("arrival_distribution", "arrival_scv", "service_distribution", "service_scv"))
    context += (int(p.get("initial_patient_count", 0)),)
    week_key = (scenario["dataset_tag"], int(scenario["scenario_id"]), context, tuple(int(t["seed"]) for t in trials), total_staff)
    qos_cost = week_cache.get(week_key)
    if qos_cost is not None:
        week_cache.move_to_end(week_key)
    else:
        values = []
        for trial in trials:
            base = (scenario["dataset_tag"], int(scenario["scenario_id"]), context, int(trial["seed"]))
            state = case_initial_state(p, int(trial["seed"]))
            queue_area = abandon_count = 0.0
            start = 0
            for end in range(len(total_staff), 0, -1):
                cached = prefix_cache.get((*base, total_staff[:end]))
                if cached is not None:
                    prefix_cache.move_to_end((*base, total_staff[:end]))
                    state, queue_area, abandon_count = [dict(row) for row in cached[0]], cached[1], cached[2]
                    start = end
                    break
            for shift in range(start, len(total_staff)):
                result = case_shift(p, state, total_staff[shift], 0, float(scenario["lambda_vec"][shift]), int(trial["seed"]), shift)
                queue_area += float(result["queue_area"])
                abandon_count += float(result["abandon_count"])
                state = result["final_state"]
                lru_put(prefix_cache, (*base, total_staff[: shift + 1]), ([dict(row) for row in state], queue_area, abandon_count), 1_000)
            values.append(float(p["h"]) * queue_area + float(p["a"]) * abandon_count)
        qos_cost = float(np.mean(values))
        lru_put(week_cache, week_key, qos_cost, 20_000)
    return float(p["c1"]) * sum(map(int, n1_vec)) + float(p["c2"]) * sum(map(int, n2_vec)) + float(qos_cost)


def one_so_worker_task(task: tuple[int, list[int], int]) -> float:
    sid, n1, reps = task
    p, scenarios = case_worker_context()
    return path_cost(p, scenarios[sid], n1, [0] * len(n1), reps,
                     _CASE_WEEK_CACHE, _CASE_PREFIX_CACHE)


def search_1so(p: dict[str, Any], dataset: list[dict[str, Any]], n1_start: list[int], reps: int, executor: Any | None) -> tuple[list[int], list[dict[str, Any]]]:
    current, upper, cache = list(n1_start), [2 * x for x in n1_start], {}
    week_cache, prefix_cache = OrderedDict(), OrderedDict()

    def objective(n1: list[int]) -> float:
        key = tuple(n1)
        if key not in cache:
            if executor is None:
                values = [path_cost(p, s, n1, [0] * len(n1), reps, week_cache, prefix_cache) for s in dataset]
            else:
                values = case_parallel_map(one_so_worker_task, [(int(s["scenario_id"]), n1, reps) for s in dataset], executor)
            cache[key] = float(np.mean(values))
        return cache[key]

    started = time.perf_counter()
    cost, trace = objective(current), []
    for step in range(int(p["outer_limit"])):
        shift = step % int(p["horizon"])
        before, n1_before, moves = cost, current[shift], 0
        while True:
            best, best_cost = list(current), cost
            for candidate in sorted({max(1, current[shift] - 1), current[shift], min(upper[shift], current[shift] + 1)}):
                trial = list(current)
                trial[shift] = candidate
                trial_cost = objective(trial)
                if trial_cost < best_cost - 1e-9:
                    best, best_cost = trial, trial_cost
            if best_cost >= cost - 1e-9:
                break
            current, cost = best, best_cost
            moves += 1
        trace.append({"coordinate": step + 1, "shift": shift + 1, "n1_before": n1_before,
                      "n1_after": current[shift], "n1_vec": str(current), "moves": moves,
                      "objective_before": before, "objective_after": cost,
                      "elapsed_sec": time.perf_counter() - started})
        print(f"[Case 1,SO] coordinate {step + 1}/{p['outer_limit']}, moves={moves}, cost={cost:.4f}", flush=True)
    return current, trace


def pathwise_task(task: tuple[dict[str, Any], dict[str, Any], list[int], int, list[int] | None]) -> tuple[list[int], float]:
    return pathwise_n2_search(*task)


def pathwise_worker_task(task: tuple[int, list[int], int, list[int] | None]) -> tuple[list[int], float]:
    sid, n1_vec, reps, initial_n2 = task
    p, scenarios = case_worker_context()
    return pathwise_n2_search(p, scenarios[int(sid)], n1_vec, int(reps), initial_n2)


def pathwise_n2_search(p: dict[str, Any], scenario: dict[str, Any], n1_vec: list[int], reps: int, initial_n2: list[int] | None = None, round_trace: list[tuple[list[int], float]] | None = None) -> tuple[list[int], float]:
    qed_start = [qed_n2(lam, n1_vec[i], p["h"], p["mu"], p["theta"], p["a"], p["c2"]) for i, lam in enumerate(scenario["lambda_vec"])]
    current = list(qed_start if initial_n2 is None else initial_n2)
    upper = [max(int(p["n2_search_upper_floor"]), 2 * max(1, qed, n2)) for qed, n2 in zip(qed_start, current)]
    cost_cache: dict[tuple[int, ...], float] = {}
    week_cache = _CASE_WEEK_CACHE if _CASE_WEEK_CACHE is not None else OrderedDict()
    prefix_cache = _CASE_PREFIX_CACHE if _CASE_PREFIX_CACHE is not None else OrderedDict()

    def cost(n2_vec: list[int]) -> float:
        key = tuple(map(int, n2_vec))
        if key not in cost_cache:
            cost_cache[key] = path_cost(p, scenario, n1_vec, n2_vec, reps, week_cache, prefix_cache)
        return cost_cache[key]

    current_cost = cost(current)
    for step in range(int(p["inner_limit"])):
        i = step % int(p["horizon"])
        while True:
            best_vec, best_cost = list(current), current_cost
            for cand in sorted({max(0, current[i] - 1), current[i], min(upper[i], current[i] + 1)}):
                trial = list(current)
                trial[i] = cand
                trial_cost = cost(trial)
                if trial_cost < best_cost - 1e-9:
                    best_vec, best_cost = trial, trial_cost
            if best_cost < current_cost - 1e-9:
                current, current_cost = best_vec, best_cost
            else:
                break
        if round_trace is not None and (step + 1) % int(p["horizon"]) == 0:
            round_trace.append((list(current), float(current_cost)))
    return current, float(current_cost)


def jso_objective(p: dict[str, Any], dataset: list[dict[str, Any]], n1_vec: list[int], reps: int, executor: Any | None, starts: list[list[int]] | None = None) -> tuple[float, list[tuple[list[int], float]]]:
    if executor is None:
        tasks = [(p, s, n1_vec, reps, None if starts is None else starts[i]) for i, s in enumerate(dataset)]
        results = parallel_map(pathwise_task, tasks, executor)
    else:
        tasks = [(int(s["scenario_id"]), n1_vec, reps, None if starts is None else starts[i]) for i, s in enumerate(dataset)]
        results = case_parallel_map(pathwise_worker_task, tasks, executor)
    return float(np.mean([cost for _, cost in results])), results


def jso_search(p: dict[str, Any], dataset: list[dict[str, Any]], n1_start: list[int], reps: int, executor: Any | None) -> tuple[list[int], list[tuple[list[int], float]], list[dict[str, Any]]]:
    current = list(n1_start)
    uppers = [max(2, 2 * max(1, int(x))) for x in n1_start]
    candidate_cache: dict[tuple[int, ...], tuple[float, list[tuple[list[int], float]]]] = {}

    def evaluate(n1_vec: list[int], starts: list[tuple[list[int], float]] | None) -> tuple[float, list[tuple[list[int], float]]]:
        key = tuple(map(int, n1_vec))
        if key not in candidate_cache:
            initial = None if starts is None else [list(n2) for n2, _cost in starts]
            candidate_cache[key] = jso_objective(p, dataset, n1_vec, reps, executor, initial)
        return candidate_cache[key]

    current_cost, current_results = evaluate(current, None)
    trace = []
    started = time.perf_counter()
    for step in range(int(p["outer_limit"])):
        i = step % int(p["horizon"])
        coordinate_n1 = int(current[i])
        coordinate_cost = float(current_cost)
        local_move = 0
        while True:
            best_vec, best_cost = list(current), current_cost
            best_results = current_results
            row = {"step": step, "local_move": local_move, "shift": i, "current_cost": current_cost}
            for cand in sorted({max(1, current[i] - 1), current[i], min(uppers[i], current[i] + 1)}):
                trial = list(current)
                trial[i] = cand
                trial_cost, trial_results = evaluate(trial, current_results)
                row[f"n1_{cand}"] = trial_cost
                if trial_cost < best_cost - 1e-9:
                    best_vec, best_cost, best_results = trial, trial_cost, trial_results
            if best_cost < current_cost - 1e-9:
                trace.append(row)
                current, current_cost, current_results = best_vec, best_cost, best_results
                local_move += 1
            else:
                row.update({
                    "coordinate_end": True, "n1_before": coordinate_n1, "n1_after": int(current[i]),
                    "n1_vec": list(current), "objective_before": coordinate_cost,
                    "objective_after": float(current_cost), "moves": local_move,
                    "elapsed_sec": time.perf_counter() - started,
                })
                trace.append(row)
                break
        print(f"[JSO] coordinate {step + 1}/{p['outer_limit']}: shift={i + 1}, moves={local_move}, cost={current_cost:.4f}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return current, current_results, trace


def point_rows(source: str, dataset: list[dict[str, Any]], n1_vec: list[int], reps: int, results: list[tuple[list[int], float]]) -> list[dict[str, Any]]:
    rows = []
    for scenario, (n2_vec, cost) in zip(dataset, results):
        for shift, n2 in enumerate(n2_vec):
            rows.append({"source": source, "scenario_id": int(scenario["scenario_id"]), "shift": shift, "lambda_fixed": scenario["lambda_vec"][shift], "n1_shift": n1_vec[shift], "n2_pathwise": n2, "objective_value": cost, "replications_used": reps})
    return rows


def collect_points(p: dict[str, Any], source: str, dataset: list[dict[str, Any]], n1_vec: list[int], reps: int, executor: Any | None) -> list[dict[str, Any]]:
    if executor is None:
        tasks = [(p, s, n1_vec, reps, None) for s in dataset]
        results = parallel_map(pathwise_task, tasks, executor)
    else:
        tasks = [(int(s["scenario_id"]), n1_vec, reps, None) for s in dataset]
        results = case_parallel_map(pathwise_worker_task, tasks, executor)
    return point_rows(source, dataset, n1_vec, reps, results)


def build_lookup(points: list[dict[str, Any]], bins: int, policy: str) -> list[dict[str, Any]]:
    order = {"jso_search": 0, "lhs_extra": 1}
    rows = []
    for shift in sorted({int(r["shift"]) for r in points}):
        group = sorted([r for r in points if int(r["shift"]) == shift], key=lambda r: (r["lambda_fixed"], order.get(r["source"], 9), r["scenario_id"]))
        if int(bins) < 1 or int(bins) > len(group):
            raise ValueError(f"bins must be between 1 and {len(group)} for shift {shift}")
        chunks = [[dict(x) for x in chunk.tolist()] for chunk in np.array_split(group, int(bins))]
        cuts = [(chunks[i][-1]["lambda_fixed"] + chunks[i + 1][0]["lambda_fixed"]) / 2.0 for i in range(len(chunks) - 1)]
        for bin_id, vals in enumerate(chunks):
            n2_mean = float(np.mean([v["n2_pathwise"] for v in vals]))
            lam = [float(v["lambda_fixed"]) for v in vals]
            lower = min(lam) if bin_id == 0 else cuts[bin_id - 1]
            upper = max(lam) if bin_id == len(chunks) - 1 else cuts[bin_id]
            rows.append({"policy": policy, "shift": shift, "bin_id": bin_id, "lambda_lower": lower, "lambda_center": float(np.mean(lam)), "lambda_upper": upper, "source_count": len(vals), "n2_mean": n2_mean, "n2": max(0, int(math.floor(n2_mean + 0.5)))})
    return rows


def lookup_n2(lookup: list[dict[str, Any]], shift: int, lam: float) -> int:
    rows = [r for r in lookup if int(r["shift"]) == int(shift)]
    inside = [r for r in rows if float(r["lambda_lower"]) <= float(lam) <= float(r["lambda_upper"])]
    if inside:
        return int(min(inside, key=lambda r: int(r["bin_id"]))["n2"])
    return int(min(rows, key=lambda r: (abs(float(r["lambda_center"]) - float(lam)), float(r["lambda_center"])))["n2"])


def eval_policy_task(task: tuple[dict[str, Any], str, dict[str, Any], list[int], list[int] | None, list[dict[str, Any]] | None, int]) -> dict[str, Any]:
    p, label, scenario, n1_vec, fixed_n2, lookup, reps = task
    vals, n2_rows = [], []
    _, n1_qed, n1_adj = n1_vectors_for_params(p)
    for trial in scenario["trials"][: int(reps)]:
        initial = case_initial_state(p, int(trial["seed"]))
        if fixed_n2 is not None:
            rule = lambda i, _lam, _state, vec=fixed_n2: vec[i]
        elif lookup is not None:
            rule = lambda i, lam, _state, table=lookup: lookup_n2(table, i, lam)
        else:
            rule = lambda i, lam, state: adjusted_n2(lam, n1_adj[i], n1_qed[i], len(state), p["h"], p["mu"], p["theta"], p["a"], p["c2"], p["xi2"])
        metrics, n2_vec = run_case_path(p, initial, n1_vec, rule, scenario["lambda_vec"], int(trial["seed"]))
        vals.append(metrics)
        n2_rows.append(n2_vec)
    mean = {key: float(np.mean([v[key] for v in vals])) for key in vals[0]}
    avg_n2 = np.mean(np.asarray(n2_rows, dtype=float), axis=0).tolist()
    row = {
        "benchmark": label, "dataset_tag": scenario["dataset_tag"], "scenario_id": int(scenario["scenario_id"]),
        "total_lambda": float(sum(scenario["lambda_vec"])), "total_cost": mean["total_cost"], "base_cost": mean["base_cost"],
        "surge_cost": mean["surge_cost"], "waiting_cost": mean["waiting_cost"], "abandonment_cost": mean["abandonment_cost"],
        "avg_queue": mean["avg_queue"], "avg_wait_time": mean["avg_wait_time"], "wait_over_10min_rate": mean["wait_over_10min_rate"],
        "lwbs_rate": mean["lwbs_rate"], "patient_count": mean["patient_count"], "lwbs_risk_count": mean["lwbs_risk_count"],
        "wait_over_10min_count": mean["wait_over_10min_count"], "n2_sum": float(sum(avg_n2)),
    }
    for i, lam in enumerate(scenario["lambda_vec"]):
        row[f"lambda_{i + 1}"] = lam
    for i, n2 in enumerate(avg_n2):
        row[f"n2_{i + 1}"] = n2
    return row


def eval_policy_worker_task(task: tuple[str, int, list[int], list[int] | None, list[dict[str, Any]] | None, int]) -> dict[str, Any]:
    label, sid, n1_vec, fixed_n2, lookup, reps = task
    p, scenarios = case_worker_context()
    return eval_policy_task((p, label, scenarios[int(sid)], n1_vec, fixed_n2, lookup, int(reps)))


def evaluate_policy(p: dict[str, Any], label: str, dataset: list[dict[str, Any]], n1_vec: list[int], fixed_by_scenario: dict[int, list[int]] | None, lookup: list[dict[str, Any]] | None, reps: int, executor: Any | None) -> list[dict[str, Any]]:
    if executor is None:
        tasks = [(p, label, s, n1_vec, None if fixed_by_scenario is None else fixed_by_scenario[int(s["scenario_id"])], lookup, reps) for s in dataset]
        return parallel_map(eval_policy_task, tasks, executor)
    tasks = [(label, int(s["scenario_id"]), n1_vec, None if fixed_by_scenario is None else fixed_by_scenario[int(s["scenario_id"])], lookup, reps) for s in dataset]
    return case_parallel_map(eval_policy_worker_task, tasks, executor)


def aggregate_case(rows: list[dict[str, Any]], n1_vec: list[int], search_time: float) -> dict[str, Any]:
    mean, low, high = mean_ci([float(r["total_cost"]) for r in rows])
    return {
        "mean_total_cost": mean, "ci95_lower": low, "ci95_upper": high,
        "mean_base_cost": float(np.mean([r["base_cost"] for r in rows])), "mean_surge_cost": float(np.mean([r["surge_cost"] for r in rows])),
        "mean_waiting_cost": float(np.mean([r["waiting_cost"] for r in rows])), "mean_abandonment_cost": float(np.mean([r["abandonment_cost"] for r in rows])),
        "mean_avg_queue": float(np.mean([r["avg_queue"] for r in rows])), "mean_avg_wait_time": float(np.mean([r["avg_wait_time"] for r in rows])),
        "mean_wait_over_10min_rate": float(np.mean([r["wait_over_10min_rate"] for r in rows])), "mean_lwbs_rate": float(np.mean([r["lwbs_rate"] for r in rows])),
        "mean_patient_count": float(np.mean([r["patient_count"] for r in rows])), "mean_lwbs_risk_count": float(np.mean([r["lwbs_risk_count"] for r in rows])),
        "mean_wait_over_10min_count": float(np.mean([r["wait_over_10min_count"] for r in rows])), "n1_vec": str(list(map(int, n1_vec))),
        "avg_n2_vec": str([round_half_up(float(np.mean([r[f'n2_{i + 1}'] for r in rows])), 4) for i in range(len(n1_vec))]), "search_time_sec": float(search_time),
    }


def paired_case_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {int(r["scenario_id"]): r for r in rows if r["benchmark"] == "1,SO"}
    other = {int(r["scenario_id"]): r for r in rows if r["benchmark"] == "JSO"}
    assert baseline.keys() == other.keys()
    ids = sorted(baseline)
    assert all(float(baseline[sid][key]) == float(other[sid][key])
               for sid in ids for key in baseline[sid] if key.startswith("lambda_"))
    differences = [float(baseline[sid]["total_cost"]) - float(other[sid]["total_cost"]) for sid in ids]
    mean, low, high = mean_ci(differences)
    return [{"baseline": "1,SO", "comparison": "JSO", "mean_cost_saving": mean,
             "saving_ci95_lower": low, "saving_ci95_upper": high,
             "saving_pct": 100.0 * mean / float(np.mean([float(baseline[sid]["total_cost"]) for sid in ids])),
             "scenario_count": len(ids), "win_rate_pct": 100.0 * sum(x > 0 for x in differences) / len(ids)}]


def run_case(output_root: Path, use_parallel: bool = True, params: dict[str, Any] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    p = dict(params or case_params())
    output_root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    start = time.perf_counter()
    search = build_case_dataset(p, p["search_scenarios"], p["search_replications"], p["search_seed"], "search")
    evl = build_case_dataset(p, p["eval_scenarios"], p["eval_replications"], p["eval_seed"], "eval")
    recourse = build_case_dataset(p, p["eval_scenarios"], p["search_replications"], p["eval_recourse_seed"], "eval_recourse", [s["lambda_vec"] for s in evl])
    lhs = build_lhs_dataset(p)
    timings["environment"] = time.perf_counter() - start
    n1_nv, _n1_qed, n1_adj = n1_vectors_for_params(p)
    # Each policy owns fresh workers and caches; reuse stays within that policy.
    eta_star.cache_clear()
    search_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_case_worker, initargs=(p, search)) if use_parallel else None
    if search_executor is None:
        init_case_worker(p, search)
    start = time.perf_counter()
    try:
        n1_1so, trace_1so = search_1so(p, search, n1_nv, p["search_replications"], search_executor)
        one_so_time = time.perf_counter() - start
        timings["search_1so"] = one_so_time
    finally:
        if search_executor is not None:
            search_executor.shutdown()
    eta_star.cache_clear()
    search_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_case_worker, initargs=(p, search)) if use_parallel else None
    if search_executor is None:
        init_case_worker(p, search)
    start = time.perf_counter()
    try:
        n1_jso, jso_results, jso_trace = jso_search(p, search, n1_adj, p["search_replications"], search_executor)
        jso_time = time.perf_counter() - start
        timings["search_jso"] = jso_time
        lookup_start = time.perf_counter()
        points = point_rows("jso_search", search, n1_jso, p["search_replications"], jso_results)
    finally:
        if search_executor is not None:
            search_executor.shutdown()
    lhs_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_case_worker, initargs=(p, lhs)) if use_parallel else None
    if lhs_executor is None:
        init_case_worker(p, lhs)
    try:
        checkpoint = output_root / "lhs_pathwise_checkpoint.csv"
        signature_path = output_root / "lhs_checkpoint_config.json"
        signature = {
            "n1": n1_jso,
            "lhs": [p["lhs_scenarios"], p["lhs_replications"], p["lhs_seed"], p["lhs_stream_base"]],
            "pathwise_search": ["qed", p["n2_search_upper_floor"]],
            "model": [list(p["lambda_means"]), *[p[key] for key in ("alpha", "rho", "sigma2", "mu", "theta", "c1", "c2", "h", "a", "xi1", "xi2")]],
            "inner_limit": p["inner_limit"],
        }
        signature["revision"] = {"config": {k: v for k, v in p.items() if k != "workers"},
                                 "source": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                            for name in ("case_study.py", "simulation.py", "numerical.py", "formulas.py")}}
        signature = json.loads(json.dumps(signature))
        if checkpoint.exists() and (not signature_path.exists() or json.loads(signature_path.read_text(encoding="utf-8")) != signature):
            raise ValueError("LHS checkpoint belongs to another experiment")
        if not checkpoint.exists():
            write_json(signature_path, signature)
        lhs_points = load_lhs_checkpoint(checkpoint, lhs, n1_jso, p["lhs_replications"])
        completed = {int(row["scenario_id"]) for row in lhs_points}
        missing = [scenario for scenario in lhs if int(scenario["scenario_id"]) not in completed]
        batch_size = int(p["lhs_batch_size"])
        for offset in range(0, len(missing), batch_size):
            new = collect_points(p, "lhs_extra", missing[offset:offset + batch_size], n1_jso, p["lhs_replications"], lhs_executor)
            lhs_points.extend(new)
            lhs_points.sort(key=lambda row: (int(row["scenario_id"]), int(row["shift"])))
            save_lhs_checkpoint(checkpoint, lhs_points)
            print(f"[LHS N2 search] {len({row['scenario_id'] for row in lhs_points})}/{p['lhs_scenarios']} paths", flush=True)
        if len(lhs_points) != int(p["lhs_scenarios"]) * int(p["horizon"]):
            raise ValueError("LHS pathwise search is incomplete")
        points += lhs_points
    finally:
        if lhs_executor is not None:
            lhs_executor.shutdown()
    lookups = [("JSO", bins, build_lookup(points, bins, "JSO")) for bins in p["lhs_bin_counts"]]
    if p.get("lookup_full_domain", False):
        for _label, _bins, lookup in lookups:
            for shift in range(p["horizon"]):
                group = [r for r in lookup if r["shift"] == shift]
                group[0]["lambda_lower"], group[-1]["lambda_upper"] = 0.0, float("inf")
    lookup_time = time.perf_counter() - lookup_start
    timings["lookup_training"] = lookup_time
    start = time.perf_counter()
    eta_star.cache_clear()
    recourse_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_case_worker, initargs=(p, recourse)) if use_parallel else None
    if recourse_executor is None:
        init_case_worker(p, recourse)
    try:
        if recourse_executor is None:
            oracle_results = parallel_map(pathwise_task, [(p, s, n1_jso, p["search_replications"], None) for s in recourse], None)
        else:
            oracle_results = case_parallel_map(pathwise_worker_task, [(int(s["scenario_id"]), n1_jso, p["search_replications"], None) for s in recourse], recourse_executor)
        n2_oracle = {int(s["scenario_id"]): vec for s, (vec, _cost) in zip(recourse, oracle_results)}
        timings["eval_recourse_n2"] = time.perf_counter() - start
    finally:
        if recourse_executor is not None:
            recourse_executor.shutdown()
    policies = [
        ("1,NV", n1_nv, {int(s["scenario_id"]): [0] * p["horizon"] for s in evl}, None, 0.0),
        ("1,SO", n1_1so, {int(s["scenario_id"]): [0] * p["horizon"] for s in evl}, None, one_so_time),
        ("2,QED", n1_adj, None, None, 0.0),
        *[(label, n1_jso, None, lookup, jso_time + lookup_time) for label, _bins, lookup in lookups],
    ]
    scenario_rows: list[dict[str, Any]] = []
    bench = []
    oracle_rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for label, n1, fixed, table, stime in [*policies, ("JSO-Oracle", n1_jso, n2_oracle, None, jso_time)]:
        eta_star.cache_clear()
        eval_executor = ProcessPoolExecutor(max_workers=p.get("workers"), initializer=init_case_worker, initargs=(p, evl)) if use_parallel else None
        if eval_executor is None:
            init_case_worker(p, evl)
        policy_start = time.perf_counter()
        try:
            rows = evaluate_policy(p, label, evl, n1, fixed, table, p["eval_replications"], eval_executor)
            policy_time = time.perf_counter() - policy_start
        finally:
            if eval_executor is not None:
                eval_executor.shutdown()
        if label == "1,SO":
            timings["evaluation_1so"] = policy_time
        benchmark = {"benchmark": label, **aggregate_case(rows, n1, stime)}
        if label == "JSO-Oracle":
            oracle_rows, oracle_benchmark = rows, benchmark
        else:
            scenario_rows.extend(rows)
            bench.append(benchmark)
    timings["evaluation"] = time.perf_counter() - start
    order = {label: i for i, (label, *_rest) in enumerate(policies)}
    start = time.perf_counter()
    write_csv(output_root / "benchmark_table.csv", bench)
    write_csv(output_root / "scenario_costs.csv", sorted(scenario_rows, key=lambda r: (order[r["benchmark"]], r["scenario_id"])))
    write_csv(output_root / "paired_comparison.csv", paired_case_comparison(scenario_rows))
    write_csv(output_root / "oracle_benchmark.csv", [oracle_benchmark])
    write_csv(output_root / "oracle_scenario_costs.csv", oracle_rows)
    for _label, bins, lookup in lookups:
        write_csv(output_root / f"jso_lookup_{bins}bins.csv", sorted(lookup, key=lambda r: (r["shift"], r["bin_id"])))
    write_csv(output_root / "jso_pathwise_n2_points.csv", sorted(points, key=lambda r: (0 if r["source"] == "jso_search" else 1, r["scenario_id"], r["shift"])))
    from figures import save_cost_decomposition

    save_cost_decomposition(bench, output_root / "cost_decomposition.png", 1000.0)
    timings["outputs"] = time.perf_counter() - start
    timings["total"] = time.perf_counter() - t0
    summary = {"config": p, "paths": {"summary_json": str(output_root / "summary.json")}, "timings_sec": timings, "benchmarks": {r["benchmark"]: r for r in bench}, "oracle_benchmark": oracle_benchmark, "n1_search_trace_1so": trace_1so, "jso_outer_trace": jso_trace, "lookup_training": {"point_count": len(points), "bin_counts": [bins for _label, bins, _lookup in lookups]}}
    summary["n1_search_starts"] = {"1,SO": n1_nv, "JSO": n1_adj}
    summary["analytic_sigma"] = math.sqrt(float(p["sigma2"])) / float(p["mu"]) ** (1.0 - float(p["alpha"]))
    if p.get("record_search_bounds", False):
        bounds = []

        def record(stage: str, label: str, values: list[int], uppers: list[int]) -> None:
            hits = sum(v == u for v, u in zip(values, uppers))
            bounds.append({"stage": stage, "benchmark": label, "decisions": len(values), "upper_hits": hits,
                           "upper_hit_rate": hits / len(values), "zero_count": values.count(0),
                           "upper_min": min(uppers), "upper_max": max(uppers)})

        for label, values, initial in (("1,SO", n1_1so, n1_nv), ("JSO", n1_jso, n1_adj)):
            record("final_n1", label, values, [2 * max(1, n) for n in initial])
        lhs_only = [r for r in points if r["source"] == "lhs_extra"]
        record("lhs_n2", "JSO", [r["n2_pathwise"] for r in lhs_only],
               [max(p["n2_search_upper_floor"], 2 * max(1, qed_n2(r["lambda_fixed"], r["n1_shift"], p["h"], p["mu"], p["theta"], p["a"], p["c2"]))) for r in lhs_only])
        write_csv(output_root / "boundary_diagnostics.csv", bounds)
        summary["boundary_diagnostics"] = bounds
        summary["boundary_scope"] = "Final N1 and canonical-start LHS N2. Continuation-search N2 bounds depend on the warm start and are not reconstructed from final decisions."
    summary["initialization_semantics"] = (
        f"Each full path starts with {p.get('initial_patient_count', 0)} previously served patients "
        "with exponential remaining service times and zero historical wait; state carries between shifts.")
    write_json(output_root / "summary.json", summary)
    return summary


def case_sensitivity_row(summary: dict[str, Any], rho: float, sigma2: float) -> dict[str, Any]:
    costs = {label: float(row["mean_total_cost"]) for label, row in summary["benchmarks"].items()}
    return {
        "sigma2": float(sigma2), "rho": float(rho),
        "cost_2qed": round_half_up(costs["2,QED"]), "cost_jso": round_half_up(costs["JSO"]),
        "jso_cost_reduction_pct": round_half_up(100.0 * (costs["2,QED"] - costs["JSO"]) / costs["2,QED"], 2),
        "cost_1so": round_half_up(costs["1,SO"]),
        "jso_saving_vs_1so_pct": round_half_up(100.0 * (costs["1,SO"] - costs["JSO"]) / costs["1,SO"], 2),
    }
