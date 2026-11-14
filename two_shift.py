from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from case_study import lru_put
from formulas import qed_n1, qed_n2
from numerical import hill_climb, mean_ci, namespaced_seed, write_csv, write_json
from simulation import build_model_stream, service_samples, simulate_shift

STAT_KEYS = ("des_calls", "des_cache_hits", "second_solves", "decision_cache_hits", "qos_cache_hits", "stream_cache_hits",
             "first_value_solves", "first_value_cache_hits", "shared_qos_cache_hits", "path_value_cache_hits", "cost_only_des_calls")
METHODS = ("SEQ_REF", "COND_MEAN", "BIN_LOOKUP")
_WORKER_CONTEXT: dict | None = None
_EMPTY_STATE_KEY = hashlib.sha256(b"[]").digest()


def two_params(pilot: bool = True) -> dict[str, Any]:
    return {
        "lambda_means": [100.0, 100.0], "alpha": 0.75, "sigma": 1.0, "rho": 0.3,
        "mu": 1.0, "theta": 0.1, "c1": 1.0, "c2": 1.5, "h": 1.5, "a": 3.0,
        "arrival_distribution": "exponential", "arrival_scv": 1.0,
        "service_distribution": "exponential", "service_scv": 1.0,
        "renewal_boundary": "restart_each_shift", "initialization": "empty",
        "model_status": "M/M/k+M; exponential service selected for the two-shift comparison",
        "M": 2 if pilot else 100, "J": 2 if pilot else 40, "R": 2 if pilot else 20,
        "K": 2 if pilot else 10, "path_replications": 2 if pilot else 20,
        "lookup_bins": 2 if pilot else 20,
        "lookup_extra_paths": 2 if pilot else 900,
        "eval_scenarios": 2 if pilot else 100, "eval_replications": 2 if pilot else 100,
        "outer_rounds": 1, "inner_rounds": 2, "n2_upper_floor": 20,
        "seed_namespace": 2026102820 if pilot else 2026102821,
        "workers": max(1, (os.cpu_count() or 2) // 2), "cache_limit": 2000,
    }


def seed(p: dict, stage: int, sid: int = 0, branch: int = 0, rep: int = 0, stream: int = 0) -> int:
    if not 0 <= sid < 2 ** 20 or not 0 <= branch < 2 ** 20:
        raise ValueError("Two-shift scenario and branch identifiers need at most 20 bits each")
    return namespaced_seed(int(p["seed_namespace"]), stage, (sid << 20) | branch, rep, stream)


def joint_rate(p: dict, rng: np.random.Generator) -> list[float]:
    means = np.asarray(p["lambda_means"], dtype=float)
    scale = p["sigma"] * means ** p["alpha"]
    while True:
        z = rng.normal(size=2)
        z[1] = p["rho"] * z[0] + math.sqrt(1 - p["rho"] ** 2) * z[1]
        values = means + scale * z
        if np.all(values > 0):
            return values.tolist()


def conditional_rate(p: dict, known_shift: int, known_rate: float, rng: np.random.Generator) -> float:
    other = 1 - known_shift
    means = p["lambda_means"]
    sd = [p["sigma"] * float(m) ** p["alpha"] for m in means]
    mean = means[other] + p["rho"] * sd[other] / sd[known_shift] * (known_rate - means[known_shift])
    while True:
        value = float(rng.normal(mean, sd[other] * math.sqrt(1 - p["rho"] ** 2)))
        if value > 0:
            return value


def observable_state(state: list[dict]) -> list[dict]:
    # Never expose latent residual service/patience or a pre-generated full service draw.
    keys = ("priority", "arrival_rank", "has_started_service", "wait_total", "wait_over_10min_counted")
    return [{key: row.get(key, False if key in ("has_started_service", "wait_over_10min_counted") else 0.0)
             for key in keys} for row in state]


def conditional_state(p: dict, observed: list[dict], service_seed: int, patience_seed: int) -> list[dict]:
    if p["service_distribution"] != "exponential":
        raise ValueError("The two-shift sequential reference requires exponential service")
    services, patience = np.random.default_rng(service_seed), np.random.default_rng(patience_seed)
    result = []
    for row in observable_state(observed):
        patient = dict(row)
        if row["has_started_service"]:
            patient["remaining_service_time"] = float(services.exponential(1.0 / p["mu"]))
            patient["remaining_patience_time"] = None
        else:
            patient["service_time"] = float(service_samples(services, p["mu"], p["service_distribution"], p["service_scv"], 1)[0])
            patient["remaining_patience_time"] = float(patience.exponential(1.0 / p["theta"]))
        result.append(patient)
    return result


def context() -> dict:
    return {"perf": OrderedDict(), "decisions": OrderedDict(), "qos": OrderedDict(), "streams": OrderedDict(),
            "first_values": OrderedDict(), "second_values": OrderedDict(),
            "queries": [], "query_index": {}, **dict.fromkeys(STAT_KEYS, 0)}


def reset_diagnostics(ctx: dict) -> None:
    ctx["queries"] = []
    ctx["query_index"] = {}
    ctx.update(dict.fromkeys(STAT_KEYS, 0))


def init_worker() -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = None


def worker_context(p: dict) -> dict:
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None or _WORKER_CONTEXT.get("config") != p:
        _WORKER_CONTEXT = context()
        check_context(p, _WORKER_CONTEXT)
    # Retain bounded caches, but return each task's diagnostics only once.
    reset_diagnostics(_WORKER_CONTEXT)
    return _WORKER_CONTEXT


def task_result(ctx: dict, **values: Any) -> dict:
    return {**values, "queries": ctx["queries"], "stats": {key: ctx[key] for key in STAT_KEYS}}


def merge_results(ctx: dict, results: list[dict]) -> None:
    for result in results:
        for row in result["queries"]:
            record_query(ctx, row)
        for key in STAT_KEYS:
            ctx[key] += result["stats"][key]


def record_query(ctx: dict, row: dict) -> None:
    key = tuple((key, value) for key, value in row.items() if key != "request_count")
    index = ctx["query_index"].get(key)
    if index is None:
        ctx["query_index"][key] = len(ctx["queries"])
        ctx["queries"].append(dict(row))
    else:
        ctx["queries"][index]["request_count"] += row["request_count"]


def progress_map(fn: Any, tasks: list, executor: Any | None, label: str) -> list:
    started = time.perf_counter()
    if executor is None:
        results = []
        last_report = started
        for task in tasks:
            results.append(fn(task))
            now = time.perf_counter()
            if now - last_report >= 30 or len(results) == len(tasks):
                print(f"[Two shift] {label}: {len(results)}/{len(tasks)}, elapsed={now - started:.1f}s", flush=True)
                last_report = now
        return results
    pending = {executor.submit(fn, task): i for i, task in enumerate(tasks)}
    results = [None] * len(tasks)
    completed, last_report = 0, started
    while pending:
        done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
        for future in done:
            results[pending.pop(future)] = future.result()
            completed += 1
        now = time.perf_counter()
        if now - last_report >= 30 or not pending:
            print(f"[Two shift] {label}: {completed}/{len(tasks)}, elapsed={now - started:.1f}s", flush=True)
            last_report = now
    return results


def check_context(p: dict, ctx: dict) -> None:
    if "config" not in ctx:
        ctx["config"] = deepcopy(p)
        ctx["config_key"] = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).digest()
    if ctx["config"] != p:
        raise ValueError("Use a fresh context when changing model parameters or sampling budgets")


def shift_cost(p: dict, state: list[dict], capacity: int, lam: float, stage: int, sid: int,
               branch: int, rep: int, shift: int, ctx: dict, state_key: Any | None = None, need_state: bool = True) -> dict:
    check_context(p, ctx)
    key = (capacity, lam, stage, sid, branch, rep, shift, need_state,
           hashlib.sha256(json.dumps(state, sort_keys=True).encode()).digest() if state_key is None else state_key)
    if key in ctx["perf"]:
        ctx["des_cache_hits"] += 1
        ctx["perf"].move_to_end(key)
        return ctx["perf"][key]
    base = 1 + shift * 4
    stream_key = (lam, stage, sid, branch, rep, shift)
    stream = ctx["streams"].get(stream_key)
    if stream is None:
        stream = build_model_stream(p, lam, *(seed(p, stage, sid, branch, rep, base + k) for k in range(3)))
        lru_put(ctx["streams"], stream_key, stream, int(p["cache_limit"]))
    else:
        ctx["stream_cache_hits"] += 1
        ctx["streams"].move_to_end(stream_key)
    value = simulate_shift(state, capacity, 0, lam, p["mu"], p["theta"], 0, 0, p["h"], p["a"], 0,
                           stream, strict_stream=True, cost_only=not need_state)
    ctx["des_calls"] += 1
    ctx["cost_only_des_calls"] += int(not need_state)
    if need_state:
        value["_state_key"] = hashlib.sha256(json.dumps(value["final_state"], sort_keys=True).encode()).digest()
    lru_put(ctx["perf"], key, value, int(p["cache_limit"]))
    return value


def minimize(start: int, objective: Any, upper: int, lower: int = 0, expand_upper: bool = False) -> tuple[int, float, int, int]:
    expansions = 0
    while True:
        result = hill_climb(start, objective, 2, upper, lower)
        decision, value = result["best_n1"], result["best_objective"]
        if not expand_upper or decision < upper:
            return decision, value, upper, expansions
        if expansions == 20:
            raise RuntimeError("N2 still touches its upper bound after 20 expansions; do not use this as a reference result")
        # The objective closure retains exactly the same samples and cached costs.
        start, upper = decision, max(upper + 1, 2 * upper)
        expansions += 1


def coordinate_search(start: list[int], objective: Any, rounds: int, uppers: list[int], lower: int, expand_upper: bool = False) -> tuple[list[int], float, list[dict]]:
    current, trace, memo = list(start), [], {}
    uppers = list(uppers)

    def cost(vec: list[int]) -> float:
        key = tuple(vec)
        if key not in memo:
            memo[key] = float(objective(vec))
        return memo[key]

    for step in range(len(start) * rounds):
        i, before = step % len(start), list(current)
        old = cost(current)
        initial_upper = uppers[i]
        decision, value, uppers[i], expansions = minimize(current[i], lambda n: cost([n if j == i else current[j] for j in range(len(start))]), uppers[i], lower, expand_upper)
        current[i] = decision
        trace.append({"coordinate": step + 1, "shift": i, "n1_before": str(before), "n1_after": str(current),
                      "objective_before": old, "objective_after": value, "initial_upper": initial_upper,
                      "upper": uppers[i], "upper_hit": decision == uppers[i], "upper_expansions": expansions})
    return current, cost(current), trace


def n2_start_upper(p: dict, n1: int, lam: float) -> tuple[int, int]:
    start = qed_n2(lam, n1, p["h"], p["mu"], p["theta"], p["a"], p["c2"])
    return start, int(p.get("n2_upper", max(p["n2_upper_floor"], 2 * max(1, start))))


def log_decision(ctx: dict, kind: str, stage: int, sid: int, branch: int, n1: list[int], lam: float,
                 n2: int, objective: float, upper: int, observed: list[dict], initial_upper: int, expansions: int) -> None:
    record_query(ctx, {"kind": kind, "stage": stage, "scenario_id": sid, "branch": branch,
                           **{key: ctx["config"][key] for key in ("M", "J", "R", "K", "path_replications")},
                           "n1": str(n1), "lambda_fixed": lam, "n2": n2, "objective": objective,
                           "initial_upper": initial_upper, "initial_upper_hit": expansions > 0 or n2 == initial_upper,
                           "upper": upper, "upper_hit": n2 == upper, "upper_expansions": expansions,
                           "observed_patients": len(observed), "request_count": 1,
                           "observed_state_sha256": hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).hexdigest()})


def solve_second(p: dict, n1: list[int], observed: list[dict], lam: float, stage: int, sid: int, branch: int, ctx: dict, shared_qos: dict | None = None) -> tuple[int, float]:
    check_context(p, ctx)
    observed = observable_state(observed)
    observed_key = hashlib.sha256(json.dumps(observed, sort_keys=True).encode()).digest()
    if shared_qos is not None:
        sample_key = (ctx["config_key"], lam, stage, sid, branch, observed_key)
        if not shared_qos:
            shared_qos.update(key=sample_key, values={})
        if shared_qos["key"] != sample_key:
            raise ValueError("Second-shift QoS table belongs to a different state or sampling context")
    # First-shift staffing no longer matters once its observable end state is fixed.
    key = ("second", n1[1], lam, stage, sid, branch, observed_key)
    if key in ctx["decisions"]:
        ctx["decision_cache_hits"] += 1
        ctx["decisions"].move_to_end(key)
        result, final_upper, upper, expansions, queried_qos = ctx["decisions"][key]
        if shared_qos is not None:
            shared_qos["values"].update(queried_qos)
        log_decision(ctx, "SEQ_second", stage, sid, branch, n1, lam, *result, final_upper, observed, upper, expansions)
        return result
    ctx["second_solves"] += 1
    samples = None
    values = {}
    queried_qos = {}

    def objective(n2: int) -> float:
        nonlocal samples
        if n2 not in values:
            capacity = n1[1] + n2
            qos_key = (capacity, lam, stage, sid, branch, observed_key)
            if shared_qos is not None and capacity in shared_qos["values"]:
                qos = shared_qos["values"][capacity]
                ctx["shared_qos_cache_hits"] += 1
            else:
                qos = ctx["qos"].get(qos_key)
                if qos is None:
                    if samples is None:
                        samples = [conditional_state(p, observed, seed(p, stage, sid, branch, r, 8), seed(p, stage, sid, branch, r, 9)) for r in range(p["R"])]
                    # Observed state plus the sampling identifiers fully determines each residual draw.
                    qos = float(np.mean([shift_cost(p, state, capacity, lam, stage, sid, branch, r, 1, ctx,
                                                   ("conditional", observed_key), False)["total_cost"]
                                         for r, state in enumerate(samples)]))
                else:
                    ctx["qos_cache_hits"] += 1
            lru_put(ctx["qos"], qos_key, qos, 10 * int(p["cache_limit"]))
            queried_qos[capacity] = qos
            values[n2] = p["c2"] * n2 + qos
        return values[n2]

    start, upper = n2_start_upper(p, n1[1], lam)
    decision, value, final_upper, expansions = minimize(start, objective, upper, expand_upper=True)
    result = (decision, value)
    if shared_qos is not None:
        shared_qos["values"].update(queried_qos)
    lru_put(ctx["decisions"], key, (result, final_upper, upper, expansions, queried_qos), int(p["cache_limit"]))
    log_decision(ctx, "SEQ_second", stage, sid, branch, n1, lam, *result, final_upper, observed, upper, expansions)
    return result


def first_branch(p: dict, n1: list[int], capacity: int, lam: float, lam2: float, stage: int, sid: int, j: int, ctx: dict, shared_qos: dict | None = None) -> float:
    first = shift_cost(p, [], capacity, lam, stage + 1, sid, j, 0, 0, ctx)
    _n2, value = solve_second(p, n1, observable_state(first["final_state"]), lam2, stage + 2, sid, j, ctx, shared_qos)
    return first["total_cost"] + value


def branch_task(task: tuple) -> dict:
    ctx = worker_context(task[0])
    shared_qos = task[-1]
    return task_result(ctx, cost=first_branch(*task[:-1], ctx, shared_qos), shared_qos=shared_qos)


def solve_first(p: dict, n1: list[int], lam: float, stage: int, sid: int, ctx: dict, executor: Any | None = None) -> tuple[int, float]:
    check_context(p, ctx)
    key = ("first", tuple(n1), lam, stage, sid)
    if key in ctx["decisions"]:
        ctx["decision_cache_hits"] += 1
        ctx["decisions"].move_to_end(key)
        result, final_upper, upper, expansions = ctx["decisions"][key]
        log_decision(ctx, "SEQ_first", stage, sid, 0, n1, lam, *result, final_upper, [], upper, expansions)
        return result
    future = [conditional_rate(p, 0, lam, np.random.default_rng(seed(p, stage, sid, j, 0, 10))) for j in range(p["J"])]
    values = {}

    def objective(n2: int) -> float:
        if n2 not in values:
            capacity = n1[0] + n2
            value_key = (stage, sid, lam, capacity, n1[1])
            cached = ctx["first_values"].get(value_key)
            if cached is not None:
                ctx["first_value_cache_hits"] += 1
                ctx["first_values"].move_to_end(value_key)
                branch_mean, queries = cached
                for row in queries:
                    record_query(ctx, {**row, "n1": str(n1)})
            else:
                ctx["first_value_solves"] += 1
                # Share serial caches, but isolate this batch's query log and counters.
                branch_ctx = dict(ctx)
                reset_diagnostics(branch_ctx)
                # These QoS tables omit N1/N2 staffing costs and survive changes to N12.
                branch_key = (stage, sid, lam, capacity)
                tables = ctx["second_values"].get(branch_key)
                if tables is None:
                    tables = [{} for _ in future]
                tasks = [(p, n1, capacity, lam, lam2, stage, sid, j, tables[j]) for j, lam2 in enumerate(future)]
                if executor is None:
                    branches = [first_branch(*task[:-1], branch_ctx, task[-1]) for task in tasks]
                else:
                    results = list(executor.map(branch_task, tasks))
                    merge_results(branch_ctx, results)
                    branches = [row["cost"] for row in results]
                    tables = [row["shared_qos"] for row in results]
                lru_put(ctx["second_values"], branch_key, tables, int(p["cache_limit"]))
                branch_mean = float(np.mean(branches))
                merge_results(ctx, [task_result(branch_ctx)])
                lru_put(ctx["first_values"], value_key, (branch_mean, branch_ctx["queries"]), int(p["cache_limit"]))
            # The cached continuation cost excludes first-shift staffing charges.
            values[n2] = p["c2"] * n2 + branch_mean
        return values[n2]

    start, upper = n2_start_upper(p, n1[0], lam)
    decision, value, final_upper, expansions = minimize(start, objective, upper, expand_upper=True)
    result = (decision, value)
    lru_put(ctx["decisions"], key, (result, final_upper, upper, expansions), int(p["cache_limit"]))
    log_decision(ctx, "SEQ_first", stage, sid, 0, n1, lam, *result, final_upper, [], upper, expansions)
    return result


def path_n2(p: dict, n1: list[int], rates: list[float], stage: int, sid: int, branch: int, ctx: dict, shared_path: dict | None = None) -> tuple[list[int], float]:
    check_context(p, ctx)
    if shared_path is None:
        shared_path = {}
    sample_key = (ctx["config_key"], tuple(rates), stage, sid, branch)
    if not shared_path:
        shared_path.update(key=sample_key, values=OrderedDict())
    if shared_path["key"] != sample_key:
        raise ValueError("Path QoS table belongs to a different path or sampling context")
    starts, uppers = zip(*(n2_start_upper(p, n, lam) for n, lam in zip(n1, rates)))

    def objective(n2: list[int]) -> float:
        capacities = tuple(base + surge for base, surge in zip(n1, n2))
        qos = shared_path["values"].get(capacities)
        if qos is not None:
            ctx["path_value_cache_hits"] += 1
            shared_path["values"].move_to_end(capacities)
        else:
            costs = []
            for rep in range(p["path_replications"]):
                state, state_key, cost = [], _EMPTY_STATE_KEY, 0.0
                for shift in range(len(rates)):
                    need_state = shift < len(rates) - 1
                    result = shift_cost(p, state, capacities[shift], rates[shift], stage, sid, branch, rep, shift, ctx, state_key, need_state)
                    if need_state:
                        state, state_key = result["final_state"], result["_state_key"]
                    cost += result["total_cost"]
                costs.append(cost)
            qos = float(np.mean(costs))
            lru_put(shared_path["values"], capacities, qos, int(p["cache_limit"]))
        return p["c2"] * sum(n2) + qos

    n2, cost, trace = coordinate_search(list(starts), objective, p["inner_rounds"], list(uppers), 0, expand_upper=True)
    for shift in range(len(rates)):
        visits = [row for row in trace if row["shift"] == shift]
        log_decision(ctx, f"path_shift_{shift}", stage, sid, branch, n1, rates[shift], n2[shift], cost,
                     visits[-1]["upper"], [], uppers[shift], sum(row["upper_expansions"] for row in visits))
    return n2, cost


def path_task(task: tuple) -> dict:
    ctx = worker_context(task[0])
    n2, cost = path_n2(*task, ctx)
    return task_result(ctx, n2=n2, cost=cost)


def auxiliary_task(task: tuple) -> dict:
    p, n1, rates, sid, shared_path = task
    ctx = worker_context(p)
    n2, cost = path_n2(p, n1, rates, 31, sid, 0, ctx, shared_path)
    return task_result(ctx, n2=n2, cost=p["c1"] * sum(n1) + cost, shared_path=shared_path)


def sequential_task(task: tuple, executor: Any | None, ctx: dict) -> dict:
    p, n1, rates, sid = task
    check_context(p, ctx)
    reset_diagnostics(ctx)
    n2, cost = solve_first(p, n1, rates[0], 10, sid, ctx, executor)
    return task_result(ctx, cost=p["c1"] * sum(n1) + cost, n2=[n2])


def replication_row(p: dict, n1: list[int], n2: list[int], outcomes: list[dict]) -> dict:
    waiting, abandonment, queue, abandons, arrivals = (
        sum(row[key] for row in outcomes)
        for key in ("waiting_cost", "abandonment_cost", "queue_area", "abandon_count", "arrival_count"))
    base, surge = p["c1"] * sum(n1), p["c2"] * sum(n2)
    return {"total_cost": base + surge + waiting + abandonment, "base_cost": base, "surge_cost": surge,
            "waiting_cost": waiting, "abandonment_cost": abandonment, "avg_queue": queue / 2,
            "avg_wait_time": queue / arrivals if arrivals else 0, "lwbs_rate": abandons / arrivals if arrivals else 0,
            "n2_1": n2[0], "n2_2": n2[1], "terminal_count": outcomes[-1]["terminal_count"]}


def evaluate_replication(p: dict, method: str, n1: list[int], fixed_n2: list[int], rates: list[float], sid: int, rep: int, ctx: dict) -> dict:
    state, outcomes, decisions, state_key = [], [], [], _EMPTY_STATE_KEY
    for shift, decision in enumerate(fixed_n2):
        decisions.append({"benchmark": method, "scenario_id": sid, "replication_id": rep, "shift": shift,
                          "lambda_fixed": rates[shift], "initial_patients": len(state), "n1": n1[shift], "n2": decision})
        result = shift_cost(p, state, n1[shift] + decision, rates[shift], 61, sid, 0, rep, shift, ctx, state_key)
        state, state_key = result["final_state"], result["_state_key"]
        outcomes.append({**result, "terminal_count": len(state)})
    return {"row": replication_row(p, n1, fixed_n2, outcomes), "decisions": decisions}


def first_decision_task(task: tuple, executor: Any | None) -> dict:
    p, n1, lam, sid = task
    ctx = context()
    return task_result(ctx, n2=solve_first(p, n1, lam, 20, sid, ctx, executor)[0])


def second_decision_task(task: tuple) -> dict:
    p, n1, observed, lam, sid, rep = task
    ctx = worker_context(p)
    return task_result(ctx, n2=solve_second(p, n1, observed, lam, 23, sid, rep, ctx)[0])


def execution_task(task: tuple) -> dict:
    p, state, capacity, lam, sid, rep, shift = task
    ctx = worker_context(p)
    result = shift_cost(p, state, capacity, lam, 61, sid, 0, rep, shift, ctx)
    value = {key: result[key] for key in ("waiting_cost", "abandonment_cost", "queue_area", "abandon_count", "arrival_count")}
    value["terminal_count"] = len(result["final_state"])
    if shift == 0:
        value["final_state"] = result["final_state"]
    return task_result(ctx, value=value)


def evaluate_sequential(p: dict, n1: list[int], rates: list[list[float]], use_parallel: bool) -> dict:
    # Separate decision batches from actual execution to measure wall time without solving twice.
    init_worker()
    executor = ProcessPoolExecutor(max_workers=p["workers"], initializer=init_worker) if use_parallel else None
    coordinators = ThreadPoolExecutor(max_workers=p["workers"]) if use_parallel else None
    timings, ctx = {}, context()
    pairs = [(sid, rep) for sid in range(len(rates)) for rep in range(p["eval_replications"])]
    try:
        tasks = [(p, n1, path[0], sid) for sid, path in enumerate(rates)]
        begin = time.perf_counter()
        first = progress_map(lambda task: first_decision_task(task, executor), tasks, coordinators, "OPT online shift 1")
        timings["online_first_shift"] = time.perf_counter() - begin

        tasks = [(p, [], n1[0] + first[sid]["n2"], rates[sid][0], sid, rep, 0) for sid, rep in pairs]
        begin = time.perf_counter()
        first_runs = progress_map(execution_task, tasks, executor, "OPT execution shift 1")
        timings["first_shift_execution"] = time.perf_counter() - begin

        tasks = [(p, n1, observable_state(run["value"]["final_state"]), rates[sid][1], sid, rep)
                 for (sid, rep), run in zip(pairs, first_runs)]
        begin = time.perf_counter()
        second = progress_map(second_decision_task, tasks, executor, "OPT online shift 2")
        timings["online_second_shift"] = time.perf_counter() - begin

        # Execute with the actual residuals, not the conditional samples used to choose N2.
        tasks = [(p, run["value"]["final_state"], n1[1] + decision["n2"], rates[sid][1], sid, rep, 1)
                 for (sid, rep), run, decision in zip(pairs, first_runs, second)]
        begin = time.perf_counter()
        second_runs = progress_map(execution_task, tasks, executor, "OPT execution shift 2")
        timings["second_shift_execution"] = time.perf_counter() - begin
    finally:
        if coordinators is not None:
            coordinators.shutdown()
        if executor is not None:
            executor.shutdown()
    timings["online_total"] = timings["online_first_shift"] + timings["online_second_shift"]
    merge_results(ctx, first + first_runs + second + second_runs)
    rep_rows, decisions = [], []
    for (sid, rep), a, b, decision in zip(pairs, first_runs, second_runs, second):
        n2 = [first[sid]["n2"], decision["n2"]]
        rep_rows.append(replication_row(p, n1, n2, [a["value"], b["value"]]))
        decisions.extend({"benchmark": "SEQ_REF", "scenario_id": sid, "replication_id": rep, "shift": shift,
                          "lambda_fixed": rates[sid][shift], "initial_patients": 0 if shift == 0 else a["value"]["terminal_count"],
                          "n1": n1[shift], "n2": n2[shift]} for shift in range(2))
    rows = []
    for sid, path in enumerate(rates):
        group = rep_rows[sid * p["eval_replications"]:(sid + 1) * p["eval_replications"]]
        rows.append({"benchmark": "SEQ_REF", "scenario_id": sid, "lambda_1": path[0], "lambda_2": path[1], "n1_vec": str(n1),
                     **{key: float(np.mean([row[key] for row in group])) for key in group[0]}})
    return task_result(ctx, rows=rows, decisions=decisions, timings_sec=timings)


def paired_cost_rows(rows: list[dict], comparisons: tuple[tuple[str, str, str], ...]) -> list[dict]:
    by = {label: {row["scenario_id"]: row["total_cost"] for row in rows if row["benchmark"] == label}
          for label in {label for _name, left, right in comparisons for label in (left, right)}}
    result = []
    for name, left, right in comparisons:
        ids = sorted(by[left])
        if not ids or set(ids) != set(by[right]):
            raise ValueError("Paired comparisons require the same nonempty scenario IDs")
        values = [by[left][i] - by[right][i] for i in ids]
        mean, lo, hi = mean_ci(values)
        result.append({"gap": name, "left": left, "right": right, "mean_cost_difference": mean,
                       "ci95_lower": lo, "ci95_upper": hi, "relative_to_right": mean / np.mean([by[right][i] for i in ids]),
                       "left_cheaper_rate": float(np.mean(np.asarray(values) < 0)), "scenario_count": len(ids)})
    return result


def train_two(output: Path, p: dict, use_parallel: bool = True) -> dict:
    p = dict(p)
    if abs(p["rho"]) >= 1 or p["sigma"] <= 0 or p["initialization"] != "empty" or p["renewal_boundary"] != "restart_each_shift":
        raise ValueError("Two-shift solver requires empty start, positive variance, |rho|<1 and restarting renewal clocks")
    if min(p[key] for key in ("M", "J", "R", "K", "path_replications", "eval_scenarios", "eval_replications")) < 1:
        raise ValueError("All sampling budgets must be positive")
    if not 1 <= p["lookup_bins"] <= p["M"] + p["lookup_extra_paths"]:
        raise ValueError("Lookup bins must fit within the final joint-path training set")
    if p["c2"] <= 0 or min(p["h"], p["a"]) < 0 or min(p["outer_rounds"], p["inner_rounds"]) < 1:
        raise ValueError("Search requires positive surge cost/round counts and nonnegative QoS costs")
    source = Path(__file__).resolve().parent
    signature = json.loads(json.dumps({"config": {k: v for k, v in p.items() if k != "workers"},
        "source": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                   for name in ("two_shift.py", "case_study.py", "numerical.py", "simulation.py", "formulas.py")}}))
    config = output / "run_config.json"
    required = ("summary.json", "benchmark_table.csv", "scenario_costs.csv", "policy_decisions.csv",
                "policy_information.json", "online_timing.csv", "search_paths.csv", "search_results.json", "search_trace.csv")
    if output.exists() and any(output.iterdir()):
        if not config.is_file() or json.loads(config.read_text()) != signature:
            raise ValueError(f"Existing two-shift training has different parameters/source: {output}")
        if (output / "run_complete.json").is_file() and all((output / name).is_file() for name in required):
            print(f"[Two shift] SKIP completed training {output}", flush=True)
            return json.loads((output / "summary.json").read_text())
    write_json(config, signature)
    (output / "run_complete.json").unlink(missing_ok=True)
    started, timings, task_results, traces = time.perf_counter(), {}, [], []
    output.mkdir(parents=True, exist_ok=True)
    rates = [joint_rate(p, np.random.default_rng(seed(p, 1, sid, stream=10))) for sid in range(p["M"])]
    initial = [qed_n1(lam, p["c1"], p["c2"], p["h"], p["mu"], p["theta"], p["a"], p["sigma"], p["alpha"]) for lam in p["lambda_means"]]
    uppers = [int(p.get("n1_upper", 2 * max(1, x))) for x in initial]
    selected, searches, candidate_times = {}, {}, []
    init_worker()
    executor = ProcessPoolExecutor(max_workers=p["workers"], initializer=init_worker) if use_parallel else None
    # Threads only coordinate adaptive searches; all expensive branches use one process pool.
    coordinators = ThreadPoolExecutor(max_workers=p["workers"]) if use_parallel else None
    print(f"[Two shift] workers={p['workers'] if use_parallel else 1}, branch-level scheduling, "
          f"M={p['M']}, J={p['J']}, R={p['R']}, offline K={p['K']}, "
          f"evaluation={p['eval_scenarios']}x{p['eval_replications']}", flush=True)
    try:
        for kind in ("seq", "aux"):
            begin, memo = time.perf_counter(), {}
            # One coordinator owns each scenario; candidates finish before the next starts.
            scenario_contexts = [context() for _ in rates] if kind == "seq" else []
            path_tables = [{} for _ in rates] if kind == "aux" else []

            def objective(n1: list[int]) -> float:
                key = tuple(n1)
                if key not in memo:
                    query_begin = time.perf_counter()
                    label = f"{kind} candidate {len(memo) + 1}, N1={n1}"
                    print(f"[Two shift] START {label}", flush=True)
                    if kind == "seq":
                        tasks = [(p, n1, path, sid) for sid, path in enumerate(rates)]
                        result = progress_map(lambda task: sequential_task(task, executor, scenario_contexts[task[-1]]), tasks, coordinators, label)
                    else:
                        aux_tasks = [(p, n1, path, sid, path_tables[sid]) for sid, path in enumerate(rates)]
                        result = progress_map(auxiliary_task, aux_tasks, executor, label)
                        for sid, row in enumerate(result):
                            path_tables[sid] = row.pop("shared_path")
                    if not memo:
                        timings[f"initial_n1_objective_{kind}"] = time.perf_counter() - query_begin
                    task_results.extend(result)
                    memo[key] = result
                    candidate_times.append({"method": kind, "candidate": len(memo), "n1_vec": str(n1),
                                            "objective": float(np.mean([r["cost"] for r in result])),
                                            "elapsed_sec": time.perf_counter() - query_begin,
                                            "search_elapsed_sec": time.perf_counter() - begin,
                                            **{stat: sum(r["stats"][stat] for r in result) for stat in STAT_KEYS}})
                    write_csv(output / "candidate_progress.csv", candidate_times)
                    print(f"[Two shift] END {label}: cost={candidate_times[-1]['objective']:.4f}, "
                          f"elapsed={candidate_times[-1]['elapsed_sec']:.1f}s", flush=True)
                return float(np.mean([r["cost"] for r in memo[key]]))

            n1, cost, trace = coordinate_search(initial, objective, p["outer_rounds"], uppers, 1)
            selected[kind] = n1
            searches[kind] = {"n1": n1, "objective": cost, "n1_candidates": len(memo)}
            traces.extend({"method": kind, **row} for row in trace)
            if kind == "aux":
                training = memo[tuple(n1)]
            timings[f"search_{kind}"] = time.perf_counter() - begin
            write_csv(output / "search_trace.csv", traces)
            write_json(output / "search_results.json", {"config": p, "search": searches, "timings_sec": timings})
            print(f"[Two shift] {kind}: N1={n1}, objective={cost:.4f}, elapsed={timings[f'search_{kind}']:.1f}s", flush=True)
        write_csv(output / "search_paths.csv", [
            {"source": "search", "scenario_id": sid, "lambda_1": path[0], "lambda_2": path[1],
             "n1_vec": str(selected["aux"]), "n2_1": training[sid]["n2"][0],
             "n2_2": training[sid]["n2"][1], "objective_value": training[sid]["cost"], "solver_stage": 31}
            for sid, path in enumerate(rates)])
    finally:
        if coordinators is not None:
            coordinators.shutdown()
        if executor is not None:
            executor.shutdown()
    evl = [joint_rate(p, np.random.default_rng(seed(p, 60, sid, stream=10))) for sid in range(p["eval_scenarios"])]
    begin = time.perf_counter()
    print("[Two shift] START OPT independent evaluation with fresh caches", flush=True)
    evaluation = evaluate_sequential(p, selected["seq"], evl, use_parallel)
    timings.update(evaluation["timings_sec"])
    timings["independent_evaluation"] = time.perf_counter() - begin
    task_results.append(evaluation)
    rows = evaluation["rows"]
    mean, low, high = mean_ci([r["total_cost"] for r in rows])
    bench = [{"benchmark": "SEQ_REF", "n1_vec": str(selected["seq"]), "mean_total_cost": mean,
              "ci95_lower": low, "ci95_upper": high,
              **{f"mean_{key}": float(np.mean([r[key] for r in rows])) for key in
                 ("base_cost", "surge_cost", "waiting_cost", "abandonment_cost", "avg_queue", "avg_wait_time", "lwbs_rate", "n2_1", "n2_2")}}]
    write_csv(output / "benchmark_table.csv", bench)
    write_csv(output / "scenario_costs.csv", rows)
    write_csv(output / "policy_decisions.csv", evaluation["decisions"])
    write_csv(output / "online_timing.csv", [
        {"benchmark": "SEQ_REF", "phase": phase, "elapsed_sec": timings[phase], "decision_count": count}
        for phase, count in (("online_first_shift", len(evl)),
                             ("online_second_shift", len(evl) * p["eval_replications"]),
                             ("online_total", len(evl) * (1 + p["eval_replications"])))])
    print("[Two shift] OPT evaluation and online timing saved", flush=True)
    queries = [row for result in task_results for row in result["queries"]]
    stats = {key: sum(result["stats"][key] for result in task_results) for key in task_results[0]["stats"]}
    write_csv(output / "decision_queries.csv", queries)
    write_csv(output / "boundary_diagnostics.csv", [{"kind": kind, "decision_count": len(group), "upper_hits": sum(r["upper_hit"] for r in group),
                                                    "upper_hit_rate": float(np.mean([r["upper_hit"] for r in group])),
                                                    "initial_upper_hits": sum(r["initial_upper_hit"] for r in group),
                                                    "expanded_queries": sum(r["upper_expansions"] > 0 for r in group),
                                                    "total_upper_expansions": sum(r["upper_expansions"] for r in group)}
                                                   for kind in dict.fromkeys(r["kind"] for r in queries) if (group := [r for r in queries if r["kind"] == kind])])
    write_json(output / "policy_information.json", {
        "SEQ_REF": "First: current rate and empty state. Second: current rate and observable own state; no latent residuals.",
        "conditional_state": "Service residuals follow the configured law conditional on received service; exponential service and patience are memoryless. Actual execution residuals are preserved.",
        "arrival_clock": "Ordinary renewal interval restarts each shift; actual service/patience residuals persist.",
        "sampling": "J paired state-rate branches, not J squared. All candidate first-shift capacities regenerate their own states.",
        "seeds": "Training rates use stage 1; sequential searches use 10-12/20-23; auxiliary searches use 31; evaluation rates/execution use 60/61. Within-stage stream IDs separate rates, service and patience.",
        "paired_statistics": "Scenario-paired differences are left minus right; mean_saving is right minus left. Negative values are retained.",
        "solver": "N1 has fixed bounds. N2 starts from QED; a binding upper bound is doubled using unchanged samples until the solution is interior. The same rule applies to SEQ and B/C pathwise solvers; no global-optimality certificate.",
    })
    timings["total"] = time.perf_counter() - started
    summary = {"config": p, "search": searches,
               "benchmarks": {r["benchmark"]: r for r in bench}, "timings_sec": timings,
               "timing_note": "Initial-N1 objective time is included in search time. OPT online time sums the two non-overlapping decision-batch wall times, not worker times or per-decision latency. Actual execution, task preparation and pool shutdown are excluded; dispatch, result collection and lazy worker startup inside each decision batch are included. Evaluation starts with fresh caches and retains them within OPT. Each online decision is solved once and then executed on independent stage-61 streams; actual residuals carry over unchanged.",
               "simulation_counts": stats}
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    return summary


def run_two(output: Path, p: dict | None = None, use_parallel: bool = True) -> dict:
    from two_shift_policies import run_pipeline

    return run_pipeline(output, dict(two_params() if p is None else p), use_parallel)
