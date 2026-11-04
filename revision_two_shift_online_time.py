"""Time only the retained OPT policy's online searches, without repeating N1 search."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from numerical import write_csv, write_json
from two_shift import (
    context, init_worker, observable_state, progress_map, shift_cost,
    solve_first, solve_second, worker_context,
)

ROOT = Path(__file__).resolve().parent
SOURCE_OUTPUT = ROOT / "output/experiment5/lookup_1000"
OUTPUT_ROOT = ROOT / "output/experiment5/opt_online_timing"
WORKERS = max(1, (os.cpu_count() or 2) // 2)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_decision(task: tuple, executor: ProcessPoolExecutor | None) -> dict:
    p, n1, lam, sid = task
    n2 = solve_first(p, n1, lam, 20, sid, context(), executor)[0]
    return {"scenario_id": sid, "n2": n2}


def first_execution(task: tuple) -> dict:
    p, n1, lam, sid, rep, n2 = task
    result = shift_cost(p, [], n1[0] + n2, lam, 61, sid, 0, rep, 0, worker_context(p))
    return {"scenario_id": sid, "replication_id": rep,
            "observed": observable_state(result["final_state"])}


def second_decision(task: tuple) -> dict:
    p, n1, lam, sid, rep, observed = task
    n2 = solve_second(p, n1, observed, lam, 23, sid, rep, worker_context(p))[0]
    return {"scenario_id": sid, "replication_id": rep, "shift": 1,
            "lambda_fixed": lam, "initial_patients": len(observed), "n1": n1[1], "n2": n2}


def time_online(p: dict, n1: list[int], rates: dict[int, list[float]], use_parallel: bool) -> tuple[list[dict], dict]:
    init_worker()
    executor = ProcessPoolExecutor(max_workers=p["workers"], initializer=init_worker) if use_parallel else None
    coordinators = ThreadPoolExecutor(max_workers=p["workers"]) if use_parallel else None
    timings = {}
    try:
        tasks = [(p, n1, rates[sid][0], sid) for sid in sorted(rates)]
        begin = time.perf_counter()
        first = progress_map(lambda task: first_decision(task, executor), tasks, coordinators, "OPT online shift 1")
        timings["online_first_shift"] = time.perf_counter() - begin
        n2 = {r["scenario_id"]: r["n2"] for r in first}

        # Actual execution is needed to observe carry-over, but is not an online search.
        tasks = [(p, n1, rates[sid][0], sid, rep, n2[sid])
                 for sid in sorted(rates) for rep in range(p["eval_replications"])]
        begin = time.perf_counter()
        states = progress_map(first_execution, tasks, executor, "OPT carry-over generation (not timed as search)")
        timings["first_shift_execution"] = time.perf_counter() - begin

        tasks = [(p, n1, rates[r["scenario_id"]][1], r["scenario_id"], r["replication_id"], r["observed"])
                 for r in states]
        begin = time.perf_counter()
        second = progress_map(second_decision, tasks, executor, "OPT online shift 2")
        timings["online_second_shift"] = time.perf_counter() - begin
    finally:
        if coordinators is not None:
            coordinators.shutdown()
        if executor is not None:
            executor.shutdown()
    timings["online_total"] = timings["online_first_shift"] + timings["online_second_shift"]
    first_rows = [{"scenario_id": sid, "replication_id": rep, "shift": 0,
                   "lambda_fixed": rates[sid][0], "initial_patients": 0, "n1": n1[0], "n2": n2[sid]}
                  for sid in sorted(rates) for rep in range(p["eval_replications"])]
    return sorted(first_rows + second, key=lambda r: (r["scenario_id"], r["replication_id"], r["shift"])), timings


def run(source: Path = SOURCE_OUTPUT, output: Path = OUTPUT_ROOT, use_parallel: bool = True) -> dict:
    saved = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    p = {**saved["config"], "workers": WORKERS}
    n1 = saved["search"]["seq"]["n1"]
    rates = {int(r["scenario_id"]): [float(r["lambda_1"]), float(r["lambda_2"])]
             for r in read_csv(source / "scenario_costs.csv") if r["benchmark"] == "SEQ_REF"}
    original = {(int(r["scenario_id"]), int(r["replication_id"]), int(r["shift"])):
                (float(r["lambda_fixed"]), int(r["initial_patients"]), int(r["n1"]), int(r["n2"]))
                for r in read_csv(source / "policy_decisions.csv") if r["benchmark"] == "SEQ_REF"}
    if p["initialization"] != "empty" or len(rates) != p["eval_scenarios"]:
        raise ValueError("Use the completed empty-start two-shift experiment")
    expected_keys = {(sid, rep, shift) for sid in rates
                     for rep in range(p["eval_replications"]) for shift in range(2)}
    if set(original) != expected_keys:
        raise ValueError("The source OPT decisions are incomplete")
    signature = {
        "inputs": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                   for name in ("summary.json", "scenario_costs.csv", "policy_decisions.csv")},
        "source": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                   for name in (Path(__file__).name, "two_shift.py", "numerical.py", "simulation.py", "formulas.py")},
        "workers": p["workers"] if use_parallel else 1,
    }
    config = output / "run_config.json"
    if output.exists() and any(output.iterdir()):
        if not config.is_file() or json.loads(config.read_text()) != signature:
            raise ValueError(f"Existing OPT timing has different inputs/source/workers: {output}")
        if (output / "run_complete.json").is_file() and all((output / name).is_file() and (output / name).stat().st_size
               for name in ("summary.json", "policy_decisions.csv", "online_timing.csv")):
            print(f"[OPT online timing] SKIP completed {output}", flush=True)
            return json.loads((output / "summary.json").read_text())
        print("[OPT online timing] Recomputing unfinished timing stage", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_complete.json").unlink(missing_ok=True)
    write_json(config, signature)
    print(f"[OPT online timing] Retain N1={n1}; {len(rates)} scenarios x "
          f"{p['eval_replications']} replications; workers={p['workers'] if use_parallel else 1}", flush=True)
    started = time.perf_counter()
    decisions, timings = time_online(p, n1, rates, use_parallel)
    for row in decisions:
        key = (row["scenario_id"], row["replication_id"], row["shift"])
        actual = tuple(row[k] for k in ("lambda_fixed", "initial_patients", "n1", "n2"))
        if actual != original[key]:
            raise ValueError(f"OPT decision differs from the retained result at {key}: {actual} != {original[key]}")
    write_csv(output / "policy_decisions.csv", [{"benchmark": "SEQ_REF", **r} for r in decisions])
    write_csv(output / "online_timing.csv", [
        {"benchmark": "SEQ_REF", "phase": phase, "elapsed_sec": timings[phase], "decision_count": count}
        for phase, count in (("online_first_shift", len(rates)),
                             ("online_second_shift", len(rates) * p["eval_replications"]),
                             ("online_total", len(rates) * (1 + p["eval_replications"])))])
    summary = {
        "source_result": str(source.resolve()), "config": p, "retained_n1": n1,
        "actual_workers": p["workers"] if use_parallel else 1,
        "timings_sec": {**timings, "total": time.perf_counter() - started},
        "decisions_match_source": True,
        "provenance": signature,
        "timing_note": "Online time is the sum of two non-overlapping decision-batch wall times, not summed "
                       "worker times or per-decision latency. First-shift searches use stages 20-22; second-shift "
                       "searches use stage 23 and the actual observable stage-61 carry-over. Actual first-shift "
                       "execution, task preparation and pool shutdown are excluded; dispatch, result collection "
                       "and lazy worker startup inside the timed batches are included. Worker caches start fresh "
                       "for this benchmark and remain enabled. N1 search and other policies are not rerun.",
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    print(f"[OPT online timing] First={timings['online_first_shift']:.2f}s, "
          f"second={timings['online_second_shift']:.2f}s, total={timings['online_total']:.2f}s; "
          "all decisions match the saved OPT result", flush=True)
    return summary


if __name__ == "__main__":
    run()
