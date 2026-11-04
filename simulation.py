from __future__ import annotations

import heapq
from functools import lru_cache
from itertools import count
from typing import Any

import numpy as np
from scipy.special import ndtri_exp
from scipy.stats import lognorm

SHIFT_DURATION = 1.0
PRIORITY_NORMAL = 0
PRIORITY_INTERRUPTED = 1
WAIT_OVER_10MIN = 10.0 / (12.0 * 60.0)


def rep_seed(dataset_seed: int, scenario_id: int, replication_id: int) -> int:
    return int(dataset_seed + 200_000 * int(scenario_id) + int(replication_id))


def shift_seed(replication_seed: int, shift: int) -> int:
    return int(replication_seed + 10_000 * int(shift))


@lru_cache(maxsize=10_000)
def build_shift_stream(lambda_fixed: float, mu: float, theta: float, seed: int) -> dict[str, tuple[float, ...]]:
    if float(lambda_fixed) <= 0.0:
        return {"arrival_offsets": (), "service_times": (), "patience_times": ()}
    rng = np.random.RandomState(int(seed))
    t = 0.0
    arrivals: list[float] = []
    services: list[float] = []
    patiences: list[float] = []
    while t < SHIFT_DURATION:
        t += float(rng.exponential(1.0 / float(lambda_fixed)))
        if t >= SHIFT_DURATION:
            break
        arrivals.append(float(t))
        services.append(float(rng.exponential(1.0 / float(mu))))
        patiences.append(float(rng.exponential(1.0 / float(theta))))
    return {"arrival_offsets": tuple(arrivals), "service_times": tuple(services), "patience_times": tuple(patiences)}


def build_shift_stream_crn(lambda_fixed: float, mu: float, theta: float, arrival_seed: int, service_seed: int, patience_seed: int) -> dict[str, tuple[float, ...]]:
    if float(lambda_fixed) <= 0.0:
        return {"arrival_offsets": (), "service_times": (), "patience_times": ()}
    arrival_rng = np.random.default_rng(int(arrival_seed))
    service_rng = np.random.default_rng(int(service_seed))
    patience_rng = np.random.default_rng(int(patience_seed))
    t = 0.0
    arrivals: list[float] = []
    services: list[float] = []
    patiences: list[float] = []
    while t < SHIFT_DURATION:
        t += float(arrival_rng.exponential(1.0 / float(lambda_fixed)))
        if t >= SHIFT_DURATION:
            break
        arrivals.append(float(t))
        services.append(float(service_rng.exponential(1.0 / float(mu))))
        patiences.append(float(patience_rng.exponential(1.0 / float(theta))))
    return {"arrival_offsets": tuple(arrivals), "service_times": tuple(services), "patience_times": tuple(patiences)}


def service_samples(rng: np.random.Generator, mu: float, distribution: str, scv: float, size: int) -> np.ndarray:
    if distribution == "exponential":
        return rng.exponential(1.0 / float(mu), size=size)
    if distribution == "lognormal" and scv > 0:
        variance = np.log1p(scv)
        return rng.lognormal(-np.log(mu) - variance / 2, np.sqrt(variance), size=size)
    raise ValueError("Service distribution must be exponential or lognormal with positive SCV")


def patience_samples(rng: np.random.Generator, theta: float, distribution: str, scv: float, size: int) -> np.ndarray:
    if distribution == "exponential":
        return rng.exponential(1.0 / float(theta), size=size)
    if distribution == "gamma" and np.isfinite(scv) and scv > 0:
        return rng.gamma(1.0 / scv, scv / float(theta), size=size)
    raise ValueError("Patience distribution must be exponential or gamma with positive finite SCV")


@lru_cache(maxsize=10_000)
def _lognormal_survival_parameters(mu: float, scv: float, received: float) -> tuple[float, float, float]:
    scale = float(np.sqrt(np.log1p(scv)))
    location = float(-np.log(mu) - scale * scale / 2)
    log_survival = float(lognorm.logsf(received, s=scale, scale=np.exp(location)))
    return scale, location, log_survival


def conditional_service_residual(rng: np.random.Generator, mu: float, distribution: str, scv: float, received: float) -> float:
    if received < 0:
        raise ValueError("Received service must be nonnegative")
    if distribution == "exponential":
        return float(rng.exponential(1.0 / mu))
    if distribution != "lognormal" or scv <= 0:
        raise ValueError("Unsupported conditional service distribution")
    # Invert the conditional survival probability in log space, including old survivors.
    scale, location, log_survival = _lognormal_survival_parameters(mu, scv, received)
    log_tail = np.log(max(float(rng.random()), np.finfo(float).tiny)) + log_survival
    full = float(np.exp(location - scale * ndtri_exp(log_tail)))
    return max(np.finfo(float).eps, full - received)


def build_model_stream(p: dict[str, Any], lam: float, arrival_seed: int, service_seed: int, patience_seed: int) -> dict[str, tuple[float, ...]]:
    arrival = p.get("arrival_distribution", "exponential")
    service = p.get("service_distribution", "exponential")
    patience_distribution = p.get("patience_distribution", "exponential")
    if arrival == service == patience_distribution == "exponential":
        return build_shift_stream_crn(lam, p["mu"], p["theta"], arrival_seed, service_seed, patience_seed)
    if p.get("renewal_boundary", "restart_each_shift") != "restart_each_shift":
        raise ValueError("Only a fresh ordinary renewal interval at each shift start is implemented")
    arrival_rng = np.random.default_rng(arrival_seed)
    offsets, now = [], 0.0
    while lam > 0:
        if arrival == "exponential":
            now += float(arrival_rng.exponential(1.0 / lam))
        elif arrival == "gamma" and p["arrival_scv"] > 0:
            scv = float(p["arrival_scv"])
            now += float(arrival_rng.gamma(1.0 / scv, scv / lam))
        else:
            raise ValueError("Arrival distribution must be exponential or gamma with positive SCV")
        if now >= SHIFT_DURATION:
            break
        offsets.append(now)
    services = service_samples(np.random.default_rng(service_seed), p["mu"], service, p.get("service_scv", 1.0), len(offsets))
    patience = patience_samples(np.random.default_rng(patience_seed), p["theta"], patience_distribution,
                                p.get("patience_scv", 1.0), len(offsets))
    return {"arrival_offsets": tuple(offsets), "service_times": tuple(map(float, services)), "patience_times": tuple(map(float, patience))}


def patient_started(patient: dict[str, Any]) -> bool:
    return bool(patient.get("has_started_service", False))


def lwbs_risk_count(state: list[dict[str, Any]]) -> int:
    return sum(1 for patient in state if not patient_started(patient))


def _clean_queue(queue: list[tuple[int, float, int, str]], patients: dict[str, dict[str, Any]]) -> None:
    while queue:
        _, _, _, pid = queue[0]
        patient = patients.get(pid)
        if patient is not None and patient.get("in_queue") and not patient.get("abandoned"):
            break
        heapq.heappop(queue)


def _queue_state(queue: list[tuple[int, float, int, str]], patients: dict[str, dict[str, Any]], now: float, track_service: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, _, _, pid in sorted(queue):
        patient = patients.get(pid)
        if patient is None or not patient.get("in_queue") or patient.get("abandoned"):
            continue
        abandon_time = patient.get("abandon_time")
        rows.append(
            {
                "priority": int(patient.get("priority", PRIORITY_NORMAL)),
                "arrival_rank": float(patient.get("arrival_rank", 0.0)),
                "service_time": patient.get("_service_time"),
                "has_started_service": bool(patient_started(patient)),
                "remaining_service_time": patient.get("remaining_service_time"),
                "remaining_patience_time": None if abandon_time is None else max(0.0, float(abandon_time) - float(now)),
                "wait_total": float(patient.get("wait_total", 0.0)),
                "wait_over_10min_counted": bool(patient.get("wait_over_10min_counted", False)),
            }
        )
        if track_service:
            rows[-1]["service_received"] = float(patient.get("service_received", 0.0))
    return rows


def simulate_shift(
    initial_state: list[dict[str, Any]],
    n1: int,
    n2: int,
    lambda_fixed: float,
    mu: float,
    theta: float,
    c1: float,
    c2: float,
    h: float,
    a: float,
    seed: int,
    stream: dict[str, tuple[float, ...]],
    strict_stream: bool = False,
    track_service: bool = False,
    cost_only: bool = False,
    balk_threshold: int | None = None,
    balk_penalty: float | None = None,
    quadratic_queue_cost: float | None = None,
) -> dict[str, Any]:
    if balk_threshold is not None and (isinstance(balk_threshold, bool) or int(balk_threshold) != balk_threshold or balk_threshold < 0):
        raise ValueError("balk_threshold must be a nonnegative integer or None")
    if balk_penalty is not None and (not np.isfinite(balk_penalty) or balk_penalty < 0):
        raise ValueError("balk_penalty must be nonnegative and finite or None")
    if quadratic_queue_cost is not None and (not np.isfinite(quadratic_queue_cost) or quadratic_queue_cost < 0):
        raise ValueError("quadratic_queue_cost must be nonnegative and finite or None")
    events: list[tuple[float, int, int, str, dict[str, Any]]] = []
    serial = count()
    queue: list[tuple[int, float, int, str]] = []
    patients: dict[str, dict[str, Any]] = {}
    active: dict[str, float] = {}
    rng = None if strict_stream else np.random.RandomState(int(seed))
    available = max(0, int(n1) + int(n2))
    queue_count = 0
    last_time = 0.0
    queue_area = 0.0
    queue_squared_area = 0.0
    arrivals = 0
    abandons = 0
    balks = 0
    wait_over = 0
    patient_count = len(initial_state)
    risk_count = lwbs_risk_count(initial_state)
    max_initial_rank = max((float(state.get("arrival_rank", idx)) for idx, state in enumerate(initial_state)), default=-1.0)

    def schedule(t: float, priority: int, kind: str, payload: dict[str, Any]) -> None:
        # Residual times live on patients/active; events beyond this shift never run.
        if t > SHIFT_DURATION:
            return
        heapq.heappush(events, (float(t), int(priority), next(serial), str(kind), payload))

    def update_area(t: float) -> None:
        nonlocal last_time, queue_area, queue_squared_area
        t = min(float(t), SHIFT_DURATION)
        if t > last_time:
            queue_area += queue_count * (t - last_time)
            if quadratic_queue_cost is not None:
                queue_squared_area += queue_count ** 2 * (t - last_time)
            last_time = t

    def mark_wait(patient: dict[str, Any], now: float, keep_waiting: bool) -> None:
        nonlocal wait_over
        if cost_only:
            return
        enter = patient.get("queue_enter_time")
        if enter is not None:
            patient["wait_total"] = float(patient.get("wait_total", 0.0)) + max(0.0, float(now) - float(enter))
            patient["queue_enter_time"] = float(now) if keep_waiting else None
        if not patient.get("wait_over_10min_counted") and float(patient.get("wait_total", 0.0)) > WAIT_OVER_10MIN:
            patient["wait_over_10min_counted"] = True
            wait_over += 1

    def push(pid: str) -> None:
        nonlocal queue_count
        patient = patients[pid]
        if patient.get("in_queue") and not patient.get("abandoned"):
            return
        patient["in_queue"] = True
        heapq.heappush(queue, (-int(patient.get("priority", PRIORITY_NORMAL)), float(patient.get("arrival_rank", 0.0)), next(serial), pid))
        queue_count += 1

    def pop() -> str | None:
        nonlocal queue_count
        _clean_queue(queue, patients)
        if not queue:
            return None
        _, _, _, pid = heapq.heappop(queue)
        patients[pid]["in_queue"] = False
        queue_count = max(0, queue_count - 1)
        return pid

    def assign(pid: str, now: float) -> None:
        nonlocal available
        patient = patients[pid]
        mark_wait(patient, now, keep_waiting=False)
        patient["has_started_service"] = True
        patient["abandon_time"] = None
        service_time = patient.get("remaining_service_time")
        if service_time is None or float(service_time) <= 0.0:
            service_time = patient.pop("_service_time", None)
            if service_time is None:
                if strict_stream:
                    raise ValueError("Missing service sample in strict-stream simulation")
                service_time = rng.exponential(1.0 / float(mu))
            service_time = float(service_time)
        patient["remaining_service_time"] = None
        end_time = float(now) + float(service_time)
        active[pid] = end_time
        if track_service:
            patient["service_started_at"] = float(now)
        available -= 1
        schedule(end_time, 2, "service_end", {"pid": pid, "end_time": end_time})

    def try_assign(now: float) -> None:
        nonlocal available
        while available > 0:
            pid = pop()
            if pid is None:
                break
            assign(pid, now)

    for idx, state in enumerate(initial_state):
        pid = f"I{idx + 1}"
        patients[pid] = {
            "id": pid,
            "arrival_time": 0.0,
            "arrival_rank": float(state.get("arrival_rank", idx)),
            "queue_enter_time": 0.0,
            "priority": int(state.get("priority", PRIORITY_NORMAL)),
            "has_started_service": patient_started(state),
            "_service_time": state.get("service_time"),
            "remaining_service_time": state.get("remaining_service_time"),
            "remaining_patience_time": state.get("remaining_patience_time"),
            "wait_total": float(state.get("wait_total", 0.0) or 0.0),
            "wait_over_10min_counted": bool(state.get("wait_over_10min_counted", False)),
            "abandoned": False,
            "in_queue": False,
        }
        if track_service:
            patients[pid]["service_received"] = float(state.get("service_received", 0.0))
        if not patients[pid]["has_started_service"] and state.get("remaining_patience_time") is not None:
            abandon_time = max(0.0, float(state["remaining_patience_time"]))
            patients[pid]["abandon_time"] = abandon_time
            schedule(abandon_time, 3, "abandon", {"pid": pid})
        else:
            patients[pid]["abandon_time"] = None
        push(pid)

    for i, offset in enumerate(stream["arrival_offsets"]):
        schedule(float(offset), 1, "arrival", {"service_time": stream["service_times"][i], "patience_time": stream["patience_times"][i]})
    try_assign(0.0)

    while events:
        t, _, _, kind, payload = heapq.heappop(events)
        if t > SHIFT_DURATION:
            break
        update_area(t)
        if kind == "arrival":
            arrivals += 1
            patient_count += 1
            risk_count += 1
            # Test the live waiting count before admission, including interrupted patients.
            if balk_threshold is not None and queue_count > balk_threshold:
                balks += 1
                continue
            pid = f"A{arrivals}"
            patience = float(payload["patience_time"])
            patients[pid] = {
                "id": pid,
                "arrival_time": float(t),
                "arrival_rank": float(max_initial_rank + arrivals),
                "queue_enter_time": float(t),
                "priority": PRIORITY_NORMAL,
                "has_started_service": False,
                "remaining_service_time": None,
                "wait_total": 0.0,
                "wait_over_10min_counted": False,
                "abandoned": False,
                "in_queue": False,
                "_service_time": float(payload["service_time"]),
                "abandon_time": float(t) + patience,
            }
            if available > 0 and queue_count == 0:
                assign(pid, t)
            else:
                push(pid)
                schedule(float(t) + patience, 3, "abandon", {"pid": pid})
                try_assign(t)
        elif kind == "service_end":
            pid = str(payload["pid"])
            if pid not in active or abs(active[pid] - float(payload["end_time"])) > 1e-12:
                continue
            del active[pid]
            available += 1
            try_assign(t)
        elif kind == "abandon":
            pid = str(payload["pid"])
            patient = patients.get(pid)
            if patient is None or patient.get("abandoned") or patient.get("has_started_service") or not patient.get("in_queue"):
                continue
            patient["in_queue"] = False
            queue_count = max(0, queue_count - 1)
            patient["abandoned"] = True
            mark_wait(patient, t, keep_waiting=False)
            abandons += 1

    update_area(SHIFT_DURATION)
    base_cost = float(c1) * int(n1)
    surge_cost = float(c2) * int(n2)
    waiting_cost = float(h) * queue_area
    queue_metrics = {}
    if quadratic_queue_cost is not None:
        quadratic_cost = float(quadratic_queue_cost) * queue_squared_area
        queue_metrics = {"queue_squared_area": queue_squared_area,
                         "linear_waiting_cost": waiting_cost, "quadratic_waiting_cost": quadratic_cost}
        waiting_cost += quadratic_cost
    abandonment_cost = float(a) * abandons
    balk_cost = float(a if balk_penalty is None else balk_penalty) * balks
    balk_metrics = {"balk_count": int(balks), "balk_cost": balk_cost} if balk_threshold is not None else {}
    if cost_only:
        # Terminal search replications need costs, not patient histories or carry-over.
        return {"total_cost": base_cost + surge_cost + waiting_cost + abandonment_cost + balk_cost,
                "base_cost": base_cost, "surge_cost": surge_cost,
                "waiting_cost": waiting_cost, "abandonment_cost": abandonment_cost,
                "queue_area": float(queue_area), "abandon_count": int(abandons), **balk_metrics, **queue_metrics}
    for patient in patients.values():
        if patient.get("in_queue") and not patient.get("abandoned"):
            mark_wait(patient, SHIFT_DURATION, keep_waiting=True)

    final_state = _queue_state(queue, patients, SHIFT_DURATION, track_service)
    for pid, end_time in active.items():
        patient = patients[pid]
        final_state.append(
            {
                "priority": PRIORITY_INTERRUPTED,
                "arrival_rank": float(patient.get("arrival_rank", 0.0)),
                "service_time": None,
                "has_started_service": True,
                "remaining_service_time": max(0.0, float(end_time) - SHIFT_DURATION),
                "remaining_patience_time": None,
                "wait_total": float(patient.get("wait_total", 0.0)),
                "wait_over_10min_counted": bool(patient.get("wait_over_10min_counted", False)),
            }
        )

        if track_service:
            final_state[-1]["service_received"] = float(patient.get("service_received", 0.0)) + SHIFT_DURATION - patient["service_started_at"]

    lwbs_rate = float((abandons + balks) / risk_count) if risk_count > 0 else 0.0
    if balk_threshold is not None:
        balk_metrics.update(admitted_arrival_count=int(arrivals - balks),
                            balk_rate=float(balks / arrivals) if arrivals else 0.0)
    return {
        "total_cost": base_cost + surge_cost + waiting_cost + abandonment_cost + balk_cost,
        "base_cost": base_cost,
        "surge_cost": surge_cost,
        "waiting_cost": waiting_cost,
        "abandonment_cost": abandonment_cost,
        "queue_area": float(queue_area),
        "abandon_count": int(abandons),
        "arrival_count": int(arrivals),
        "patient_count": int(patient_count),
        "lwbs_risk_count": int(risk_count),
        "wait_over_10min_count": int(wait_over),
        "avg_queue": float(queue_area),
        "avg_wait_time": float(queue_area / patient_count) if patient_count > 0 else 0.0,
        "lwbs_rate": float(lwbs_rate),
        "final_state": final_state,
        **balk_metrics,
        **queue_metrics,
    }


def pool_by_id(pool: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["pool_id"]): item for item in pool}


def pool_by_shift(pool: list[dict[str, Any]], horizon: int) -> dict[int, list[dict[str, Any]]]:
    grouped = {shift: [] for shift in range(int(horizon))}
    for item in pool:
        grouped[int(item["shift_type"])].append(item)
    return grouped
