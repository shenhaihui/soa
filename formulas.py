from __future__ import annotations

import math
from functools import lru_cache

from scipy.optimize import minimize_scalar
from scipy.stats import norm


def hazard_rate(x: float) -> float:
    log_survival = float(norm.logsf(float(x)))
    if not math.isfinite(log_survival):
        return float("inf")
    return float(math.exp(float(norm.logpdf(float(x))) - log_survival))


def beta_star(c1: float, c2: float, sigma: float) -> float:
    return float(norm.ppf(1.0 - float(c1) / float(c2), loc=0.0, scale=float(sigma)))


def beta_newsvendor(c1: float, h: float, mu: float, theta: float, a: float, sigma: float) -> float:
    ratio = float(c1) / (float(h) * float(mu) / float(theta) + float(a) * float(mu))
    return float(norm.ppf(1.0 - ratio, loc=0.0, scale=float(sigma)))


def eta_objective(eta: float, h: float, mu: float, theta: float, a: float, c2: float) -> float:
    root = math.sqrt(float(theta) / float(mu))
    scaled = float(eta) * math.sqrt(float(mu) / float(theta))
    numerator = root * (hazard_rate(scaled) - scaled)
    denominator = 1.0 + root * hazard_rate(scaled) / hazard_rate(-float(eta))
    penalty = (float(h) * float(mu) / float(theta) + float(a) * float(mu))
    second = penalty * numerator / denominator if denominator != 0 else float("inf")
    return float(float(c2) * float(eta) + second)


@lru_cache(maxsize=None)
def eta_star(h: float, mu: float, theta: float, a: float, c2: float) -> float:
    result = minimize_scalar(
        lambda eta: eta_objective(float(eta), h, mu, theta, a, c2),
        bounds=(-5.0, 5.0),
        method="bounded",
    )
    return float(result.x)


def base_staffing(lambda_mean: float, mu: float, alpha: float, beta: float, eta: float) -> float:
    if float(lambda_mean) <= 0.0:
        return 0.0
    load = float(lambda_mean) / float(mu)
    return float(load + float(beta) * load ** float(alpha) + float(eta) * math.sqrt(load))


def newsvendor_n1(lambda_mean: float, c1: float, h: float, mu: float, theta: float, a: float, sigma: float, alpha: float) -> int:
    load = float(lambda_mean) / float(mu)
    beta = beta_newsvendor(c1, h, mu, theta, a, sigma)
    return max(1, int(round(load + beta * load ** float(alpha))))


def qed_n1(lambda_mean: float, c1: float, c2: float, h: float, mu: float, theta: float, a: float, sigma: float, alpha: float) -> int:
    beta = beta_star(c1, c2, sigma)
    eta = eta_star(h, mu, theta, a, c2)
    return max(1, int(round(base_staffing(lambda_mean, mu, alpha, beta, eta))))


def qed_n2(lambda_fixed: float, n1_qed: int, h: float, mu: float, theta: float, a: float, c2: float) -> int:
    if float(lambda_fixed) <= 0.0:
        return 0
    load = float(lambda_fixed) / float(mu)
    target = load + eta_star(h, mu, theta, a, c2) * math.sqrt(load)
    return max(0, int(round(target)) - int(n1_qed))


def mmk_queue(lambda_fixed: float, mu: float, theta: float, servers: int, tol: float = 1e-10, max_iter: int = 10000) -> float:
    k = int(servers)
    lam = float(lambda_fixed)
    mu = float(mu)
    theta = float(theta)
    if lam <= 0.0:
        return 0.0
    if k <= 0 or mu <= 0.0 or theta < 0.0:
        raise ValueError("mmk_queue requires positive servers and mu, and nonnegative theta")
    if theta == 0.0 and lam >= k * mu:
        raise ValueError("M/M/k has no stationary distribution when lambda >= k * mu")

    log_weights = [0.0]
    peak = 0.0
    for n in range(1, int(max_iter) + 1):
        death_rate = min(n, k) * mu + max(0, n - k) * theta
        log_weight = log_weights[-1] + math.log(lam) - math.log(death_rate)
        log_weights.append(log_weight)
        peak = max(peak, log_weight)

        next_death_rate = min(n + 1, k) * mu + max(0, n + 1 - k) * theta
        next_ratio = lam / next_death_rate
        if n >= k and next_ratio < 1.0:
            relative_tail_bound = math.exp(log_weight - peak) * next_ratio / (1.0 - next_ratio)
            if relative_tail_bound < float(tol):
                break
    else:
        raise RuntimeError("mmk_queue did not converge within max_iter")

    weights = [math.exp(value - peak) for value in log_weights]
    normalizer = sum(weights)
    return float(sum(max(0, n - k) * weight for n, weight in enumerate(weights)) / normalizer)


def adjusted_n1_vec(lambda_means: tuple[float, ...], n1_qed: list[int], mu: float, theta: float, xi1: float) -> list[int]:
    qbars = [mmk_queue(float(lam), mu, theta, int(n1)) for lam, n1 in zip(lambda_means, n1_qed)]
    return [
        max(1, int(round(int(n1_qed[i]) + float(xi1) * (qbars[(i - 1) % len(qbars)] - qbars[i]))))
        for i in range(len(n1_qed))
    ]


def adjusted_n2(lambda_fixed: float, n1_adj: int, n1_qed: int, state_size: int, h: float, mu: float, theta: float, a: float, c2: float, xi2: float) -> int:
    base_n2 = qed_n2(lambda_fixed, n1_qed, h, mu, theta, a, c2)
    baseline = int(n1_adj) + int(base_n2)
    qbar = mmk_queue(lambda_fixed, mu, theta, baseline)
    excess = max(0, int(state_size) - baseline)
    return max(0, int(round(base_n2 + float(xi2) * (excess - qbar))))
