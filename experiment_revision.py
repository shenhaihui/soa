"""Shared parameters and output handling for the six manuscript experiments."""
from __future__ import annotations

import ast
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from numerical import numerical_params, run_single, write_json

ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
SOURCE_FILES = ("experiment_revision.py", "numerical.py", "simulation.py", "formulas.py", "figures.py")


def single_params(pilot: bool = False) -> dict:
    p = numerical_params()
    p.update({
        "initialization": "fixed_count", "initial_patient_count": 0,
        "include_1nv": True, "include_qed_n2_search": True, "include_so_lookup": True,
        "record_terminal_counts": True, "record_search_bounds": True, "strict_stream": True,
        "arrival_distribution": "exponential", "arrival_scv": 1.0,
        "service_distribution": "exponential", "service_scv": 1.0,
        "initial_residual_distribution": "exponential", "initial_residual_scv": 1.0,
        "renewal_boundary": "restart_each_shift",
        "seed_namespace": 2026102801 if pilot else 2026102802,
        "search_seed": 1, "online_n2_seed": 2, "eval_seed": 3, "lookup_seed": 4,
        "lookup_scenarios": 1000, "lookup_bins": 20,
    })
    if pilot:
        p.update(search_scenarios=4, eval_scenarios=4, replications=2,
                 lookup_scenarios=4, lookup_bins=2, workers=2)
    return p


def case_settings(pilot: bool = False, rhos: tuple[float, ...] = (0.3, 0.0)) -> list[tuple[str, dict]]:
    from case_study import case_params

    p = case_params()
    p.update(initialization="empty", seed_namespace=2026102810 if pilot else 2026102811,
             search_seed=1, eval_seed=2, eval_recourse_seed=3, lhs_stream_base=4, lhs_seed=5,
             record_search_bounds=True, lookup_full_domain=True)
    if pilot:
        p.update(search_scenarios=2, search_replications=2, eval_scenarios=2, eval_replications=2,
                 lhs_scenarios=2, lhs_replications=2, lhs_bin_counts=(2,), outer_limit=1, inner_limit=2,
                 workers=2)
    return [(f"rho{rho:g}_sigma2{variance:g}", {**p, "rho": rho, "sigma2": variance})
            for rho in rhos for variance in (0.118, 0.236, 0.354)]


def write_latex(output: Path, summary: dict, kind: str) -> None:
    labels = {"1,NV": r"$\mathcal{P}_{1,\mathrm{NV}}$", "1,SO": r"$\mathcal{P}_{1,\mathrm{SOA}}$",
              "2,QED": r"$\mathcal{P}_{2,\mathrm{QED}}$", "2,SO-Direct": r"$\mathcal{P}_{2,\mathrm{SOA}}$-D",
              "2,SO-Lookup": r"$\mathcal{P}_{2,\mathrm{SOA}}$", "JSO": r"$\mathcal{P}_{2,\mathrm{SOA}}$",
              "QED-N1+SO-N2": r"$\mathcal{P}_{\mathrm{QED+SOA}}$",
              "SEQ_REF": "OPT", "COND_MEAN": r"$\mathcal{P}_{2,\mathrm{SOA}}$-D",
              "BIN_LOOKUP": r"$\mathcal{P}_{2,\mathrm{SOA}}$"}
    lines = [r"\begin{tabular}{lrrr}", r"\hline",
             r"Policy & Average total cost & Base staffing (sum) & Average surge per shift \\", r"\hline"]
    for label, row in summary["benchmarks"].items():
        if kind == "single":
            cost, n1, n2 = row["MeanTotalCost"], row["N1"], row["AvgN2"]
        else:
            cost, n1 = row["mean_total_cost"], sum(ast.literal_eval(row["n1_vec"]))
            n2 = float(np.mean(ast.literal_eval(row["avg_n2_vec"]))) if kind == "case" else (row["mean_n2_1"] + row["mean_n2_2"]) / 2
        display = labels.get(label, label.replace("_", r"\_"))
        lines.append(f"{display} & {cost:.3f} & {n1} & {n2:.3f} " + r"\\")
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "benchmark_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_point(output: Path, p: dict, kind: str, purpose: str = "formal", use_parallel: bool = True) -> dict:
    extra = {"single": (), "case": ("case_study.py",), "two": ("case_study.py", "two_shift.py")}[kind]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (*SOURCE_FILES, *extra)}
    signature = json.loads(json.dumps({"purpose": purpose, "kind": kind,
        "config": {k: v for k, v in p.items() if k != "workers"}, "source_sha256": hashes}))
    config = output / "run_config.json"
    required = ("summary.json", "benchmark_table.csv", "scenario_costs.csv", "paired_comparison.csv", *{
        "single": ("lookup.csv", "lookup_points.csv", "policy_decisions.csv", "boundary_diagnostics.csv"),
        "case": ("jso_pathwise_n2_points.csv", "oracle_benchmark.csv", "oracle_scenario_costs.csv",
                 *(f"jso_lookup_{bins}bins.csv" for bins in p.get("lhs_bin_counts", ()))),
        "two": ("lookup.csv", "lookup_points.csv", "search_results.json", "search_trace.csv",
                "policy_decisions.csv", "policy_information.json", "approximation_gap_decomposition.csv"),
    }[kind])
    if output.exists() and any(output.iterdir()):
        if not config.is_file() or json.loads(config.read_text()) != signature:
            raise ValueError(f"Existing output has different parameters/source; use a new directory: {output}")
        if (output / "run_complete.json").is_file() and all(
                (output / name).is_file() and (output / name).stat().st_size for name in required):
            print(f"[Experiment] SKIP completed {output}", flush=True)
            return json.loads((output / "summary.json").read_text())
        print(f"[Experiment] Restart unfinished {output}; Case can reuse its LHS checkpoint", flush=True)
        (output / "run_complete.json").unlink(missing_ok=True)
    write_json(config, signature)
    print(f"[Experiment] START {kind}: {output}", flush=True)
    if kind == "single":
        summary = run_single(output, use_parallel, p)
    elif kind == "case":
        from case_study import run_case

        summary = run_case(output, use_parallel, p)
    else:
        from two_shift import run_two

        summary = run_two(output, p, use_parallel)
    summary.update(source_sha256=hashes, runtime={"python": platform.python_version(),
                   "numpy": np.__version__, "scipy": scipy.__version__})
    write_latex(output, summary, kind)
    write_json(output / "summary.json", summary)
    write_json(output / "run_complete.json", signature)
    print(f"[Experiment] END {kind}: {output}", flush=True)
    return summary
