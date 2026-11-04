from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "1,NV": "#4d596d",
    "1,SO": "#ad7c70",
    "2,QED": "#789383",
    "2,SO": "#c2a66d",
    "2,SO-Direct": "#c2a66d",
    "2,SO-Lookup": "#927643",
    "QED-N1+SO-N2": "#536e63",
    "JSO": "#c7a095",
}
POLICY_LABELS = {
    "1,NV": r"$\pi_{1,NV}$",
    "1,SO": r"$\pi_{1,SO}$",
    "2,QED": r"$\pi_{2,QED}$",
    "2,SO": r"$\pi_{2,SO}$",
    "2,SO-Direct": r"$\pi_{2,SO}^{\mathrm{Direct}}$",
    "2,SO-Lookup": r"$\pi_{2,SO}^{\mathrm{Lookup}}$",
    "QED-N1+SO-N2": r"$\pi_{\mathrm{QED+SO}}$",
    "JSO": r"$\pi_{JSO}$",
}


def policy_label(name: str) -> str:
    return POLICY_LABELS[name]


def save_cost_decomposition(rows: list[dict[str, object]], output: Path, scale: float = 1.0) -> None:
    names = [str(row["benchmark"]) for row in rows]
    labels = [policy_label(name) for name in names]
    metrics = [
        ("mean_total_cost", "Total Cost"),
        ("mean_base_cost", "Base Cost"),
        ("mean_surge_cost", "Surge Cost"),
        ("mean_qos_cost", "QoS Cost"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5) if len(names) > 5 else (12, 3.2))
    for ax, (key, title) in zip(axes, metrics):
        vals = [float(row[key] if key != "mean_qos_cost" or key in row else float(row["mean_waiting_cost"]) + float(row["mean_abandonment_cost"])) / scale for row in rows]
        ax.bar(labels, vals, color=[COLORS.get(name, "#777777") for name in names], edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        for idx, val in enumerate(vals):
            ax.text(idx, val, f"{val:.1f}" if scale != 1.0 else f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("Average Cost" if scale == 1.0 else "Average Cost (thousand)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_qos_ecdf(rows: list[dict[str, object]], output: Path) -> None:
    metrics = [
        ("avg_queue", "Average Queue Length", "Average Queue Length"),
        ("avg_wait_time", "Average Waiting Time", "Average Waiting Time"),
        ("lwbs_rate", "LWBS Rate (%)", "LWBS Rate (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    labels = list(dict.fromkeys(str(row["benchmark"]) for row in rows))
    for ax, (key, title, xlabel) in zip(axes, metrics):
        for label in labels:
            vals = sorted(float(row[key]) * (100.0 if key == "lwbs_rate" else 1.0) for row in rows if row["benchmark"] == label)
            if not vals:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", label=policy_label(label), color=COLORS[label], linewidth=1.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("ECDF")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_round_gain(rows: list[dict[str, object]], output: Path) -> None:
    x = [0, *[int(row["coordinate"]) for row in rows]]
    y = [float(rows[0]["objective_before"]), *[float(row["objective_after"]) for row in rows]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, color="#4d596d", marker="o", markersize=3, linewidth=1.5)
    for boundary in (14, 28):
        ax.axvline(boundary, color="#ad7c70", linestyle="--", linewidth=1)
    ax.set_xlabel("Coordinate Update")
    ax.set_ylabel("SAA Search Objective")
    ax.set_xticks(range(0, 43, 7))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_sensitivity(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["1,NV", "1,SO", "2,QED", "2,SO"]
    sigma_rows = [row for row in rows if row["sweep"] == "x_variance"]
    if sigma_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label in labels:
            pts = sorted((float(row["value"]), float(row["mean_total_cost"]), float(row["ci95_lower"]), float(row["ci95_upper"])) for row in sigma_rows if row["benchmark"] == label)
            if pts:
                x, mean, low, high = map(list, zip(*pts))
                ax.plot(x, mean, marker="o", label=policy_label(label), color=COLORS[label])
                ax.fill_between(x, low, high, color=COLORS[label], alpha=0.12)
        ax.set_xlabel("Var(X)")
        ax.set_ylabel("Mean Total Cost")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "sigma_sweep_total_cost.png", dpi=200)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, sweep in zip(axes, ("h_a", "c2", "alpha")):
        for label in labels:
            pts = sorted((float(row["value"]), float(row["mean_total_cost"]), float(row["ci95_lower"]), float(row["ci95_upper"])) for row in rows if row["sweep"] == sweep and row["benchmark"] == label)
            if pts:
                x, mean, low, high = map(list, zip(*pts))
                ax.plot(x, mean, marker="o", label=policy_label(label), color=COLORS[label])
                ax.fill_between(x, low, high, color=COLORS[label], alpha=0.12)
        ax.set_title(sweep)
        ax.set_xlabel(sweep)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean Total Cost")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output_dir / "sensitivity_total_cost.png", dpi=200)
    plt.close(fig)
