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
    "1,NV": r"$\mathcal{P}_{1,\mathrm{NV}}$",
    "1,SO": r"$\mathcal{P}_{1,\mathrm{SOA}}$",
    "2,QED": r"$\mathcal{P}_{2,\mathrm{QED}}$",
    "2,SO": r"$\mathcal{P}_{2,\mathrm{SOA}}$-D",
    "2,SO-Direct": r"$\mathcal{P}_{2,\mathrm{SOA}}$-D",
    "2,SO-Lookup": r"$\mathcal{P}_{2,\mathrm{SOA}}$",
    "QED-N1+SO-N2": r"$\mathcal{P}_{\mathrm{QED+SOA}}$",
    "JSO": r"$\mathcal{P}_{2,\mathrm{SOA}}$",
    "SEQ_REF": r"$\mathcal{P}_{2,\mathrm{OPT}}$",
    "COND_MEAN": r"$\mathcal{P}_{2,\mathrm{SOA}}$-D",
    "BIN_LOOKUP": r"$\mathcal{P}_{2,\mathrm{SOA}}$",
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
    style = {
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 4, figsize=(max(6.5, 1.625 * len(names)), 2.55))
        fig.subplots_adjust(left=0.10, right=0.995, bottom=0.22, top=0.86, wspace=0.32)
        for ax, (key, title) in zip(axes, metrics):
            vals = [float(row[key] if key != "mean_qos_cost" or key in row else float(row["mean_waiting_cost"]) + float(row["mean_abandonment_cost"])) / scale for row in rows]
            bars = ax.bar(range(len(names)), vals, color=[COLORS.get(name, "#777777") for name in names],
                          edgecolor="black", linewidth=0.6, width=0.8)
            if ax is axes[0]:
                handles = bars
            ax.set_xticks([])
            ax.set_title(title, fontsize=10, pad=6)
            ax.tick_params(axis="both", labelsize=8.5, length=3, pad=2)
            ax.margins(y=0.25 if key == "mean_base_cost" else 0.18)
            for idx, (bar, val) in enumerate(zip(bars, vals)):
                vertical = (11, 3, 3, 7)[idx] if key == "mean_base_cost" and len(names) == 4 else 3
                center = bar.get_x() + bar.get_width() / 2
                ax.annotate(f"{val:.1f}" if scale != 1.0 else f"{val:.2f}", (center, val),
                            xytext=(0, vertical), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("Average Cost" if scale == 1.0 else "Average Cost (thousand)",
                           fontsize=9, labelpad=4)
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.55, 0.015),
                   ncol=len(names), frameon=False, fontsize=9, handlelength=1.1,
                   handletextpad=0.5, columnspacing=1.7)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=400)
        plt.close(fig)


def save_qos_ecdf(rows: list[dict[str, object]], output: Path) -> None:
    metrics = [
        ("avg_queue", "Average Queue Length", "Average Queue Length"),
        ("avg_wait_time", "Average Waiting Time", "Average Waiting Time"),
        ("lwbs_rate", "LWBS Rate (%)", "LWBS Rate (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    labels = list(dict.fromkeys(str(row["benchmark"]) for row in rows))
    for ax, (key, title, xlabel) in zip(axes, metrics):
        for label in labels:
            vals = sorted(float(row[key]) * (100.0 if key == "lwbs_rate" else 1.0) for row in rows if row["benchmark"] == label)
            if not vals:
                continue
            y = np.arange(1, len(vals) + 1) / len(vals)
            ax.step(vals, y, where="post", label=policy_label(label), color=COLORS[label], linewidth=1.8)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("ECDF", fontsize=14)
    axes[1].legend(loc="lower right", fontsize=11)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def save_round_gain(rows: list[dict[str, object]], output: Path, horizon: int = 14) -> None:
    x = [0, *[int(row["coordinate"]) for row in rows]]
    y = [float(rows[0]["objective_before"]), *[float(row["objective_after"]) for row in rows]]
    with plt.rc_context({"font.family": "serif", "font.serif": ["cmr10"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True}):
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(x, y, color=COLORS["1,NV"], marker="o", markersize=3, linewidth=1.5)
        for boundary in range(horizon, len(rows), horizon):
            ax.axvline(boundary, color=COLORS["1,SO"], linestyle="--", linewidth=1)
        ax.set_xlabel("Coordinate Update", fontsize=11)
        ax.set_ylabel("Average Total Cost", fontsize=11)
        ax.set_xticks(range(0, len(rows) + 1, max(1, horizon // 2)))
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.2)
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=400)
        plt.close(fig)
