"""Plotting utilities for the Whack-a-Mole benchmark."""

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(records: list, ax=None, title: str = "Trajectory"):
    """Plot EE path, target positions, and hit events."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ee_xs = [r["ee_pos"][0] for r in records]
    ee_ys = [r["ee_pos"][1] for r in records]
    ax.plot(ee_xs, ee_ys, color="steelblue", lw=1.0, alpha=0.6, label="EE path")
    ax.plot(ee_xs[0], ee_ys[0], "bs", markersize=7, label="EE start")

    seen_spawns = set()
    for r in records:
        tp = r.get("target_pos")
        sp = r.get("target_spawn_step")
        if tp is not None and tp[0] >= 0 and sp not in seen_spawns:
            seen_spawns.add(sp)
            ax.scatter(tp[0], tp[1], c="green", s=120, marker="o", zorder=5,
                       label="target" if len(seen_spawns) == 1 else "")

    hit_plotted = False
    for r in records:
        if r.get("hit"):
            ax.scatter(r["ee_pos"][0], r["ee_pos"][1], c="red", s=80,
                       marker="x", zorder=6, label="" if hit_plotted else "hit")
            hit_plotted = True

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_action_norms(records: list, ax=None, title: str = "Action norms"):
    """Plot ‖action‖ over time with shaded target-visible windows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    steps = [r["step"] for r in records]
    norms = [np.linalg.norm(r["action"]) if r.get("action") else 0.0 for r in records]
    ax.plot(steps, norms, color="black", lw=0.8, alpha=0.8)

    in_window = False
    for r in records:
        if r.get("target_visible") and not in_window:
            window_start = r["step"]
            in_window = True
        elif not r.get("target_visible") and in_window:
            ax.axvspan(window_start - 0.5, r["step"] - 0.5, alpha=0.12, color="green")
            in_window = False
    if in_window:
        ax.axvspan(window_start - 0.5, steps[-1] + 0.5, alpha=0.12,
                   color="green", label="target visible")

    ax.set_xlabel("Step")
    ax.set_ylabel("‖action‖")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_delay_sweep(summary_list: list, out_path: str = None):
    """Four-panel plot across delay values."""
    delays = [s["delay_ms"] for s in summary_list]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Whack-a-Mole: Delay Sweep", fontsize=13)

    panels = [
        ("success_rate",       "Success Rate",         axes[0, 0]),
        ("reaction_latency_s", "Reaction Latency (s)", axes[0, 1]),
        ("hit_error",          "Hit Error (dist)",     axes[1, 0]),
        ("jerk",               "Jerk (mean ‖Δ³a‖)",   axes[1, 1]),
    ]
    for key, label, ax in panels:
        vals = [s.get(key, float("nan")) for s in summary_list]
        ax.plot(delays, vals, "o-", color="steelblue", linewidth=2, markersize=7)
        ax.set_xlabel("Inference Delay (ms)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(delays)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
    return fig
