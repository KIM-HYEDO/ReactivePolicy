#!/usr/bin/env python3
"""Standalone rollout runner for the Whack-a-Mole benchmark.

Runs a delay sweep (0 / 50 / 100 / 150 ms) with a DummyPolicy and writes
per-step JSONL logs plus summary JSON and PNG plots.

Usage
-----
# Full sweep (seeds 0-4, all four delays)
python env_runner/mole_runner.py

# Single run
python env_runner/mole_runner.py --seeds 0 --delay_ms 0

# Custom
python env_runner/mole_runner.py \\
    --seeds 0 1 2 3 4 \\
    --max_steps 500 \\
    --delay_ms 0 50 100 150 \\
    --log_dir logs/mole \\
    --plot_dir plots/mole
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.mole.mole_env import WhackAMoleReactiveEnv as WhackAMoleEnv
from env.mole.dummy_policy import DummyPolicy
from env.mole.event_logger import EventLogger
import env.mole.metrics as mole_metrics
import env.mole.plot_utils as plot_utils

DELAY_SWEEP_MS = [0, 50, 100, 150]


def run_episode(
    env: WhackAMoleEnv,
    policy: DummyPolicy,
    delay_ms: int = 0,
    seed: int = 0,
    max_steps: int = 500,
    log_dir: str = "logs/mole",
) -> str:
    """Run one episode and write a JSONL log. Returns the log file path."""
    env.seed(seed)
    obs = env.reset()

    episode_id = f"seed{seed}_delay{delay_ms}ms"

    with EventLogger(log_dir=log_dir, episode_id=episode_id) as logger:
        t_start = time.perf_counter()

        prev_target_visible = obs[2] >= 0.0
        current_spawn_time = None

        for step in range(max_steps):
            obs_capture_time = time.perf_counter() - t_start

            target_visible_in_obs = obs[2] >= 0.0
            if target_visible_in_obs and not prev_target_visible:
                current_spawn_time = obs_capture_time
            elif not target_visible_in_obs:
                current_spawn_time = None
            prev_target_visible = target_visible_in_obs

            inference_start_time = time.perf_counter() - t_start
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
            action = policy.select_action(obs)
            inference_end_time = time.perf_counter() - t_start

            action_apply_time = time.perf_counter() - t_start
            obs, reward, done, info = env.step(action)

            record = {
                "step": step,
                "seed": seed,
                "obs_capture_time": round(obs_capture_time, 6),
                "inference_start_time": round(inference_start_time, 6),
                "inference_end_time": round(inference_end_time, 6),
                "action_apply_time": round(action_apply_time, 6),
                "target_spawn_time": round(current_spawn_time, 6)
                    if current_spawn_time is not None else None,
                "hit_time": round(action_apply_time, 6) if info["hit"] else None,
                "ee_pos": info["ee_pos"].tolist(),
                "target_pos": info["target_pos"].tolist()
                    if info["target_pos"] is not None else None,
                "action": action.tolist(),
                "hit": bool(info["hit"]),
                "target_visible": bool(info["target_visible"]),
                "target_spawn_step": info["target_spawn_step"],
                "reward": float(reward),
            }
            logger.log_step(record)

            if done:
                break

    return logger.log_path


def run_delay_sweep(
    seeds: list = None,
    max_steps: int = 500,
    log_dir: str = "logs/mole",
    plot_dir: str = "plots/mole",
    delay_sweep: list = None,
):
    if seeds is None:
        seeds = list(range(5))
    if delay_sweep is None:
        delay_sweep = DELAY_SWEEP_MS

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    env = WhackAMoleEnv()
    all_summaries = []

    for delay_ms in delay_sweep:
        policy = DummyPolicy(ee_speed=env.ee_speed)
        delay_records = []

        for seed in tqdm.tqdm(seeds, desc=f"delay={delay_ms:>4}ms"):
            log_path = run_episode(
                env, policy,
                delay_ms=delay_ms,
                seed=seed,
                max_steps=max_steps,
                log_dir=log_dir,
            )
            delay_records.extend(mole_metrics.load_log(log_path))

        summary = mole_metrics.summarize(delay_records, delay_ms=delay_ms)
        all_summaries.append(summary)
        print(
            f"  delay={delay_ms}ms | "
            f"success={summary['success_rate']:.2f} | "
            f"latency={summary['reaction_latency_s']:.4f}s | "
            f"hit_err={summary['hit_error']:.4f} | "
            f"jerk={summary['jerk']:.4f} | "
            f"hits={summary['n_hits']}/{summary['n_targets']}"
        )

    env.close()

    summary_path = os.path.join(log_dir, "delay_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary  → {summary_path}")

    sweep_plot = os.path.join(plot_dir, "delay_sweep.png")
    plot_utils.plot_delay_sweep(all_summaries, out_path=sweep_plot)
    plt.close("all")
    print(f"Sweep    → {sweep_plot}")

    for delay_ms in delay_sweep:
        seed = seeds[0]
        lp = os.path.join(log_dir, f"episode_seed{seed}_delay{delay_ms}ms.jsonl")
        if not os.path.exists(lp):
            continue
        records = mole_metrics.load_log(lp)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_utils.plot_trajectory(
            records, ax=axes[0],
            title=f"Trajectory  seed={seed}  delay={delay_ms}ms"
        )
        plot_utils.plot_action_norms(
            records, ax=axes[1],
            title=f"Action norms  seed={seed}  delay={delay_ms}ms"
        )
        plt.tight_layout()
        traj_path = os.path.join(plot_dir, f"trajectory_seed{seed}_delay{delay_ms}ms.png")
        fig.savefig(traj_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"Traj     → {traj_path}")

    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Whack-a-Mole benchmark runner")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--log_dir", type=str, default="logs/mole")
    parser.add_argument("--plot_dir", type=str, default="plots/mole")
    parser.add_argument("--delay_ms", type=int, nargs="+", default=None)
    args = parser.parse_args()

    run_delay_sweep(
        seeds=args.seeds,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        plot_dir=args.plot_dir,
        delay_sweep=args.delay_ms if args.delay_ms is not None else DELAY_SWEEP_MS,
    )


if __name__ == "__main__":
    main()
