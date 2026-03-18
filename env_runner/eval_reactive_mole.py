#!/usr/bin/env python3
"""
Evaluate a trained LeRobot policy on WhackAMoleReactiveEnv.

Usage:
    python3 eval_reactive_mole.py \
        --policy_path /home/dyros/ReactivePolicy/outputs/diffusion_policy/checkpoints/last \
        --dataset_root /home/dyros/ReactivePolicy/data/lerobot/reactive_mole_test \
        --episodes 100 \
        --seed 9999
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ── LeRobot imports ────────────────────────────────────────────────────── #
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

# ── Local env ─────────────────────────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.mole.mole_image_env import WhackAMoleReactiveEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path",   required=True,  type=str,
                    help="Path to checkpoint dir (contains pretrained_model/)")
    ap.add_argument("--dataset_root",  required=True,  type=str,
                    help="Root of the LeRobot dataset used during training")
    ap.add_argument("--dataset_repo_id", default="reactive_mole", type=str)
    ap.add_argument("--episodes",      default=100,    type=int)
    ap.add_argument("--max_steps",     default=200,    type=int)
    ap.add_argument("--visible_duration", default=20,  type=int)
    ap.add_argument("--render_size",   default=96,     type=int)
    ap.add_argument("--fps",           default=10,     type=int)
    ap.add_argument("--seed",          default=9999,   type=int)
    ap.add_argument("--render",        action="store_true",
                    help="Render human window during evaluation")
    ap.add_argument("--save_video",    default=None,   type=str,
                    help="Directory to save episode videos (MP4)")
    ap.add_argument("--n_videos",      default=5,      type=int,
                    help="Number of episodes to save as video")
    ap.add_argument("--out_json",      default=None,   type=str,
                    help="Save summary metrics to this JSON file")
    return ap.parse_args()


def obs_to_batch(obs: dict, info: dict, device: str) -> dict[str, torch.Tensor]:
    """Convert a single env obs + info to a policy batch (B=1)."""
    # image: env returns CHW float32 [0-255] → float32 [0,1], add batch dim
    img = np.clip(obs["image"], 0, 255).astype(np.float32) / 255.0   # CHW float32
    img_t = torch.from_numpy(img).unsqueeze(0)                        # (1, C, H, W)

    # state: agent_pos only
    state_t = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0)  # (1, 2)

    return {
        "observation.image": img_t.to(device),
        "observation.state": state_t.to(device),
    }


def main():
    args = parse_args()
    pretrained_model_path = Path(args.policy_path) / "pretrained_model"

    # ── Load dataset metadata (for normalization stats) ────────────────── #
    print("Loading dataset metadata...")
    ds_meta = LeRobotDatasetMetadata(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )

    # ── Load policy config & weights ───────────────────────────────────── #
    print("Loading policy...")
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_model_path))
    policy_cfg.pretrained_path = pretrained_model_path

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(DEVICE)

    # ── Load preprocessor (normalizer) ─────────────────────────────────── #
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_model_path),
    )

    # ── Create environment ─────────────────────────────────────────────── #
    WhackAMoleReactiveEnv.metadata["video.frames_per_second"] = args.fps
    env = WhackAMoleReactiveEnv(
        visible_duration=args.visible_duration,
        max_steps=args.max_steps,
        render_size=args.render_size,
    )

    # ── Video output setup ─────────────────────────────────────────────── #
    if args.save_video:
        video_dir = Path(args.save_video)
        video_dir.mkdir(parents=True, exist_ok=True)

    # ── Eval loop ──────────────────────────────────────────────────────── #
    all_hit_rates   = []
    all_latencies   = []
    all_hit_counts  = []
    all_miss_counts = []

    pbar = tqdm(range(args.episodes), desc="Evaluating")
    for ep in pbar:
        env.seed(args.seed + ep)
        obs, info = env.reset()

        policy.reset()

        # Setup video writer for this episode
        writer = None
        if args.save_video and ep < args.n_videos:
            video_path = str(video_dir / f"ep{ep:04d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(video_path, fourcc, args.fps,
                                     (args.render_size, args.render_size))

        for _ in range(args.max_steps):
            batch = obs_to_batch(obs, info, DEVICE)
            batch = preprocessor(batch)

            with torch.no_grad():
                action_t = policy.select_action(batch)   # normalized (1, 2) or (2,)
                action_t = postprocessor(action_t)        # unnormalize to pixel space

            action = action_t.to("cpu").numpy()
            if action.ndim == 2:
                action = action[0]
            action = action.astype(np.float32)

            obs, _, terminated, truncated, info = env.step(action)

            if args.render:
                env._render_frame("human")

            if writer is not None:
                frame = env._render_frame("rgb_array")  # HWC uint8
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if terminated or truncated:
                break

        if writer is not None:
            writer.release()

        hits  = info["hit_count"]
        miss  = info["miss_count"]
        total = hits + miss
        hr    = hits / max(1, total)
        lats  = [e["reaction_latency"] for e in info["event_log"]
                 if e["reaction_latency"] is not None]

        all_hit_rates.append(hr)
        all_latencies.extend(lats)
        all_hit_counts.append(hits)
        all_miss_counts.append(miss)

        pbar.set_postfix({
            "hit_rate": f"{np.mean(all_hit_rates):.3f}",
            "avg_lat":  f"{np.mean(all_latencies):.1f}" if all_latencies else "N/A",
        })

    env.close()

    # ── Summary ────────────────────────────────────────────────────────── #
    summary = {
        "episodes":       args.episodes,
        "hit_rate_mean":  float(np.mean(all_hit_rates)),
        "hit_rate_std":   float(np.std(all_hit_rates)),
        "total_hits":     int(np.sum(all_hit_counts)),
        "total_misses":   int(np.sum(all_miss_counts)),
        "react_lat_mean": float(np.mean(all_latencies))  if all_latencies else None,
        "react_lat_std":  float(np.std(all_latencies))   if all_latencies else None,
        "react_lat_n":    len(all_latencies),
    }

    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k:20s}: {v}")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved to: {args.out_json}")


if __name__ == "__main__":
    main()
