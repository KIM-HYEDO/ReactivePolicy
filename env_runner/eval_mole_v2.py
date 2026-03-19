#!/usr/bin/env python3
"""
Evaluate a trained LeRobot policy on WhackAMoleFreeImageEnv.

No per-mole timeout — mole stays until hit.
Metric: how many moles the agent hits within max_steps.

Usage:
    python3 eval_free_mole.py \
        --policy_path /home/dyros/ReactivePolicy/outputs/diffusion_policy/checkpoints/last \
        --dataset_root /home/dyros/ReactivePolicy/data/lerobot/reactive_mole \
        --episodes 100 \
        --seed 9999
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.mole.mole_image_env import WhackAMoleV2ImageEnv as WhackAMoleFreeImageEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path",    required=True,  type=str)
    ap.add_argument("--dataset_root",   required=True,  type=str)
    ap.add_argument("--dataset_repo_id", default="reactive_mole", type=str)
    ap.add_argument("--episodes",       default=100,    type=int)
    ap.add_argument("--max_steps",      default=200,    type=int)
    ap.add_argument("--render_size",    default=96,     type=int)
    ap.add_argument("--fps",            default=10,     type=int)
    ap.add_argument("--seed",           default=9999,   type=int)
    ap.add_argument("--render",         action="store_true")
    ap.add_argument("--save_video",     default=None,   type=str)
    ap.add_argument("--n_videos",       default=5,      type=int)
    ap.add_argument("--out_json",       default=None,   type=str)
    ap.add_argument("--delay_steps",    default=0,      type=int)
    ap.add_argument("--measure_delay",  action="store_true")
    return ap.parse_args()


def obs_to_batch(obs: dict, device: str) -> dict[str, torch.Tensor]:
    img = np.clip(obs["image"], 0, 255).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0)
    state_t = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0)
    return {
        "observation.image": img_t.to(device),
        "observation.state": state_t.to(device),
    }


def main():
    args = parse_args()
    pretrained_model_path = Path(args.policy_path) / "pretrained_model"

    print("Loading dataset metadata...")
    ds_meta = LeRobotDatasetMetadata(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )

    print("Loading policy...")
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained_model_path))
    policy_cfg.pretrained_path = pretrained_model_path

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval()
    policy.to(DEVICE)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(pretrained_model_path),
    )

    if args.measure_delay:
        dummy_env = WhackAMoleFreeImageEnv(render_size=args.render_size)
        dummy_obs, _ = dummy_env.reset()
        dummy_batch = preprocessor(obs_to_batch(dummy_obs, DEVICE))
        policy.reset()
        for _ in range(5):
            with torch.no_grad():
                policy.select_action(dummy_batch)
        policy.reset()
        latencies = []
        for _ in range(30):
            policy.reset()
            t0 = time.perf_counter()
            with torch.no_grad():
                policy.select_action(dummy_batch)
            latencies.append(time.perf_counter() - t0)
        dummy_env.close()
        avg_ms = float(np.mean(latencies)) * 1000
        step_ms = 1000.0 / args.fps
        args.delay_steps = max(0, round(avg_ms / step_ms))
        print(f"Inference latency: {avg_ms:.1f} ms  (step={step_ms:.0f} ms)  →  delay_steps = {args.delay_steps}")

    if args.delay_steps > 0:
        print(f"Action delay: {args.delay_steps} step(s) = {args.delay_steps * 1000 / args.fps:.0f} ms")

    WhackAMoleFreeImageEnv.metadata["video.frames_per_second"] = args.fps
    env = WhackAMoleFreeImageEnv(
        max_steps=args.max_steps,
        render_size=args.render_size,
    )

    if args.save_video:
        video_dir = Path(args.save_video)
        video_dir.mkdir(parents=True, exist_ok=True)

    n_action_steps = getattr(policy_cfg, 'n_action_steps', 1)

    all_hit_counts = []
    all_latencies  = []

    pbar = tqdm(range(args.episodes), desc="Evaluating")
    for ep in pbar:
        env.seed(args.seed + ep)
        obs, info = env.reset()
        policy.reset()

        exec_queue    = collections.deque()
        delayed_chunk = None
        last_action   = np.array(obs["agent_pos"], dtype=np.float32)

        writer = None
        if args.save_video and ep < args.n_videos:
            video_path = str(video_dir / f"ep{ep:04d}.mp4")
            writer = imageio.get_writer(video_path, fps=args.fps)

        for _ in range(args.max_steps):
            if not exec_queue:
                if delayed_chunk is not None:
                    chunk_actions, countdown = delayed_chunk
                    if countdown <= 1:
                        exec_queue.extend(chunk_actions)
                        delayed_chunk = None
                    else:
                        delayed_chunk = (chunk_actions, countdown - 1)
                else:
                    batch = obs_to_batch(obs, DEVICE)
                    batch = preprocessor(batch)
                    new_chunk = []
                    for _c in range(n_action_steps):
                        with torch.no_grad():
                            at = policy.select_action(batch)
                            at = postprocessor(at)
                        a = at.cpu().numpy()
                        if a.ndim == 2:
                            a = a[0]
                        new_chunk.append(a.astype(np.float32))
                    if args.delay_steps > 0:
                        delayed_chunk = (new_chunk, args.delay_steps)
                    else:
                        exec_queue.extend(new_chunk)

            if exec_queue:
                apply_action = exec_queue.popleft()
            else:
                apply_action = last_action
            last_action = apply_action

            obs, _, terminated, truncated, info = env.step(apply_action)

            if args.render:
                env._render_frame("human")

            if writer is not None:
                frame = env._render_frame("rgb_array_hud")
                writer.append_data(frame)

            if terminated or truncated:
                break

        if writer is not None:
            writer.close()

        hits = info["hit_count"]
        lats = [e["reaction_latency"] for e in info["event_log"]
                if e["reaction_latency"] is not None]
        all_hit_counts.append(hits)
        all_latencies.extend(lats)

        pbar.set_postfix({
            "hits/ep":  f"{np.mean(all_hit_counts):.2f}",
            "avg_lat":  f"{np.mean(all_latencies):.1f}" if all_latencies else "N/A",
        })

    env.close()

    summary = {
        "episodes":         args.episodes,
        "max_steps":        args.max_steps,
        "hits_mean":        float(np.mean(all_hit_counts)),
        "hits_std":         float(np.std(all_hit_counts)),
        "hits_per_step":    float(np.mean(all_hit_counts)) / args.max_steps,
        "total_hits":       int(np.sum(all_hit_counts)),
        "react_lat_mean":   float(np.mean(all_latencies))  if all_latencies else None,
        "react_lat_std":    float(np.std(all_latencies))   if all_latencies else None,
        "react_lat_n":      len(all_latencies),
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
