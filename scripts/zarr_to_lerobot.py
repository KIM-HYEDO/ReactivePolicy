"""
Convert reactive mole zarr dataset to LeRobot v3.0 format
using the official LeRobotDataset API.

Usage:
    python scripts/zarr_to_lerobot.py \
        --zarr_path data/reactive_mole/<timestamp>/reactive_mole.zarr \
        --out_dir   data/lerobot/reactive_mole \
        --fps 10
"""

import argparse
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK = "Whack the mole: move the agent to hit the mole before it disappears."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr_path", required=True, type=str)
    ap.add_argument("--out_dir",   required=True, type=str)
    ap.add_argument("--fps",       default=10,    type=int)
    ap.add_argument("--repo_id",   default="reactive_mole", type=str)
    return ap.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Load zarr
    # ------------------------------------------------------------------ #
    print(f"Loading zarr: {args.zarr_path}")
    store = zarr.open(args.zarr_path, mode="r")

    imgs        = store["data"]["img"]           # (T, H, W, 3) float32  [0-255]
    agent_pos   = store["data"]["agent_pos"]     # (T, 2) float32
    actions     = store["data"]["action"]        # (T, 2) float32
    ep_ends     = store["meta"]["episode_ends"][:]  # (E,) int64 cumulative

    H, W = int(imgs.shape[1]), int(imgs.shape[2])
    n_episodes = len(ep_ends)
    ep_starts = np.concatenate([[0], ep_ends[:-1]])

    print(f"  episodes : {n_episodes}")
    print(f"  frames   : {int(imgs.shape[0])}")
    print(f"  image    : {H}x{W}")

    # ------------------------------------------------------------------ #
    # Create LeRobotDataset
    # ------------------------------------------------------------------ #
    features = {
        "observation.image": {
            "dtype": "video",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["agent_x", "agent_y"],
        },
        "action": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["x", "y"],
        },
    }

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.out_dir,
        features=features,
        robot_type="custom",
        use_videos=True,
    )

    # ------------------------------------------------------------------ #
    # Convert episodes
    # ------------------------------------------------------------------ #
    for ep_idx in tqdm(range(n_episodes), desc="Converting"):
        s, e = int(ep_starts[ep_idx]), int(ep_ends[ep_idx])

        ep_imgs      = imgs[s:e]        # (T, H, W, 3) float32
        ep_agent_pos = agent_pos[s:e]   # (T, 2)
        ep_actions   = actions[s:e]     # (T, 2)

        for t in range(e - s):
            img_uint8 = np.clip(ep_imgs[t], 0, 255).astype(np.uint8)  # HWC uint8
            state = ep_agent_pos[t].astype(np.float32)
            action = ep_actions[t].astype(np.float32)

            ds.add_frame({
                "task":                TASK,
                "observation.image":   img_uint8,
                "observation.state":   state,
                "action":              action,
            })

        ds.save_episode()

    # ------------------------------------------------------------------ #
    # Consolidate (compute stats, write meta)
    # ------------------------------------------------------------------ #
    print("Consolidating dataset (computing stats)...")
    ds.finalize()

    print(f"\nDone. Saved to: {Path(args.out_dir).resolve()}")
    print(f"  total_episodes : {ds.num_episodes}")
    print(f"  total_frames   : {ds.num_frames}")


if __name__ == "__main__":
    main()
