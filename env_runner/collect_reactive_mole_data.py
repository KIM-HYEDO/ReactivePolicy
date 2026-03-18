#!/usr/bin/env python3
"""
Collect demonstration data from WhackAMoleReactiveEnv.

Zarr layout
-----------
data/
  img          (T, H, W, 3)  float32
  agent_pos    (T, 2)         float32
  action       (T, 2)         float32
  mole_pos     (T, 2)         float32
  reward       (T,)           float32
  truncated    (T,)           int8
  mole_step    (T,)           int32   steps since current mole appeared
  steps_remaining (T,)        int32   visible_duration - mole_step
meta/
  episode_ends  (E,)  int64
  episode_seeds (E,)  int64

event_log.jsonl  (one JSON object per hit/miss event, all episodes)
config.json
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc

from env.mole.mole_image_env import WhackAMoleReactiveEnv


# ------------------------------------------------------------------ #
# Oracle policy: quadratic bezier toward mole, replans on mole_changed
# ------------------------------------------------------------------ #

def quad_bezier_points(p0, p1, base_n: int = 20) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    L = float(np.linalg.norm(p1 - p0) + 1e-9)
    n = max(2, int(L / 400.0 * base_n))
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return (1 - t) * p0 + t * p1   # straight line (no bend needed)


class OraclePolicy:
    def __init__(self):
        self.path = None
        self.path_step = 0

    def act(self, obs, info) -> np.ndarray:
        if info.get("mole_changed", False) or self.path is None:
            start = np.asarray(obs["agent_pos"], dtype=np.float32)
            target = np.asarray(info["mole_pos"], dtype=np.float32)
            self.path = quad_bezier_points(start, target)
            self.path_step = 0

        action = self.path[self.path_step]
        self.path_step = min(self.path_step + 1, len(self.path) - 1)
        return action


# ------------------------------------------------------------------ #
# Zarr helpers
# ------------------------------------------------------------------ #

def ensure_zarr_arrays(root, H, W, C, compressor):
    data = root.require_group("data")
    meta = root.require_group("meta")

    def req(name, shape_tail, dtype, chunks=4096):
        return data.require_dataset(
            name,
            shape=(0, *shape_tail),
            chunks=(chunks, *shape_tail),
            dtype=dtype,
            compressor=compressor,
            overwrite=False,
        )

    req("img",             (H, W, C), "f4", chunks=16)
    req("agent_pos",       (2,),      "f4")
    req("action",          (2,),      "f4")
    req("mole_pos",        (2,),      "f4")
    req("reward",          tuple(),   "f4")
    req("truncated",       tuple(),   "i1")
    req("mole_step",       tuple(),   "i4")
    req("steps_remaining", tuple(),   "i4")

    for name, dtype in [("episode_ends", "i8"), ("episode_seeds", "i8")]:
        meta.require_dataset(
            name, shape=(0,), chunks=(1024,),
            dtype=dtype, compressor=compressor, overwrite=False,
        )
    return data, meta


def append_block(arr, block):
    block = np.asarray(block)
    old = arr.shape[0]
    arr.resize(old + block.shape[0], *arr.shape[1:])
    arr[old:] = block


def append_1d(arr, values):
    values = np.asarray(values)
    old = arr.shape[0]
    arr.resize(old + values.shape[0])
    arr[old:] = values


def to_hwc_f32(img) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.moveaxis(x, 0, -1)
    return x


# ------------------------------------------------------------------ #
# Argument parsing
# ------------------------------------------------------------------ #

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",           type=str,   default="data/reactive_mole")
    ap.add_argument("--episodes",          type=int,   default=1000)
    ap.add_argument("--max_steps",         type=int,   default=200)
    ap.add_argument("--visible_duration",  type=int,   default=30,
                    help="steps mole is visible before miss")
    ap.add_argument("--movement_threshold",type=float, default=3.0,
                    help="px/step to detect movement onset")
    ap.add_argument("--render_size",       type=int,   default=96)
    ap.add_argument("--fps",               type=int,   default=10)
    ap.add_argument("--seed",              type=int,   default=100000)
    ap.add_argument("--policy",            type=str,   default="oracle",
                    choices=["oracle", "random"])
    ap.add_argument("--human",             type=bool,  default=True)
    return ap.parse_args()


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    args = parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    zarr_path = os.path.join(run_dir, "reactive_mole.zarr")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(zarr_path, mode="w")

    WhackAMoleReactiveEnv.metadata["video.frames_per_second"] = args.fps
    env = WhackAMoleReactiveEnv(
        visible_duration=args.visible_duration,
        movement_threshold=args.movement_threshold,
        max_steps=args.max_steps,
        render_size=args.render_size,
    )

    # initialise zarr with first obs shape
    obs, info = env.reset(seed=args.seed)
    img0 = to_hwc_f32(obs["image"])
    H, W, C = img0.shape
    data, meta = ensure_zarr_arrays(root, H=H, W=W, C=C, compressor=compressor)

    # cache array references to avoid group key-lookup issues across many episodes
    arr_img             = root["data/img"]
    arr_agent_pos       = root["data/agent_pos"]
    arr_action          = root["data/action"]
    arr_mole_pos        = root["data/mole_pos"]
    arr_reward          = root["data/reward"]
    arr_truncated       = root["data/truncated"]
    arr_mole_step       = root["data/mole_step"]
    arr_steps_remaining = root["data/steps_remaining"]
    arr_ep_ends         = root["meta/episode_ends"]
    arr_ep_seeds        = root["meta/episode_seeds"]

    event_log_path = os.path.join(run_dir, "event_log.jsonl")
    policy = OraclePolicy()
    cursor = 0

    pbar = tqdm.tqdm(range(args.episodes), desc="Collecting")
    for ep in pbar:
        seed_ep = args.seed + ep
        env.seed(seed_ep)
        obs, info = env.reset()
        policy = OraclePolicy()
        if args.human:
            env._render_frame("human")

        imgs, agent_pos_buf, actions = [], [], []
        mole_pos_buf, rewards, truncateds = [], [], []
        mole_step_buf, steps_remaining_buf = [], []

        for _ in range(args.max_steps):
            if args.policy == "oracle":
                act = policy.act(obs, info)
            else:
                act = env.action_space.sample().astype(np.float32)

            imgs.append(to_hwc_f32(obs["image"]))
            agent_pos_buf.append(np.asarray(obs["agent_pos"], dtype=np.float32))
            actions.append(act)
            mole_pos_buf.append(np.asarray(info["mole_pos"], dtype=np.float32))
            mole_step_buf.append(int(info["mole_step"]))
            steps_remaining_buf.append(int(info["steps_remaining"]))

            obs, r, terminated, truncated, info = env.step(act)

            rewards.append(float(r))
            truncateds.append(int(bool(truncated)))

            if args.human:
                env._render_frame("human")

            if terminated or truncated:
                break

        # write step-level arrays
        T = len(imgs)
        append_block(arr_img,             np.stack(imgs))
        append_block(arr_agent_pos,        np.stack(agent_pos_buf))
        append_block(arr_action,           np.stack(actions))
        append_block(arr_mole_pos,         np.stack(mole_pos_buf))
        append_block(arr_reward,           np.asarray(rewards,           dtype=np.float32))
        append_block(arr_truncated,        np.asarray(truncateds,        dtype=np.int8))
        append_block(arr_mole_step,        np.asarray(mole_step_buf,     dtype=np.int32))
        append_block(arr_steps_remaining,  np.asarray(steps_remaining_buf, dtype=np.int32))

        cursor += T
        append_1d(arr_ep_ends,  [cursor])
        append_1d(arr_ep_seeds, [seed_ep])

        # write event log (JSONL, one line per hit/miss event)
        with open(event_log_path, "a") as f:
            for ev in info["event_log"]:
                ev["episode"] = ep
                ev["seed"] = seed_ep
                f.write(json.dumps(ev) + "\n")

        hit_rate = info["hit_count"] / max(1, info["hit_count"] + info["miss_count"])
        pbar.set_postfix({
            "hits": info["hit_count"],
            "miss": info["miss_count"],
            "hit_rate": f"{hit_rate:.2f}",
        })

    env.close()

    # summary stats from event log
    with open(event_log_path) as f:
        events = [json.loads(l) for l in f]

    hits   = [e for e in events if e["outcome"] == "hit"]
    misses = [e for e in events if e["outcome"] == "miss"]
    lats   = [e["reaction_latency"] for e in events if e["reaction_latency"] is not None]

    print(f"\nDone. {len(events)} events across {args.episodes} episodes.")
    print(f"  hit_rate        : {len(hits)/max(1,len(events)):.3f}")
    print(f"  avg hit_time    : {np.mean([e['hit_time'] for e in hits]):.1f} steps" if hits else "  no hits")
    print(f"  avg react_lat   : {np.mean(lats):.1f} steps" if lats else "  no latency data")
    print(f"  Zarr            : {zarr_path}")
    print(f"  event_log       : {event_log_path}")


if __name__ == "__main__":
    main()
