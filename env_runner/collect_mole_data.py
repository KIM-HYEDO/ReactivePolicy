#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import numpy as np
import tqdm
import zarr
from numcodecs import Blosc

from env.mole.mole_image_env import WhackAMoleV1ImageEnv as MoleImageEnv


# ---------------- Policy helpers ----------------
def teleop_action(env, obs) -> np.ndarray:
    if not hasattr(env, "teleop_agent"):
        return np.asarray(obs["agent_pos"], dtype=np.float32)

    agent = env.teleop_agent()
    if agent is None:
        return np.asarray(obs["agent_pos"], dtype=np.float32)

    a = agent.act(obs)
    if a is None:
        return np.asarray(obs["agent_pos"], dtype=np.float32)

    return np.asarray(a, dtype=np.float32)


def quad_bezier_points(p0, p1, bend=0.0, base_n=20) -> np.ndarray:
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)

    d = p1 - p0
    L = float(np.linalg.norm(d) + 1e-9)
    n = np.array([-d[1], d[0]], dtype=np.float32) / L

    alpha = 0.5
    mid = alpha * p0 + (1.0 - alpha) * p1

    c = mid + bend * n

    length = max(2, int(L / 400.0 * base_n))
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t**2 * p1

# ---------------- Zarr utils ----------------
def ensure_zarr_arrays(root, H, W, C, compressor, img_chunks=16, lowdim_chunks=4096):
    data = root.require_group("data")
    meta = root.require_group("meta")

    def req(name, shape_tail, dtype, chunks):
        return data.require_dataset(
            name,
            shape=(0, *shape_tail),
            chunks=(chunks, *shape_tail),
            dtype=dtype,
            compressor=compressor,
            overwrite=False,
        )

    # (T, H, W, C) float32 in [0, 1]
    req("img", (H, W, C), "f4", chunks=img_chunks)

    req("agent_pos", (2,), "f4", chunks=lowdim_chunks)
    req("action", (2,), "f4", chunks=lowdim_chunks)
    req("mole_pos", (2,), "f4", chunks=lowdim_chunks)
    req("reward", tuple(), "f4", chunks=lowdim_chunks)
    req("terminated", tuple(), "i1", chunks=lowdim_chunks)
    req("truncated", tuple(), "i1", chunks=lowdim_chunks)

    meta.require_dataset(
        "episode_ends",
        shape=(0,),
        chunks=(1024,),
        dtype="i8",
        compressor=compressor,
        overwrite=False,
    )
    meta.require_dataset(
        "episode_seeds",
        shape=(0,),
        chunks=(1024,),
        dtype="i8",
        compressor=compressor,
        overwrite=False,
    )
    return data, meta

def append_1d(arr, values):
    values = np.asarray(values)
    old = arr.shape[0]
    arr.resize(old + values.shape[0])
    arr[old : old + values.shape[0]] = values

def append_block(arr, block):
    block = np.asarray(block)
    old = arr.shape[0]
    arr.resize(old + block.shape[0], *arr.shape[1:])
    arr[old : old + block.shape[0]] = block

# ---------------- Image utils ----------------
def to_hwc_f32(img) -> np.ndarray:
    x = np.asarray(img)
    if x.ndim != 3:
        raise ValueError(f"image must be 3D, got {x.shape}")

    # CHW -> HWC if needed
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.moveaxis(x, 0, -1)

    if x.shape[-1] != 3:
        raise ValueError(f"expected last dim=3, got {x.shape}")

    if x.dtype == np.uint8:
        x = x.astype(np.float32)
    else:
        x = x.astype(np.float32)

    return x

def get_obs_image(obs) -> np.ndarray:
    return to_hwc_f32(obs["image"])

# ---------------- Main ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/mole")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--max_steps", type=int, default=3*10)
    ap.add_argument("--render_size", type=int, default=96)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--policy", type=str, default="oracle", choices=["oracle", "teleop", "random"])
    ap.add_argument("--alpha", type=float, default=0.5)  # kept for compat
    ap.add_argument("--human", type=bool, default=True)
    ap.add_argument("--save_video", action="store_true")  # kept for compat (unused)
    return ap.parse_args()


def make_run_dir(out_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def main():
    args = parse_args()

    run_dir = make_run_dir(args.out_dir)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    zarr_path = os.path.join(run_dir, "mole.zarr")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    root = zarr.open_group(zarr_path, mode="w")

    MoleImageEnv.metadata["video.frames_per_second"] = int(args.fps)
    env = MoleImageEnv(render_size=args.render_size)
    env.seed(args.seed)

    obs, info = env.reset()
    img0 = get_obs_image(obs)
    H, W, C = img0.shape

    data, meta = ensure_zarr_arrays(root, H=H, W=W, C=C, compressor=compressor)

    cursor = 0
    pbar = tqdm.tqdm(range(args.episodes), desc="Collecting episodes")

    for ep in pbar:
        seed_ep = int(args.seed + ep)
        env.seed(seed_ep)
        obs, info = env.reset()

        start_pos = None
        target_pos = None
        path = None
        path_step = 0

        if args.human:
            env._render_frame("human")

        imgs = []
        agent_pos = []
        actions = []
        mole_pos = []
        rewards = []
        terminateds = []
        truncateds = []

        for _ in range(args.max_steps):
            if args.policy == "oracle":
                if info.get("mole_changed", 0):
                    start_pos = np.asarray(obs["agent_pos"], dtype=np.float32)
                    target_pos = np.asarray(info["mole_pos"], dtype=np.float32)
                    path = quad_bezier_points(start_pos, target_pos)
                    path_step = 0

                if path is None:
                    act = np.asarray(obs["agent_pos"], dtype=np.float32)
                else:
                    act = np.asarray(path[path_step], dtype=np.float32)
                    path_step = min(path_step + 1, len(path) - 1)

            elif args.policy == "teleop":
                act = teleop_action(env, obs)

            else:
                act = env.action_space.sample().astype(np.float32)

            imgs.append(get_obs_image(obs))
            agent_pos.append(np.asarray(obs["agent_pos"], dtype=np.float32))
            actions.append(act)

            mp = info.get("mole_pos", None)
            if mp is None:
                mole_pos.append(np.zeros((2,), dtype=np.float32))
            else:
                mole_pos.append(np.asarray(mp, dtype=np.float32))

            obs, r, terminated, truncated, info = env.step(act)

            rewards.append(float(r))
            terminateds.append(int(bool(terminated)))
            truncateds.append(int(bool(truncated)))

            if args.human:
                env._render_frame("human")

            if terminated or truncated:
                break

        imgs = np.stack(imgs, axis=0).astype(np.float32)
        agent_pos = np.stack(agent_pos, axis=0).astype(np.float32)
        actions = np.stack(actions, axis=0).astype(np.float32)
        mole_pos = np.stack(mole_pos, axis=0).astype(np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        terminateds = np.asarray(terminateds, dtype=np.int8)
        truncateds = np.asarray(truncateds, dtype=np.int8)

        T = int(imgs.shape[0])
        append_block(data["img"], imgs)
        append_block(data["agent_pos"], agent_pos)
        append_block(data["action"], actions)
        append_block(data["mole_pos"], mole_pos)
        append_block(data["reward"], rewards)
        append_block(data["terminated"], terminateds)
        append_block(data["truncated"], truncateds)

        cursor += T
        append_1d(meta["episode_ends"], [cursor])
        append_1d(meta["episode_seeds"], [seed_ep])

    env.close()
    print(f"\nDone. Zarr saved at: {zarr_path}")
    print("Keys:")
    print("  data/img (T,H,W,3) float32 in [0,1]")
    print("  data/agent_pos (T,2) float32")
    print("  data/action (T,2) float32")
    print("  data/mole_pos (T,2) float32 (zeros if inactive/None)")
    print("  meta/episode_ends (E,) int64")


if __name__ == "__main__":
    main()
