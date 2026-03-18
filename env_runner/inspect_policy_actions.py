#!/usr/bin/env python3
"""
Single-episode policy action inspection.
Prints action vs mole_pos vs agent_pos at every step.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig
from env.mole.mole_image_env import WhackAMoleReactiveEnv

POLICY_PATH   = "outputs/diffusion_policy/checkpoints/last"
DATASET_ROOT  = "data/lerobot/reactive_mole_test"
REPO_ID       = "reactive_mole"
SEED          = 9999
MAX_STEPS     = 100
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def obs_to_batch(obs, info, device):
    img   = np.clip(obs["image"], 0, 255).astype(np.float32) / 255.0
    img_t = torch.from_numpy(img).unsqueeze(0)
    state_t = torch.from_numpy(np.asarray(obs["agent_pos"], dtype=np.float32)).unsqueeze(0)
    return {
        "observation.image": img_t.to(device),
        "observation.state": state_t.to(device),
    }


def main():
    pretrained = Path(POLICY_PATH) / "pretrained_model"

    ds_meta    = LeRobotDatasetMetadata(repo_id=REPO_ID, root=DATASET_ROOT)
    policy_cfg = PreTrainedConfig.from_pretrained(str(pretrained))
    policy_cfg.pretrained_path = pretrained
    policy     = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    policy.eval().to(DEVICE)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg, pretrained_path=str(pretrained)
    )

    env = WhackAMoleReactiveEnv(visible_duration=20, max_steps=MAX_STEPS, render_size=96)
    env.seed(SEED)
    obs, info = env.reset()
    policy.reset()

    print(f"{'step':>5} | {'agent_x':>8} {'agent_y':>8} | "
          f"{'mole_x':>8} {'mole_y':>8} | "
          f"{'act_x':>8} {'act_y':>8} | "
          f"{'dist_to_mole':>12} | outcome")
    print("-" * 90)

    for t in range(MAX_STEPS):
        agent = np.asarray(obs["agent_pos"], dtype=np.float32)
        mole  = np.asarray(info["mole_pos"],  dtype=np.float32)

        batch  = obs_to_batch(obs, info, DEVICE)
        batch  = preprocessor(batch)
        with torch.no_grad():
            action_t = policy.select_action(batch)
            action_t = postprocessor(action_t)
        action = action_t.to("cpu").numpy()
        if action.ndim == 2:
            action = action[0]
        action = action.astype(np.float32)

        dist = float(np.linalg.norm(mole - agent))
        mole_changed = info.get("mole_changed", False)
        marker = " <-- NEW MOLE" if mole_changed else ""

        print(f"{t:5d} | {agent[0]:8.1f} {agent[1]:8.1f} | "
              f"{mole[0]:8.1f} {mole[1]:8.1f} | "
              f"{action[0]:8.1f} {action[1]:8.1f} | "
              f"{dist:12.1f}{marker}")

        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print(f"\nhits={info['hit_count']}  misses={info['miss_count']}")


if __name__ == "__main__":
    main()
