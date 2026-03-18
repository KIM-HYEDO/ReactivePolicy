"""Train Diffusion Policy on reactive_mole LeRobot dataset.

Usage:
    python scripts/train_diffusion.py [--steps 100000] [--batch_size 64] [--output_dir outputs/train/diffusion_mole]
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="data/lerobot/reactive_mole")
    ap.add_argument("--repo_id", default="reactive_mole")
    ap.add_argument("--output_dir", default="outputs/train/diffusion_mole")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_freq", type=int, default=200)
    ap.add_argument("--save_freq", type=int, default=20_000)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Dataset features → policy input/output
    # ------------------------------------------------------------------ #
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features  = {k: ft for k, ft in features.items() if k not in output_features}

    print("Input features:", list(input_features.keys()))
    print("Output features:", list(output_features.keys()))

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # 96x96 이미지는 작으므로 resize 불필요
        resize_shape=None,
    )

    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=meta.stats)

    # ------------------------------------------------------------------ #
    # Dataset
    # ------------------------------------------------------------------ #
    fps = meta.fps
    delta_timestamps = {
        "observation.image": [i / fps for i in cfg.observation_delta_indices],
        "observation.state": [i / fps for i in cfg.observation_delta_indices],
        "action":            [i / fps for i in cfg.action_delta_indices],
    }

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        delta_timestamps=delta_timestamps,
    )
    print(f"Dataset: {len(dataset)} frames")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------ #
    # Optimizer
    # ------------------------------------------------------------------ #
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    step = 0
    pbar = tqdm(total=args.steps, desc="Training")

    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            loss, info = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                tqdm.write(f"step={step:6d}  loss={loss.item():.4f}")

            if args.save_freq > 0 and step > 0 and step % args.save_freq == 0:
                ckpt = output_dir / f"checkpoint_{step:07d}"
                policy.save_pretrained(ckpt)
                preprocessor.save_pretrained(ckpt)
                postprocessor.save_pretrained(ckpt)
                tqdm.write(f"  → saved checkpoint: {ckpt}")

            step += 1
            pbar.update(1)
            if step >= args.steps:
                break

    pbar.close()

    # Final save
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(f"\nDone. Model saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
