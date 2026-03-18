"""Train Diffusion Policy on the WhackAMole dataset using lerobot-train CLI.

Imports MoleEnvConfig before lerobot parses CLI args so that
--env.type=mole is recognised by draccus ChoiceRegistry.

Usage:
    python scripts/train_mole.py \
        --dataset.repo_id=reactive_mole \
        --dataset.root=data/lerobot/reactive_mole \
        --policy.type=diffusion \
        --env.type=mole \
        --output_dir=outputs/train/diffusion_mole \
        --steps=100000 \
        --batch_size=64

    # Training only (no env eval):
    python scripts/train_mole.py \
        --dataset.repo_id=reactive_mole \
        --dataset.root=data/lerobot/reactive_mole \
        --policy.type=diffusion \
        --output_dir=outputs/train/diffusion_mole \
        --steps=100000 \
        --batch_size=64 \
        --eval_freq=0
"""

import env.mole.lerobot_config  # noqa: F401 — registers MoleEnvConfig + mole_lerobot gym env

from lerobot.scripts.lerobot_train import main

if __name__ == "__main__":
    main()
