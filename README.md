# ReactivePolicy

Reactive imitation learning benchmark using a **Whack-a-Mole** task.
A mole appears at a random position and the agent must reach it before it disappears.
The benchmark measures how quickly and accurately a policy reacts to sudden target changes.

## Repository Structure

```
env/
  mole/
    mole_env.py           — WhackAMoleEnv (state obs only)
    mole_image_env.py     — WhackAMoleImageEnv / WhackAMoleReactiveEnv (image + state)
    lerobot_config.py     — lerobot CLI integration (MoleEnvConfig, gym wrapper)
    metrics.py            — success_rate, reaction_latency, hit_error, jerk
    event_logger.py       — JSONL event logger
    dummy_policy.py       — DummyPolicy (oracle, moves toward mole)

env_runner/
  collect_mole_data.py          — collect demos (simple bezier oracle)
  collect_reactive_mole_data.py — collect demos (reactive oracle, zarr output)
  mole_runner.py                — benchmark runner with delay sweep
  eval_reactive_mole.py         — evaluate a trained policy on the env

scripts/
  zarr_to_lerobot.py    — convert zarr demo data → LeRobot v3.0 dataset
  train_diffusion.py    — standalone diffusion policy training (argparse)
  train_mole.py         — lerobot-train CLI wrapper (recommended)

data/
  reactive_mole/        — raw zarr demo data
  lerobot/reactive_mole — converted LeRobot dataset

outputs/
  train/diffusion_mole/ — training checkpoints and eval results

lerobot/                — git submodule (huggingface/lerobot v0.5.0)
```

---

## Setup

### 1. Create conda environment

```bash
conda create -n pusht python=3.12
conda activate pusht
```

### 2. Install lerobot

```bash
cd lerobot
pip install -e ".[smolvla]"
cd ..
```

### 3. Install this package

```bash
pip install -e .
```

### 4. Verify

```bash
conda run -n pusht python -c "import env.mole; import lerobot; print('OK')"
```

---

## Pipeline Overview

```
1. Collect demos  →  2. Convert to LeRobot format  →  3. Train  →  4. Evaluate
```

---

## Step 1 — Collect Demonstration Data

Runs the reactive oracle policy and saves trajectories as a zarr archive.

```bash
conda run -n pusht python env_runner/collect_reactive_mole_data.py \
    --n_episodes 1000 \
    --max_steps 200 \
    --visible_duration 20 \
    --out_dir data/reactive_mole
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--n_episodes` | 1000 | Number of episodes to collect |
| `--max_steps` | 200 | Max steps per episode |
| `--visible_duration` | 20 | Steps mole stays visible before miss |
| `--out_dir` | `data/reactive_mole` | Output directory |

Output layout:
```
data/reactive_mole/<timestamp>/reactive_mole.zarr
  data/
    img           (T, H, W, 3)  float32  [0-255]
    agent_pos     (T, 2)         float32
    action        (T, 2)         float32
    mole_pos      (T, 2)         float32
    reward        (T,)           float32
  meta/
    episode_ends  (E,)           int64
    episode_seeds (E,)           int64
```

---

## Step 2 — Convert to LeRobot Dataset Format

Converts the zarr archive to the LeRobot v3.0 format required for training.

```bash
conda run -n pusht python scripts/zarr_to_lerobot.py \
    --zarr_path data/reactive_mole/<timestamp>/reactive_mole.zarr \
    --out_dir   data/lerobot/reactive_mole \
    --fps 10
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--zarr_path` | (required) | Path to the zarr archive from Step 1 |
| `--out_dir` | (required) | Output path for the LeRobot dataset |
| `--fps` | 10 | Frames per second of the dataset |
| `--repo_id` | `reactive_mole` | Dataset ID used during training |

The resulting dataset contains:
- `observation.image` — (96, 96, 3) video frames
- `observation.state` — (2,) agent position `[x, y]`
- `action` — (2,) target position `[x, y]`

---

## Step 3 — Train

### Option A: lerobot-train CLI (recommended)

Uses `lerobot-train` with the mole env registered for eval during training.

```bash
conda run -n pusht python scripts/train_mole.py \
    --dataset.repo_id=reactive_mole \
    --dataset.root=data/lerobot/reactive_mole \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --env.type=mole \
    --output_dir=outputs/train/diffusion_mole \
    --steps=100000 \
    --batch_size=64
```

Training-only (no env eval):

```bash
conda run -n pusht python scripts/train_mole.py \
    --dataset.repo_id=reactive_mole \
    --dataset.root=data/lerobot/reactive_mole \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/diffusion_mole \
    --steps=100000 \
    --batch_size=64 \
    --eval_freq=0
```

Resume from checkpoint:

```bash
conda run -n pusht python scripts/train_mole.py \
    --config_path=outputs/train/diffusion_mole/checkpoints/020000/pretrained_model/train_config.json \
    --resume=true
```

Key training arguments:

| Argument | Default | Description |
|---|---|---|
| `--policy.type` | — | Policy type: `diffusion`, `act`, etc. |
| `--steps` | 100000 | Total training steps |
| `--batch_size` | 64 | Batch size |
| `--eval_freq` | 20000 | Eval every N steps (0 = disable) |
| `--save_freq` | 20000 | Save checkpoint every N steps |
| `--log_freq` | 200 | Log loss every N steps |

Checkpoints are saved to `outputs/train/diffusion_mole/checkpoints/`.

### Option B: standalone script

Simpler alternative without lerobot's eval loop.

```bash
conda run -n pusht python scripts/train_diffusion.py \
    --dataset_root data/lerobot/reactive_mole \
    --repo_id reactive_mole \
    --output_dir outputs/train/diffusion_mole \
    --steps 100000 \
    --batch_size 64
```

---

## Step 4 — Evaluate

Runs the trained policy on the WhackAMole env and reports metrics.

```bash
conda run -n pusht python env_runner/eval_reactive_mole.py \
    --policy_path outputs/train/diffusion_mole/checkpoints/last \
    --dataset_root data/lerobot/reactive_mole \
    --episodes 100 \
    --seed 9999
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--policy_path` | (required) | Path to checkpoint directory |
| `--dataset_root` | (required) | Path to LeRobot dataset (for normalization stats) |
| `--episodes` | 100 | Number of evaluation episodes |
| `--seed` | 9999 | Random seed |

Reported metrics:
- `success_rate` — fraction of moles hit before timeout
- `reaction_latency` — steps from mole appearance to agent movement onset
- `hit_error` — distance between agent and mole at hit time
- `jerk` — smoothness of agent trajectory

---

## Benchmark Runner (delay sweep)

Evaluates the oracle DummyPolicy across different action delays.

```bash
conda run -n pusht python env_runner/mole_runner.py \
    --seeds 0 1 2 3 4 \
    --delay_ms 0 50 100 150 \
    --max_steps 200 \
    --log_dir logs/mole \
    --plot_dir plots/mole
```

Outputs JSONL logs and PNG plots to `logs/mole/` and `plots/mole/`.
