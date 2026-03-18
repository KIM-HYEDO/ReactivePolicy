# Task Specification: Whack-a-Mole Benchmark

## Overview

A minimal 2-D end-effector benchmark for measuring **reactive policy latency** under
artificial inference delays.  Week-1 goal: make the problem *measurable*, not
high-performing.

---

## Task Description

| Property | Value |
|---|---|
| State space | 2-D workspace `[0, 1] × [0, 1]` |
| End-effector | Point mass; action = `(dx, dy)` |
| Target | Single circle; random spawn; visible for limited steps |
| Hit condition | `‖ee − target‖ < hit_radius` |
| Episode length | `max_steps` (runner-controlled) |

### Target lifecycle

```
GAP (gap_steps) → VISIBLE (visible_steps) → [hit or timeout] → GAP → …
```

- During **GAP** the target is hidden; obs reports `target = (−1, −1)`.
- During **VISIBLE** the target position is reported in the observation.
- A **hit** ends the visible window immediately.

---

## Observation & Action

```
obs  = [ee_x, ee_y, target_x, target_y]   # target = (−1, −1) when hidden
action = [dx, dy]                           # clipped to action_space bounds
```

### Observation space

| Field | Range | Dtype |
|---|---|---|
| `ee_x`, `ee_y` | `[0, workspace_size]` | float64 |
| `target_x`, `target_y` | `[−1, workspace_size]` | float64 |

### Action space

| Field | Range |
|---|---|
| `dx`, `dy` | `[−ee_speed, ee_speed]` |

Default `ee_speed = 0.05`, `workspace_size = 1.0`.

---

## Default Parameters

```python
workspace_size  = 1.0    # normalised workspace
hit_radius      = 0.05   # hit if ‖ee − target‖ < hit_radius
target_radius   = 0.05   # visual radius (rendering only)
visible_steps   = 20     # steps target stays visible
gap_steps       = 5      # steps between target disappear and next spawn
ee_speed        = 0.05   # max displacement per step per axis
render_size     = 256    # pixels
```

---

## Per-step Log Schema (JSONL)

Each step appends one JSON object to `logs/mole/episode_<id>.jsonl`.

| Field | Type | Description |
|---|---|---|
| `step` | int | Step index within episode |
| `obs_capture_time` | float | Seconds from episode start when obs was read |
| `inference_start_time` | float | Seconds from episode start before policy call |
| `inference_end_time` | float | Seconds from episode start after policy returns |
| `action_apply_time` | float | Seconds from episode start before `env.step()` |
| `target_spawn_time` | float \| null | `obs_capture_time` when target first appeared in obs |
| `hit_time` | float \| null | `action_apply_time` of the step where hit occurred |
| `ee_pos` | [x, y] | End-effector position |
| `target_pos` | [x, y] \| null | Target position (null when hidden) |
| `action` | [dx, dy] | Action applied this step |
| `hit` | bool | Whether a hit occurred this step |
| `target_visible` | bool | Whether target was visible in returned obs |
| `target_spawn_step` | int \| null | Env step when current target spawned |
| `reward` | float | Step reward (1.0 on hit, else 0.0) |

---

## Metric Definitions

### Success rate
```
success_rate = hits / targets_spawned
```
A target is "hit" if `‖ee − target‖ < hit_radius` while the target is visible.

### Reaction latency
```
reaction_latency = inference_end_time[first step with ‖action‖ > ε]
                 − target_spawn_time
```
Measured per target spawn, averaged over episode.  `ε = 1e-6`.

### Hit error
```
hit_error = ‖ee_pos − target_pos‖  at the hit step
```
Averaged over all hit events.

### Jerk
```
jerk = mean_t ‖Δ³ action_t‖   (3rd finite difference of action sequence)
```

---

## Artificial Delay Injection

The runner inserts `time.sleep(delay_ms / 1000)` between
`inference_start_time` and `inference_end_time` to simulate slow policy
inference.  Sweep: **0 / 50 / 100 / 150 ms**.

---

## Dummy Policy

Moves straight toward the visible target at `ee_speed`.  Outputs zero when
target is hidden.

```python
direction = target_pos − ee_pos
action = direction / ‖direction‖ * min(‖direction‖, ee_speed)
```

---

## File Layout

```
env/mole/
    __init__.py         # gymnasium registration
    mole_env.py         # WhackAMoleEnv(gym.Env)
    event_logger.py     # EventLogger  → JSONL
    metrics.py          # success_rate, reaction_latency, hit_error, jerk
    dummy_policy.py     # DummyPolicy
    plot_utils.py       # trajectory + delay-sweep plots
env_runner/
    mole_runner.py      # standalone rollout + delay sweep
docs/
    task_spec_whack_a_mole.md   # this file
```

---

## Run Commands

```bash
# Quick single run (seed 0, no delay)
python env_runner/mole_runner.py --seeds 0 --delay_ms 0

# Full delay sweep  (seeds 0-4, all four delays)
python env_runner/mole_runner.py

# Custom parameters
python env_runner/mole_runner.py \
    --seeds 0 1 2 3 4 \
    --max_steps 200 \
    --delay_ms 0 50 100 150 \
    --log_dir logs/mole \
    --plot_dir plots/mole
```

## Output Locations

| Artifact | Path |
|---|---|
| Per-episode JSONL logs | `logs/mole/episode_seed<N>_delay<D>ms.jsonl` |
| Delay sweep summary | `logs/mole/delay_sweep_summary.json` |
| Delay sweep plot | `plots/mole/delay_sweep.png` |
| Per-episode trajectories | `plots/mole/trajectory_seed<N>_delay<D>ms.png` |
