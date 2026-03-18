#!/usr/bin/env python3
"""
Visualize WhackAMoleReactiveEnv with human rendering.

Controls
--------
  Mouse click  : set action target (teleop)
  Q / ESC      : quit
  R            : reset episode

Usage
-----
  python3 visualize_reactive_mole.py                  # oracle policy
  python3 visualize_reactive_mole.py --policy teleop  # mouse control
  python3 visualize_reactive_mole.py --visible_duration 10  # harder
"""

import argparse
import numpy as np
import pygame
from env.mole.mole_image_env import WhackAMoleReactiveEnv


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",           type=str,   default="oracle",
                    choices=["oracle", "teleop"])
    ap.add_argument("--visible_duration", type=int,   default=20)
    ap.add_argument("--max_steps",        type=int,   default=300)
    ap.add_argument("--render_size",      type=int,   default=96)
    ap.add_argument("--fps",              type=int,   default=10)
    ap.add_argument("--seed",             type=int,   default=42)
    return ap.parse_args()


def main():
    args = parse_args()

    WhackAMoleReactiveEnv.metadata["video.frames_per_second"] = args.fps
    env = WhackAMoleReactiveEnv(
        visible_duration=args.visible_duration,
        max_steps=args.max_steps,
        render_size=args.render_size,
        render_mode="human",
    )
    env.seed(args.seed)
    obs, info = env.reset()
    env._render_frame("human")

    # oracle path tracker
    path, path_step = None, 0
    mouse_target = np.array(obs["agent_pos"], dtype=np.float32)

    episode = 0
    running = True

    while running:
        # ---- event handling ------------------------------------------ #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    episode += 1
                    env.seed(args.seed + episode)
                    obs, info = env.reset()
                    path, path_step = None, 0
                    env._render_frame("human")
                    continue
            elif event.type == pygame.MOUSEBUTTONDOWN and args.policy == "teleop":
                mx, my = pygame.mouse.get_pos()
                mouse_target = np.array([float(mx), float(my)], dtype=np.float32)

        # ---- action -------------------------------------------------- #
        if args.policy == "oracle":
            if info.get("mole_changed", False) or path is None:
                start  = np.asarray(obs["agent_pos"], dtype=np.float32)
                target = np.asarray(info["mole_pos"], dtype=np.float32)
                n = max(2, int(np.linalg.norm(target - start) / 400.0 * 20))
                t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                path = (1 - t) * start + t * target
                path_step = 0
            action = path[path_step]
            path_step = min(path_step + 1, len(path) - 1)
        else:
            action = mouse_target

        # ---- step ---------------------------------------------------- #
        obs, reward, terminated, truncated, info = env.step(action)
        env._render_frame("human")

        if terminated or truncated:
            n_ev   = len(info["event_log"])
            lats   = [e["reaction_latency"] for e in info["event_log"]
                      if e["reaction_latency"] is not None]
            print(
                f"[ep {episode}] hits={info['hit_count']}  "
                f"misses={info['miss_count']}  "
                f"hit_rate={info['hit_count']/max(1,n_ev):.2f}  "
                f"avg_react_lat={np.mean(lats):.1f}steps"
                if lats else
                f"[ep {episode}] hits={info['hit_count']}  "
                f"misses={info['miss_count']}  no latency data"
            )
            episode += 1
            env.seed(args.seed + episode)
            obs, info = env.reset()
            path, path_step = None, 0
            env._render_frame("human")

    env.close()


if __name__ == "__main__":
    main()
