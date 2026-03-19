"""WhackAMoleV1Env — reactive task with per-mole timeout.

Mole disappears after `visible_duration` steps if not hit (miss).
Tracks both hits and misses.
"""

import math
import pygame
import numpy as np
from env.mole.mole_env import WhackAMoleBaseEnv


class WhackAMoleV1Env(WhackAMoleBaseEnv):
    """Reactive whack-a-mole: hit before timeout or it's a miss."""

    def __init__(self, visible_duration: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.visible_duration = visible_duration
        self._miss_count = 0

    def reset(self, *, seed=None, options=None, **kwargs):
        self._miss_count = 0
        return super().reset(seed=seed, options=options, **kwargs)

    def step(self, action):
        self._step_count += 1
        self._last_action = action
        self.mole_changed = False

        hit, _agent_pos, _agent_vel = self._physics_step(action)

        self._global_step += 1
        self._mole_step += 1
        self._track_movement()

        timeout = (not hit) and (self._mole_step >= self.visible_duration)

        if hit or timeout:
            self._close_event("hit" if hit else "miss")
            if hit:
                self._hit_count += 1
            else:
                self._miss_count += 1
            self._spawn_mole()
            obs = self._get_obs()
            self._begin_event(obs)
        else:
            obs = self._get_obs()

        truncated = self._global_step >= self.max_steps
        return obs, float(hit), False, truncated, self._make_info()

    def _make_info(self) -> dict:
        info = super()._make_info()
        info["miss_count"]      = self._miss_count
        info["steps_remaining"] = max(0, self.visible_duration - self._mole_step)
        return info

    def _draw_countdown_arc(self, canvas, mx, my):
        ratio = 1.0 - self._mole_step / max(1, self.visible_duration)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        r_val = int(255 * (1.0 - ratio))
        g_val = int(255 * ratio)
        arc_color = (r_val, g_val, 0)
        arc_r = int(self.hole_radius) + 6
        arc_rect = pygame.Rect(mx - arc_r, my - arc_r, arc_r * 2, arc_r * 2)
        arc_end = math.pi * 2 * ratio
        if arc_end > 0.01:
            pygame.draw.arc(canvas, arc_color, arc_rect,
                            -math.pi / 2, -math.pi / 2 + arc_end, 4)

    def _draw_hud(self, canvas):
        hit_surf  = self._hud_font.render(f"HIT:{self._hit_count}",  True, (0, 128, 0))
        miss_surf = self._hud_font.render(f"MISS:{self._miss_count}", True, (180, 0, 0))
        time_surf = self._hud_font.render(
            f"T:{self._mole_step}/{self.visible_duration}", True, (60, 60, 60)
        )
        canvas.blit(hit_surf,  (8, 8))
        canvas.blit(miss_surf, (8, 28))
        canvas.blit(time_surf, (8, 48))
