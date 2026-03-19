"""WhackAMoleV2Env — free task, no per-mole timeout.

Mole stays until hit. New mole spawns immediately on hit.
Metric: total hits within the episode (max_steps).
"""

import pygame
from env.mole.mole_env import WhackAMoleBaseEnv


class WhackAMoleV2Env(WhackAMoleBaseEnv):
    """Free whack-a-mole: no timeout, count as many hits as possible."""

    def step(self, action):
        self._step_count += 1
        self._last_action = action
        self.mole_changed = False

        hit, _agent_pos, _agent_vel = self._physics_step(action)

        self._global_step += 1
        self._mole_step += 1
        self._track_movement()

        if hit:
            self._close_event("hit")
            self._hit_count += 1
            self._spawn_mole()
            obs = self._get_obs()
            self._begin_event(obs)
        else:
            obs = self._get_obs()

        truncated = self._global_step >= self.max_steps
        return obs, float(hit), False, truncated, self._make_info()

    def _draw_hud(self, canvas):
        hit_surf  = self._hud_font.render(f"HIT:{self._hit_count}",   True, (0, 128, 0))
        step_surf = self._hud_font.render(f"T:{self._global_step}/{self.max_steps}", True, (60, 60, 60))
        canvas.blit(hit_surf,  (8, 8))
        canvas.blit(step_surf, (8, 28))
