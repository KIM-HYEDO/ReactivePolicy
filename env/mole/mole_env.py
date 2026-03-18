import math
import logging
import collections
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pygame

import pymunk
from pymunk.vec2d import Vec2d

from env.mole.pymunk_override import DrawOptions

logger = logging.getLogger(__name__)


class WhackAMoleEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0.0, 1.0)

    def __init__(
        self,
        visible_duration: int = 20,       # steps mole stays before miss
        movement_threshold: float = 3.0,  # px/step to detect movement onset
        max_steps: int = 200,
        render_size: int = 96,
        # inherited physics params
        legacy: bool = False,
        render_action: bool = False,
        render_mode: str = "rgb_array",
        damping: float = 0.0,
    ):
        self._seed = 0
        self.seed()
        self.window_size = ws = 512
        self.render_size = render_size
        self.sim_hz = 100
        self.dt = 1.0 / self.sim_hz
        self.render_action = bool(render_action)
        self.render_mode = render_mode
        # Local controller params
        self.k_p, self.k_v = 100, 20    # PD control gains
        self.control_hz = self.metadata['video.frames_per_second']
        self.legacy = legacy
        self.damping = damping
        # env parameters
        self.agent_radius = 15.0
        self.n_holes = 1
        self.hole_margin = 20.0
        self.hole_radius = 30.0
        self.mole_radius = 30.0
        self.hit_radius = 15.0
        self.min_spawn_dist = 50.0
        self.min_steps = 20
        self.episode_level = 1

        # reactive params
        self.visible_duration = visible_duration
        self.movement_threshold = movement_threshold
        self.max_steps = max_steps

        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
            "mole_pos":  spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float64),
            high=np.array([ws, ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64,
        )

        self.window = None
        self.clock = None
        self.screen = None
        self._canvas = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None

        self.red_mole_active = True
        self.red_mole_idx = 0
        self.mole_changed = True
        self.t = 0.0

        # reactive runtime state (initialised in reset)
        self._global_step = 0
        self._mole_step = 0
        self._hit_count = 0
        self._miss_count = 0
        self._moved = False
        self._reaction_latency = None
        self._pos_at_appear = np.zeros(2, dtype=np.float32)
        self._mole_pos_at_appear = np.zeros(2, dtype=np.float32)
        self.event_log = []

        # initialise font once to avoid per-frame allocation failures
        pygame.init()
        try:
            self._hud_font = pygame.font.SysFont("monospace", 18)
        except Exception:
            logger.warning("Could not load monospace font; HUD text will be disabled.")
            self._hud_font = None

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def seed(self, seed: int | None = None):
        if seed is None:
            seed = int(np.random.randint(0, 65535))
        self._seed = int(seed)
        self.np_random = np.random.default_rng(self._seed)
        return [self._seed]

    def reset(self, *, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        self.episode_level = kwargs.get("episode_level", 1)
        if seed is not None:
            self.seed(seed)

        self._step_count = 0
        self._last_action = None
        self._reward = 0.0
        self._episode_done = False

        self.t = 0.0
        self.red_mole_active = True
        self.red_mole_idx = 0
        self.mole_changed = True

        # Try multiple times to avoid spawning too close to the mole.
        for _ in range(10):
            self._setup_world()
            self._spawn_mole()

            ap = Vec2d(self.agent.position.x, self.agent.position.y)
            mp = Vec2d(*self.holes[0])
            if (mp - ap).length >= self.min_spawn_dist:
                break

        self._global_step = 0
        self._hit_count = 0
        self._miss_count = 0
        self.event_log = []
        obs = self._get_obs()
        self._begin_event(obs)
        return obs, self._make_info()

    def step(self, action):
        self._step_count += 1
        self._last_action = action
        self.mole_changed = False

        hit, _agent_pos, _agent_vel = self._physics_step(action)

        self._global_step += 1
        self._mole_step += 1
        # ---- movement onset ------------------------------------------- #
        cur_pos = np.array(self.agent.position, dtype=np.float32)
        if not self._moved:
            if np.linalg.norm(cur_pos - self._pos_at_appear) > self.movement_threshold:
                self._reaction_latency = self._mole_step
                self._moved = True

        # ---- event outcome -------------------------------------------- #
        timeout = (not hit) and (self._mole_step >= self.visible_duration)

        if hit or timeout:
            self._close_event("hit" if hit else "miss")
            if hit:
                self._hit_count += 1
            else:
                self._miss_count += 1
            self._spawn_mole()              # new random position
            obs = self._get_obs()           # obs with new mole
            self._begin_event(obs)
        else:
            obs = self._get_obs()

        truncated = self._global_step >= self.max_steps
        return obs, float(hit), False, truncated, self._make_info()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
        self.screen = None
        self._canvas = None

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #

    def _physics_step(self, action):
        """Run one control step of physics. Returns (hit, agent_pos, agent_vel)."""
        target = self._parse_action_to_target(action)
        n_steps = max(1, int(round(self.sim_hz / float(self.control_hz))))

        hit = False
        for _ in range(n_steps):
            acc = self.k_p * (target - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
            self.agent.velocity += acc * self.dt

            self.space.step(self.dt)
            self._clamp_agent_inside()
            self.t += self.dt

            if self.red_mole_active and self._is_hit(self.red_mole_idx):
                hit = True
                break

        agent_pos = np.array(self.agent.position, dtype=np.float32)
        agent_vel = np.array(self.agent.velocity, dtype=np.float32)
        return hit, agent_pos, agent_vel

    def _setup_world(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = float(self.damping)

        self.agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        x = float(self.np_random.integers(0, self.window_size))
        y = float(self.np_random.integers(0, self.window_size))
        self.agent.position = (x, y)
        self.agent.velocity = (0, 0)

        self.agent_shape = pymunk.Circle(self.agent, self.agent_radius)
        self.agent_shape.color = pygame.Color("RoyalBlue")
        self.agent_shape.friction = 0.0

        self.space.add(self.agent, self.agent_shape)

        self.holes = self._make_hole()

    def _make_hole(self) -> np.ndarray:
        safe = float(self.hole_margin + self.hole_radius)
        x = float(self.np_random.uniform(safe, self.window_size - safe))
        y = float(self.np_random.uniform(safe, self.window_size - safe))
        return np.array([[x, y]], dtype=np.float32)

    def _parse_action_to_target(self, action) -> Vec2d:
        if action is None:
            return Vec2d(self.agent.position.x, self.agent.position.y)

        ax = float(np.clip(action[0], 5.0, self.window_size - 5.0))
        ay = float(np.clip(action[1], 5.0, self.window_size - 5.0))
        return Vec2d(ax, ay)

    def _clamp_agent_inside(self):
        x = float(np.clip(self.agent.position.x, 5.0, self.window_size - 5.0))
        y = float(np.clip(self.agent.position.y, 5.0, self.window_size - 5.0))
        self.agent.position = (x, y)

    def _spawn_mole(self):
        self._reward = 0.0
        self.holes = self._make_hole()
        self.red_mole_idx = 0
        self.red_mole_active = True
        self.mole_changed = True

    def _is_hit(self, mole_idx: int) -> bool:
        mp = Vec2d(*self.holes[mole_idx])
        return (mp - self.agent.position).length <= self.hit_radius

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def _get_obs(self):
        agent_pos = np.array([float(self.agent.position.x), float(self.agent.position.y)], dtype=np.float32)
        mole_pos = self.holes[0].copy()
        return {"agent_pos": agent_pos, "mole_pos": mole_pos}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _begin_event(self, obs):
        self._mole_step = 0
        self._reaction_latency = None
        self._moved = False
        self._pos_at_appear = np.array(self.agent.position, dtype=np.float32)
        self._mole_pos_at_appear = self.holes[0].copy()

    def _close_event(self, outcome: str):
        self.event_log.append({
            "global_step":         self._global_step,
            "mole_pos":            self._mole_pos_at_appear.tolist(),
            "agent_pos_at_appear": self._pos_at_appear.tolist(),
            "reaction_latency":    self._reaction_latency,
            "hit_time":            self._mole_step if outcome == "hit" else None,
            "outcome":             outcome,
        })

    def _make_info(self) -> dict:
        return {
            "t":                float(self.t),
            "step":             int(self._global_step),
            "agent_pos":        np.array(self.agent.position, dtype=np.float32),
            "mole_pos":         self.holes[0].copy(),
            "mole_step":        self._mole_step,
            "steps_remaining":  max(0, self.visible_duration - self._mole_step),
            "reaction_latency": self._reaction_latency,
            "hit_count":        self._hit_count,
            "miss_count":       self._miss_count,
            "mole_changed":     bool(self.mole_changed),
            "event_log":        self.event_log,
        }

    # ------------------------------------------------------------------ #
    # Rendering — countdown arc + hit flash + HUD
    # ------------------------------------------------------------------ #

    def render(self):
        return self._render_frame(self.render_mode)

    def _render_frame(self, mode: str):
        if mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

        if self._canvas is None:
            self._canvas = pygame.Surface((self.window_size, self.window_size))
        canvas = self._canvas
        canvas.fill((255, 255, 255))
        self.screen = canvas

        mx, my = int(self.holes[0][0]), int(self.holes[0][1])

        # ---- countdown arc (human only) --------------------------------- #
        if mode == "human":
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

        # ---- hole ------------------------------------------------------ #
        pygame.draw.circle(canvas, pygame.Color("LightGray"),
                           (mx, my), int(self.hole_radius), width=0)
        pygame.draw.circle(canvas, pygame.Color("Gray"),
                           (mx, my), int(self.hole_radius), width=2)

        # ---- mole -------------------------------------------------------- #
        if self.red_mole_active:
            pygame.draw.circle(canvas, pygame.Color("OrangeRed"), (mx, my), int(self.mole_radius), width=0)

        # ---- agent + action ------------------------------------------- #
        draw_options = DrawOptions(canvas)
        self.space.debug_draw(draw_options)

        if self.render_action and self._last_action is not None:
            ax = int(np.clip(float(self._last_action[0]), 0, self.window_size - 1))
            ay = int(np.clip(float(self._last_action[1]), 0, self.window_size - 1))
            pygame.draw.circle(canvas, pygame.Color("Red"), (ax, ay), 6, width=2)

        # ---- HUD (human only) ----------------------------------------- #
        if mode == "human" and self._hud_font is not None:
            hit_surf  = self._hud_font.render(f"HIT:{self._hit_count}", True, (0, 128, 0))
            miss_surf = self._hud_font.render(f"MISS:{self._miss_count}", True, (180, 0, 0))
            time_surf = self._hud_font.render(
                f"T:{self._mole_step}/{self.visible_duration}", True, (60, 60, 60)
            )
            canvas.blit(hit_surf,  (8, 8))
            canvas.blit(miss_surf, (8, 28))
            canvas.blit(time_surf, (8, 48))

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["video.frames_per_second"])
            return None

        img = np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
        return img

    # ------------------------------------------------------------------ #
    # Teleop
    # ------------------------------------------------------------------ #

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple("TeleopAgent", ["act"])

        def act(obs):
            action = None
            mouse_pos = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_pos - self.agent.position).length < 30:
                self.teleop = True
                action = mouse_pos
            return action

        return TeleopAgent(act)
