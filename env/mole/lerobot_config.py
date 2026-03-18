"""LeRobot integration for WhackAMole env.

Registers MoleEnvConfig with lerobot's EnvConfig ChoiceRegistry and
a gym env (mole_lerobot/WhackAMole-v0) that returns observations in
the format expected by lerobot's preprocess_observation:
  - "pixels": HWC uint8  (96 x 96 x 3)
  - "agent_pos": float32 (2,)
"""

from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGE, OBS_STATE


# ------------------------------------------------------------------ #
# Gym wrapper: CHW float32 → HWC uint8 pixels
# ------------------------------------------------------------------ #

class MoleLeRobotWrapper(gym.Wrapper):
    """Wraps WhackAMoleImageEnv to produce lerobot-compatible observations.

    preprocess_observation expects:
      "pixels"    — HWC uint8
      "agent_pos" — float32 (2,)
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, render_size: int = 96, **kwargs):
        from env.mole.mole_image_env import WhackAMoleImageEnv

        env = WhackAMoleImageEnv(render_size=render_size, render_mode="rgb_array", **kwargs)
        super().__init__(env)

        ws = env.window_size
        self.observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Box(
                low=0, high=255, shape=(render_size, render_size, 3), dtype=np.uint8
            ),
            "agent_pos": gym.spaces.Box(low=0.0, high=float(ws), shape=(2,), dtype=np.float32),
        })

    def _convert(self, obs: dict) -> dict:
        # image is CHW float32 [0, 255] → HWC uint8
        pixels = np.moveaxis(obs["image"], 0, -1).astype(np.uint8)
        return {"pixels": pixels, "agent_pos": obs["agent_pos"]}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Replace None values so gymnasium vector env can store info in numpy arrays
        info = {k: (-1 if v is None else v) for k, v in info.items()
                if not isinstance(v, (list, dict, np.ndarray))}
        # lerobot eval requires is_success in terminal info
        info["is_success"] = bool(info.get("hit_count", 0) > 0)
        return self._convert(obs), reward, terminated, truncated, info


register(
    id="mole_lerobot/WhackAMole-v0",
    entry_point="env.mole.lerobot_config:MoleLeRobotWrapper",
    max_episode_steps=200,
)


# ------------------------------------------------------------------ #
# LeRobot EnvConfig
# ------------------------------------------------------------------ #

@EnvConfig.register_subclass("mole")
@dataclass
class MoleEnvConfig(EnvConfig):
    task: str = "WhackAMole-v0"
    fps: int = 10
    episode_length: int = 200
    render_size: int = 96

    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(96, 96, 3)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            ACTION: ACTION,
            "pixels": OBS_IMAGE,
            "agent_pos": OBS_STATE,
        }
    )

    @property
    def package_name(self) -> str:
        # lerobot will importlib.import_module(package_name) to register the gym env
        return "env.mole.lerobot_config"

    @property
    def gym_id(self) -> str:
        return "mole_lerobot/WhackAMole-v0"

    @property
    def gym_kwargs(self) -> dict:
        return {"render_size": self.render_size}
