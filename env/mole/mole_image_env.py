import cv2
import numpy as np
from gymnasium import spaces

from env.mole.mole_v1_env import WhackAMoleV1Env
from env.mole.mole_v2_env import WhackAMoleV2Env


class WhackAMoleV1ImageEnv(WhackAMoleV1Env):
    """V1 (timeout + miss) with image + agent_pos observation."""

    def __init__(self, render_size=96, **kwargs):
        super().__init__(render_size=render_size, **kwargs)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(low=0, high=255, shape=(3, render_size, render_size), dtype=np.float32),
            "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
        })

    def _get_obs(self):
        img = self._render_frame("rgb_array")
        img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)
        img_chw = np.moveaxis(img.astype(np.float32), -1, 0)
        agent_pos = np.array([float(self.agent.position.x), float(self.agent.position.y)], dtype=np.float32)
        return {"image": img_chw, "agent_pos": agent_pos}


class WhackAMoleV2ImageEnv(WhackAMoleV2Env):
    """V2 (no timeout, hit-count) with image + agent_pos observation."""

    def __init__(self, render_size=96, **kwargs):
        super().__init__(render_size=render_size, **kwargs)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(low=0, high=255, shape=(3, render_size, render_size), dtype=np.float32),
            "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
        })

    def _get_obs(self):
        img = self._render_frame("rgb_array")
        img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)
        img_chw = np.moveaxis(img.astype(np.float32), -1, 0)
        agent_pos = np.array([float(self.agent.position.x), float(self.agent.position.y)], dtype=np.float32)
        return {"image": img_chw, "agent_pos": agent_pos}


# backward-compat aliases
WhackAMoleImageEnv    = WhackAMoleV1ImageEnv
WhackAMoleReactiveEnv = WhackAMoleV1ImageEnv

__all__ = [
    "WhackAMoleV1ImageEnv",
    "WhackAMoleV2ImageEnv",
    "WhackAMoleImageEnv",
    "WhackAMoleReactiveEnv",
]
