import cv2
import numpy as np
from gymnasium import spaces
from env.mole.mole_env import WhackAMoleEnv


class WhackAMoleImageEnv(WhackAMoleEnv):
    def __init__(self, render_size=96, **kwargs):
        super().__init__(render_size=render_size, **kwargs)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            "image":     spaces.Box(low=0, high=255, shape=(3, render_size, render_size), dtype=np.float32),
            "agent_pos": spaces.Box(low=0, high=ws, shape=(2,), dtype=np.float32),
        })

    def _get_obs(self):
        img = self._render_frame("rgb_array")   # HWC uint8, window_size x window_size
        img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_AREA)
        img_chw = np.moveaxis(img.astype(np.float32), -1, 0)   # CHW float32 [0,255]
        agent_pos = np.array([float(self.agent.position.x), float(self.agent.position.y)], dtype=np.float32)
        return {"image": img_chw, "agent_pos": agent_pos}


# backward compatibility alias
WhackAMoleReactiveEnv = WhackAMoleImageEnv

__all__ = ["WhackAMoleImageEnv", "WhackAMoleReactiveEnv"]
