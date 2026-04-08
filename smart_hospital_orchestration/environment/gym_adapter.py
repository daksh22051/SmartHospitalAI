"""Gymnasium-compatible adapter for HospitalEnv."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from .hospital_env import HospitalEnv


class GymnasiumHospitalEnv(gym.Env):
    """Thin wrapper that adapts HospitalEnv to Gymnasium's Env interface."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, task: str = "medium") -> None:
        super().__init__()
        self.env = HospitalEnv(task=task)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _flat_observation(self) -> np.ndarray:
        return self.env.state().get("flat")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        del options
        self.env.reset(seed=seed)
        return self._flat_observation(), self.env.state().get("readable", {})

    def step(self, action: int):
        _, reward, done, step_info = self.env.step(action)
        state = self._flat_observation()
        info = self.env.state().get("readable", {})
        terminated = bool(step_info.get("terminated", False))
        truncated = bool(step_info.get("truncated", False))
        if bool(done) and not (terminated or truncated):
            truncated = True
        return state, float(reward), terminated, truncated, info

    def state(self) -> Dict[str, Any]:
        return self.env.state()

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
