"""
Smart Hospital Resource Orchestration Environment

An AI-driven simulation environment for hospital resource management
compatible with reinforcement learning frameworks.
"""

__version__ = "0.1.0"
__author__ = "Hospital AI Team"

try:
    from gymnasium.envs.registration import register, registry
except Exception:  # pragma: no cover - gymnasium is an install-time dependency
    register = None  # type: ignore[assignment]
    registry = None  # type: ignore[assignment]

from .environment import HospitalEnv
from .state import StateRepresentation
from .reward import RewardFunction


def _register_gymnasium_envs() -> None:
    """Register task-specific Gymnasium environment IDs."""
    if register is None or registry is None:
        return

    env_specs = {
        "SmartHospitalOrchestration-easy-v0": {"task": "easy", "max_episode_steps": 50},
        "SmartHospitalOrchestration-medium-v0": {"task": "medium", "max_episode_steps": 75},
        "SmartHospitalOrchestration-hard-v0": {"task": "hard", "max_episode_steps": 100},
    }

    for env_id, spec in env_specs.items():
        if env_id in registry:
            continue
        register(
            id=env_id,
            entry_point="smart_hospital_orchestration.environment.gym_adapter:GymnasiumHospitalEnv",
            kwargs={"task": spec["task"]},
            max_episode_steps=int(spec["max_episode_steps"]),
        )


_register_gymnasium_envs()

__all__ = [
    "HospitalEnv",
    "StateRepresentation",
    "RewardFunction",
]
