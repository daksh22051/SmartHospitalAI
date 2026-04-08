"""
Frontend-independent OpenEnv inference runner.

This script runs a full episode directly against HospitalEnv using a
deterministic heuristic policy and prints a reproducible summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

try:
    from smart_hospital_orchestration.environment import HospitalEnv
except ModuleNotFoundError:
    # Allow running as: `python inference.py` from project root directory.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment import HospitalEnv


@dataclass
class EpisodeResult:
    task: str
    seed: int
    steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    final_waiting: int
    final_admitted: int
    final_critical_waiting: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "steps": self.steps,
            "total_reward": round(self.total_reward, 3),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_waiting": self.final_waiting,
            "final_admitted": self.final_admitted,
            "final_critical_waiting": self.final_critical_waiting,
        }


def _visible_patients(state: Dict[str, Any]) -> np.ndarray:
    patients = state.get("patients", np.array([]))
    time_stats = state.get("time", np.array([]))
    total_patients = int(time_stats[2]) if len(time_stats) > 2 else len(patients)
    visible = min(total_patients, len(patients))
    return patients[:visible] if visible > 0 else np.array([])


def _choose_action(state: Dict[str, Any]) -> int:
    """
    Deterministic heuristic policy.

    Action mapping:
      0 WAIT
      1 ALLOCATE_RESOURCE
      2 ESCALATE_PRIORITY
      3 DEFER
      4 REASSIGN
    """
    patients = _visible_patients(state)
    doctors = state.get("doctors", np.array([]))
    beds = state.get("beds", np.array([]))

    if len(patients) == 0:
        return 0

    waiting = patients[patients[:, 1] == 0.0] if len(patients) > 0 else np.array([])
    critical_waiting = waiting[waiting[:, 0] == 2.0] if len(waiting) > 0 else np.array([])
    emergency_waiting = waiting[waiting[:, 0] == 1.0] if len(waiting) > 0 else np.array([])

    active_doctors = doctors[doctors[:, 3] > 0] if len(doctors) > 0 else np.array([])
    active_beds = beds[beds[:, 3] > 0] if len(beds) > 0 else np.array([])
    available_doctors = np.sum(active_doctors[:, 1] == 1.0) if len(active_doctors) > 0 else 0
    available_beds = np.sum(active_beds[:, 1] == 1.0) if len(active_beds) > 0 else 0

    if (len(critical_waiting) > 0 or len(emergency_waiting) > 0) and available_doctors > 0 and available_beds > 0:
        return 1

    if len(critical_waiting) > 0:
        waited_too_long = np.any(critical_waiting[:, 2] >= 3.0)
        if waited_too_long:
            return 2

    system_load = len(waiting) / max((len(active_doctors) + len(active_beds)), 1)
    if system_load > 1.4:
        return 3

    # Reassign if any doctor appears overloaded
    if len(active_doctors) > 0 and np.any(active_doctors[:, 2] >= 2):
        return 4

    if available_doctors > 0 and available_beds > 0 and len(waiting) > 0:
        return 1

    return 0


def run_episode(task: str, seed: int, max_steps: int | None = None, verbose: bool = False) -> EpisodeResult:
    """Run one full episode from reset() to done using step(action)."""
    env = HospitalEnv(task=task)
    state, _ = env.reset(seed=seed)

    done = False
    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0.0

    while not done:
        action = _choose_action(state)
        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))

        if verbose:
            print(
                f"step={step_count:3d} action={action} reward={reward:7.2f} "
                f"waiting={info.get('waiting_patients', 0)} admitted={info.get('admitted_patients', 0)}"
            )

        if max_steps is not None and step_count >= max_steps:
            break

    final = env.state()["readable"]
    env.close()

    return EpisodeResult(
        task=task,
        seed=seed,
        steps=step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        final_waiting=int(final["waiting"]),
        final_admitted=int(final["admitted"]),
        final_critical_waiting=int(final["critical_waiting"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenEnv inference runner (frontend-independent)")
    parser.add_argument("--task", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    result = run_episode(task=args.task, seed=args.seed, max_steps=args.max_steps, verbose=args.verbose)
    print("INFERENCE_RESULT=" + json.dumps(result.to_dict(), separators=(",", ":")))


if __name__ == "__main__":
    main()

