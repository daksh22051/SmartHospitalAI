"""Train/tune weighted multi-agent coordinator via random search."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:
    from environment.hospital_env import HospitalEnv
    from agent.multi_agent_extension import MultiAgentCoordinator
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from environment.hospital_env import HospitalEnv
    from agent.multi_agent_extension import MultiAgentCoordinator


def evaluate_weights(task: str, seed: int, episodes: int, max_steps: int, weights: Dict[str, float]) -> Dict[str, Any]:
    rewards = []
    waits = []
    admits = []

    for i in range(episodes):
        env = HospitalEnv(task=task)
        coord = MultiAgentCoordinator(
            triage_weight=float(weights["triage"]),
            capacity_weight=float(weights["capacity"]),
            transfer_weight=float(weights["transfer"]),
        )
        env.reset(seed=seed + i)

        total_reward = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            readable = env.state().get("readable", {})
            action, _ = coord.select_action(readable)
            _, reward, done, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

        final = env.state().get("readable", {})
        env.close()

        rewards.append(total_reward)
        waits.append(float(final.get("waiting", 0)))
        admits.append(float(final.get("admitted", 0)))

    avg_reward = float(np.mean(rewards))
    avg_wait = float(np.mean(waits))
    avg_admit = float(np.mean(admits))

    # Composite objective: maximize reward + admissions, minimize waiting.
    objective = avg_reward + (8.0 * avg_admit) - (15.0 * avg_wait)
    return {
        "objective": objective,
        "avg_total_reward": avg_reward,
        "avg_final_waiting": avg_wait,
        "avg_final_admitted": avg_admit,
        "episodes": episodes,
    }


def random_weight(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "triage": float(rng.uniform(0.6, 2.0)),
        "capacity": float(rng.uniform(0.6, 2.0)),
        "transfer": float(rng.uniform(0.6, 2.0)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=80)
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--output", default="checkpoints/multi_agent_weights.json")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    best = None
    history = []

    for _ in range(args.trials):
        weights = random_weight(rng)
        metrics = evaluate_weights(
            task=args.task,
            seed=args.seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
            weights=weights,
        )
        row = {"weights": weights, "metrics": metrics}
        history.append(row)
        if best is None or metrics["objective"] > best["metrics"]["objective"]:
            best = row

    assert best is not None

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy": "multi_agent_weighted",
        "task": args.task,
        "seed": args.seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "weights": best["weights"],
        "metrics": best["metrics"],
        "search": {
            "trials": args.trials,
            "episodes": args.episodes,
            "max_steps": args.max_steps,
        },
        "top_5": sorted(history, key=lambda x: x["metrics"]["objective"], reverse=True)[:5],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"MULTI_AGENT_TRAINED_CHECKPOINT={out}")


if __name__ == "__main__":
    main()
