"""Generate reward-ablation experiments and write report JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np

try:
    from environment.hospital_env import HospitalEnv
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from environment.hospital_env import HospitalEnv


ABLATIONS: Dict[str, Dict[str, float]] = {
    "baseline": {},
    "no_wait_penalty": {"waiting_time": 0.0},
    "no_resource_bonus": {"resource_utilization": 0.0},
    "high_emergency_weight": {"emergency_handling": 1.2},
}


def run_policy(task: str, seed: int, episodes: int, weights: Dict[str, float]) -> Dict[str, Any]:
    totals = []
    waits = []
    admits = []

    for i in range(episodes):
        env = HospitalEnv(task=task)
        env.reward_function.update_weights(weights)
        state, _ = env.reset(seed=seed + i)

        done = False
        total_reward = 0.0
        while not done:
            readable = state.get("readable", {})
            waiting = int(readable.get("waiting", 0))
            critical = int(readable.get("critical_waiting", 0))
            doctors = int(readable.get("available_doctors", 0))
            beds = int(readable.get("available_beds", 0))

            if waiting <= 0:
                action = 0
            elif critical > 0 and doctors > 0 and beds > 0:
                action = 1
            elif critical > 0 and (doctors <= 0 or beds <= 0):
                action = 2
            elif doctors > 0 and beds > 0:
                action = 1
            else:
                action = 0

            state, reward, done, _ = env.step(action)
            total_reward += float(reward)

        final = env.state().get("readable", {})
        env.close()

        totals.append(total_reward)
        waits.append(float(final.get("waiting", 0)))
        admits.append(float(final.get("admitted", 0)))

    return {
        "avg_total_reward": round(float(np.mean(totals)), 6),
        "avg_final_waiting": round(float(np.mean(waits)), 6),
        "avg_final_admitted": round(float(np.mean(admits)), 6),
        "episodes": episodes,
        "weights_override": weights,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--output", default="results/reward_ablation_report.json")
    args = p.parse_args()

    report = {
        "task": args.task,
        "seed": args.seed,
        "episodes": args.episodes,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ablations": {},
    }

    for name, weights in ABLATIONS.items():
        report["ablations"][name] = run_policy(
            task=args.task,
            seed=args.seed,
            episodes=args.episodes,
            weights=weights,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REWARD_ABLATION_REPORT={out}")


if __name__ == "__main__":
    main()
