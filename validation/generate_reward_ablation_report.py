"""Generate reward-ablation experiments and write report JSON."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
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


def _ci95(values: np.ndarray) -> float:
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


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

    totals_arr = np.array(totals, dtype=np.float64)
    waits_arr = np.array(waits, dtype=np.float64)
    admits_arr = np.array(admits, dtype=np.float64)

    return {
        "avg_total_reward": round(float(np.mean(totals_arr)), 6),
        "std_total_reward": round(float(np.std(totals_arr, ddof=0)), 6),
        "ci95_total_reward": round(_ci95(totals_arr), 6),
        "avg_final_waiting": round(float(np.mean(waits_arr)), 6),
        "std_final_waiting": round(float(np.std(waits_arr, ddof=0)), 6),
        "avg_final_admitted": round(float(np.mean(admits_arr)), 6),
        "std_final_admitted": round(float(np.std(admits_arr, ddof=0)), 6),
        "episodes": episodes,
        "weights_override": weights,
        "series": {
            "total_reward": [round(float(x), 6) for x in totals],
            "final_waiting": [round(float(x), 6) for x in waits],
            "final_admitted": [round(float(x), 6) for x in admits],
        },
    }


def write_plots(report: Dict[str, Any], plot_output: Path) -> None:
    ablations = report.get("ablations", {})
    names = list(ablations.keys())
    if not names:
        return

    rewards = [float(ablations[n].get("avg_total_reward", 0.0)) for n in names]
    cis = [float(ablations[n].get("ci95_total_reward", 0.0)) for n in names]
    waits = [float(ablations[n].get("avg_final_waiting", 0.0)) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].bar(names, rewards, yerr=cis, capsize=5, color="#0ea5e9", alpha=0.9)
    axes[0].set_title("Reward Ablation: Avg Total Reward (CI95)")
    axes[0].set_ylabel("Avg Total Reward")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, waits, color="#f97316", alpha=0.9)
    axes[1].set_title("Reward Ablation: Avg Final Waiting")
    axes[1].set_ylabel("Avg Final Waiting")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--output", default="results/reward_ablation_report.json")
    p.add_argument("--plot-output", default="results/reward_ablation_plot.png")
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
    write_plots(report, Path(args.plot_output))
    print(f"REWARD_ABLATION_REPORT={out}")
    print(f"REWARD_ABLATION_PLOT={args.plot_output}")


if __name__ == "__main__":
    main()
