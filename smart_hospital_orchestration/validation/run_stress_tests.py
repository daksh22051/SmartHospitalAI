"""Run large-load stress tests and persist summary report."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    from environment.hospital_env import HospitalEnv
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from environment.hospital_env import HospitalEnv


def run_one(task: str, seed: int, max_steps: int) -> Dict[str, Any]:
    env = HospitalEnv(task=task)
    state, _ = env.reset(seed=seed)
    start = time.perf_counter()

    done = False
    steps = 0
    total_reward = 0.0

    while not done and steps < max_steps:
        readable = state.get("readable", {})
        waiting = int(readable.get("waiting", 0))
        critical = int(readable.get("critical_waiting", 0))
        doctors = int(readable.get("available_doctors", 0))
        beds = int(readable.get("available_beds", 0))

        if waiting <= 0:
            action = 0
        elif critical > 0 and doctors > 0 and beds > 0:
            action = 1
        elif critical > 0:
            action = 2
        elif doctors > 0 and beds > 0:
            action = 1
        else:
            action = 0

        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1

    elapsed = time.perf_counter() - start
    final = env.state().get("readable", {})
    env.close()

    return {
        "seed": seed,
        "steps": steps,
        "elapsed_seconds": round(elapsed, 6),
        "total_reward": round(total_reward, 6),
        "final_waiting": int(final.get("waiting", 0)),
        "final_admitted": int(final.get("admitted", 0)),
        "final_critical_waiting": int(final.get("critical_waiting", 0)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="hard", choices=["easy", "medium", "hard"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--runtime-limit-seconds", type=float, default=1200.0)
    p.add_argument("--target-cpus", type=float, default=2.0)
    p.add_argument("--target-memory-gb", type=float, default=8.0)
    p.add_argument("--output", default="results/stress_test_report.json")
    args = p.parse_args()

    runs: List[Dict[str, Any]] = []
    for i in range(args.episodes):
        runs.append(run_one(args.task, args.seed + i, args.max_steps))

    elapsed = np.array([r["elapsed_seconds"] for r in runs], dtype=np.float64)
    waiting = np.array([r["final_waiting"] for r in runs], dtype=np.float64)
    critical = np.array([r["final_critical_waiting"] for r in runs], dtype=np.float64)
    admitted = np.array([r["final_admitted"] for r in runs], dtype=np.float64)

    summary = {
        "episodes": args.episodes,
        "task": args.task,
        "avg_elapsed_seconds": round(float(np.mean(elapsed)), 6),
        "p95_elapsed_seconds": round(float(np.percentile(elapsed, 95)), 6),
        "max_elapsed_seconds": round(float(np.max(elapsed)), 6),
        "avg_final_waiting": round(float(np.mean(waiting)), 6),
        "avg_final_critical_waiting": round(float(np.mean(critical)), 6),
        "avg_final_admitted": round(float(np.mean(admitted)), 6),
    }

    checks = {
        "runtime_within_limit": bool(float(np.sum(elapsed)) < float(args.runtime_limit_seconds)),
        "p95_runtime_reasonable": bool(float(np.percentile(elapsed, 95)) < float(args.runtime_limit_seconds) / max(args.episodes, 1)),
        "resource_envelope_declared": True,
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_envelope": {
            "cpus": args.target_cpus,
            "memory_gb": args.target_memory_gb,
            "runtime_limit_seconds": args.runtime_limit_seconds,
            "note": "Enforce CPU/memory limits at container runtime using docker --cpus and --memory.",
        },
        "summary": summary,
        "checks": checks,
        "runs": runs,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"STRESS_TEST_REPORT={out}")


if __name__ == "__main__":
    main()
