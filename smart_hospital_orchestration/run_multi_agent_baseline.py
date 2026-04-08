"""Run a multi-agent baseline episode and print structured result."""

from __future__ import annotations

import argparse
import json

from environment.hospital_env import HospitalEnv
from agent.multi_agent_extension import MultiAgentCoordinator


def run_episode(task: str, seed: int, max_steps: int, checkpoint: str = "") -> dict:
    env = HospitalEnv(task=task)
    coord = MultiAgentCoordinator.from_checkpoint(checkpoint) if checkpoint else MultiAgentCoordinator()

    env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while steps < max_steps:
        readable = env.state().get("readable", {})
        action, meta = coord.select_action(readable)
        _, reward, done, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if done:
            break

    final = env.state().get("readable", {})
    env.close()

    result = {
        "task": task,
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 6),
        "terminated": terminated,
        "truncated": truncated,
        "final_waiting": int(final.get("waiting", 0)),
        "final_admitted": int(final.get("admitted", 0)),
        "final_critical_waiting": int(final.get("critical_waiting", 0)),
        "policy_source": "multi_agent_weighted" if checkpoint else "multi_agent",
    }
    print("MULTI_AGENT_RESULT=" + json.dumps(result, separators=(",", ":")))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=75)
    p.add_argument("--checkpoint", type=str, default="")
    args = p.parse_args()
    run_episode(args.task, args.seed, args.max_steps, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
