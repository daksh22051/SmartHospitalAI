"""Main CLI for simulation, lightweight training workflow, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from smart_hospital_orchestration.environment import HospitalEnv
    from smart_hospital_orchestration.evaluation import grade_environment
    from smart_hospital_orchestration.evaluation import write_grader_report
    from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent
except ModuleNotFoundError:
    # Support execution as: python smart_hospital_orchestration/main.py
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from smart_hospital_orchestration.environment import HospitalEnv
    from smart_hospital_orchestration.evaluation import grade_environment
    from smart_hospital_orchestration.evaluation import write_grader_report
    from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent


def _task_from_config(config_path: str) -> str:
    """Infer task name from config path string with safe fallback."""
    lowered = config_path.lower()
    if "easy" in lowered:
        return "easy"
    if "hard" in lowered:
        return "hard"
    return "medium"


def _run_episode(task: str, seed: int, policy: str) -> Dict[str, Any]:
    env = HospitalEnv(task=task)
    state, _ = env.reset(seed=seed)
    agent = HospitalBaselineAgent(verbose=False)

    done = False
    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not done:
        if policy == "random":
            action = int(env.action_space.sample())
        else:
            action = int(agent.select_action(state))
        state, reward, done, info = env.step(action)
        steps += 1
        total_reward += float(reward)
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))

    final = env.state().get("readable", {})
    env.close()
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": round(total_reward, 6),
        "terminated": terminated,
        "truncated": truncated,
        "final_waiting": int(final.get("waiting", 0)),
        "final_admitted": int(final.get("admitted", 0)),
        "final_critical_waiting": int(final.get("critical_waiting", 0)),
    }


def _summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not runs:
        return {
            "episodes": 0,
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_final_waiting": 0.0,
            "avg_final_admitted": 0.0,
        }
    count = len(runs)
    return {
        "episodes": count,
        "avg_reward": round(sum(r["total_reward"] for r in runs) / count, 6),
        "avg_steps": round(sum(r["steps"] for r in runs) / count, 6),
        "avg_final_waiting": round(sum(r["final_waiting"] for r in runs) / count, 6),
        "avg_final_admitted": round(sum(r["final_admitted"] for r in runs) / count, 6),
    }


def run_simulation(
    config_path: str,
    agent_type: str = "random",
    episodes: int = 1,
    render: bool = False
) -> None:
    """
    Run a simulation with the specified configuration.
    
    Args:
        config_path: Path to configuration file
        agent_type: Type of agent to use
        episodes: Number of episodes to run
        render: Whether to render the environment
    """
    del render  # Rendering is not used in this CLI implementation.
    task = _task_from_config(config_path)
    policy = "heuristic" if agent_type == "heuristic" else "random"

    runs: List[Dict[str, Any]] = []
    for i in range(max(1, episodes)):
        runs.append(_run_episode(task=task, seed=42 + i, policy=policy))

    payload = {
        "mode": "simulate",
        "task": task,
        "policy": policy,
        "summary": _summarize_runs(runs),
        "episodes": runs,
    }
    print("SIMULATION_RESULT=" + json.dumps(payload, separators=(",", ":")))


def train_agent(
    config_path: str,
    algorithm: str = "ppo",
    timesteps: int = 1000000,
    save_path: Optional[str] = None,
    ppo_lr: float = 1e-4,
    ppo_gamma: float = 0.995,
    ppo_gae_lambda: float = 0.97,
    ppo_clip_eps: float = 0.15,
    ppo_entropy_coef: float = 0.002,
    ppo_value_coef: float = 0.7,
    ppo_update_epochs: int = 6,
    ppo_minibatch_size: int = 128,
    ppo_rollout_steps: int = 1024,
    ppo_max_grad_norm: float = 0.5,
) -> None:
    """
    Train an RL agent.
    
    Args:
        config_path: Path to configuration file
        algorithm: RL algorithm to use
        timesteps: Total training timesteps
        save_path: Path to save trained model
    """
    task = _task_from_config(config_path)
    algo = algorithm.lower().strip()

    if algo == "ppo":
        from smart_hospital_orchestration.training import train_ppo
        from smart_hospital_orchestration.training.ppo_trainer import PPOConfig

        checkpoint = save_path or f"checkpoints/ppo_{task}.pt"
        ppo_cfg = PPOConfig(
            lr=float(ppo_lr),
            gamma=float(ppo_gamma),
            gae_lambda=float(ppo_gae_lambda),
            clip_eps=float(ppo_clip_eps),
            entropy_coef=float(ppo_entropy_coef),
            value_coef=float(ppo_value_coef),
            update_epochs=int(ppo_update_epochs),
            minibatch_size=int(ppo_minibatch_size),
            rollout_steps=int(ppo_rollout_steps),
            max_grad_norm=float(ppo_max_grad_norm),
        )
        result = train_ppo(
            task=task,
            total_timesteps=max(1000, int(timesteps)),
            seed=42,
            save_path=checkpoint,
            config=ppo_cfg,
        )
        print("TRAINING_RESULT=" + json.dumps(result, separators=(",", ":")))
        return

    policy = "random" if algo == "random" else "heuristic"

    # Lightweight training loop for hackathon baseline packaging.
    episodes = max(1, min(25, timesteps // 50000))
    runs: List[Dict[str, Any]] = []
    for i in range(episodes):
        runs.append(_run_episode(task=task, seed=100 + i, policy=policy))

    summary = _summarize_runs(runs)
    artifact = {
        "artifact_type": "baseline_policy",
        "algorithm": algorithm,
        "policy": policy,
        "task": task,
        "timesteps_requested": timesteps,
        "episodes_used": episodes,
        "summary": summary,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if save_path:
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        save_file.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print("TRAINING_RESULT=" + json.dumps(artifact, separators=(",", ":")))


def evaluate_agent(
    model_path: str,
    config_path: str,
    episodes: int = 10,
    pass_threshold: Optional[float] = None,
    rubric_profile: str = "hackathon_v1",
    enable_llm_score: bool = False,
    output_path: str = "",
    save_history: bool = True,
    history_dir: str = "results/grader_history",
) -> None:
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to trained model
        config_path: Path to configuration file
        episodes: Number of evaluation episodes
    """
    task = _task_from_config(config_path)
    policy = "heuristic"
    resolved_model_path = ""

    if model_path and os.path.exists(model_path):
        try:
            raw = json.loads(Path(model_path).read_text(encoding="utf-8"))
            candidate = str(raw.get("policy", "heuristic")).lower()
            if candidate in {"heuristic", "random", "ppo"}:
                policy = candidate
            if policy == "ppo":
                checkpoint = str(raw.get("checkpoint", "")).strip()
                if checkpoint:
                    cp_path = Path(checkpoint)
                    if cp_path.is_absolute():
                        resolved_model_path = str(cp_path)
                    else:
                        meta_base = Path(model_path).resolve().parent
                        candidate_paths = [
                            (meta_base / cp_path).resolve(),
                            (Path.cwd() / cp_path).resolve(),
                            (Path(__file__).resolve().parent.parent / cp_path).resolve(),
                        ]
                        existing = next((p for p in candidate_paths if p.exists()), None)
                        resolved_model_path = str(existing if existing is not None else candidate_paths[0])
        except Exception:
            # Non-JSON file is likely a direct checkpoint path.
            if str(model_path).lower().endswith(".pt"):
                policy = "ppo"
                resolved_model_path = model_path
            else:
                policy = "heuristic"

    if policy == "ppo" and not resolved_model_path and model_path:
        if str(model_path).lower().endswith(".pt"):
            resolved_model_path = model_path

    report = grade_environment(
        task=task,
        episodes=max(1, episodes),
        seed_start=42,
        policy=policy,
        model_path=resolved_model_path,
        pass_threshold=pass_threshold,
        rubric_profile=rubric_profile,
        enable_llm_score=enable_llm_score,
    )
    writes = write_grader_report(
        report,
        output_path=output_path,
        save_history=save_history,
        history_dir=history_dir,
    )
    print("EVALUATION_RESULT=" + json.dumps(report.to_dict(), separators=(",", ":")))
    if writes:
        print("EVALUATION_WRITES=" + json.dumps(writes, separators=(",", ":")))


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Smart Hospital Orchestration Environment"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to configuration file"
    )
    sim_parser.add_argument(
        "--agent", "-a",
        default="random",
        choices=["random", "heuristic"],
        help="Agent type"
    )
    sim_parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=1,
        help="Number of episodes"
    )
    sim_parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render environment"
    )
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train agent")
    train_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to configuration file"
    )
    train_parser.add_argument(
        "--algorithm", "-alg",
        default="ppo",
        help="RL algorithm"
    )
    train_parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=1000000,
        help="Training timesteps"
    )
    train_parser.add_argument(
        "--save-path", "-s",
        help="Path to save model"
    )
    train_parser.add_argument("--ppo-lr", type=float, default=1e-4, help="PPO learning rate")
    train_parser.add_argument("--ppo-gamma", type=float, default=0.995, help="PPO discount factor")
    train_parser.add_argument("--ppo-gae-lambda", type=float, default=0.97, help="PPO GAE lambda")
    train_parser.add_argument("--ppo-clip-eps", type=float, default=0.15, help="PPO clip epsilon")
    train_parser.add_argument("--ppo-entropy-coef", type=float, default=0.002, help="PPO entropy coefficient")
    train_parser.add_argument("--ppo-value-coef", type=float, default=0.7, help="PPO value loss coefficient")
    train_parser.add_argument("--ppo-update-epochs", type=int, default=6, help="PPO epochs per rollout")
    train_parser.add_argument("--ppo-minibatch-size", type=int, default=128, help="PPO minibatch size")
    train_parser.add_argument("--ppo-rollout-steps", type=int, default=1024, help="PPO rollout steps per update")
    train_parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5, help="PPO max gradient norm")
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agent")
    eval_parser.add_argument(
        "--model", "-m",
        default="",
        help="Optional model artifact path (JSON produced by train command)"
    )
    eval_parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to configuration file"
    )
    eval_parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=10,
        help="Number of evaluation episodes"
    )
    eval_parser.add_argument(
        "--pass-threshold",
        type=float,
        default=None,
        help="Optional pass threshold override. Uses task defaults when omitted."
    )
    eval_parser.add_argument(
        "--rubric-profile",
        choices=["hackathon_v1", "balanced"],
        default="hackathon_v1",
        help="Scoring rubric profile"
    )
    eval_parser.add_argument(
        "--enable-llm-score",
        action="store_true",
        default=False,
        help="Enable optional LLM scoring hook (uses env-configured endpoint if available)"
    )
    eval_parser.add_argument(
        "--output",
        default="",
        help="Optional path to write full evaluation report JSON"
    )
    eval_parser.add_argument(
        "--history-dir",
        default="results/grader_history",
        help="Directory for timestamped evaluation history files"
    )
    eval_parser.add_argument(
        "--no-history",
        action="store_true",
        default=False,
        help="Disable automatic history file writes"
    )
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        run_simulation(
            config_path=args.config,
            agent_type=args.agent,
            episodes=args.episodes,
            render=args.render
        )
    elif args.command == "train":
        train_agent(
            config_path=args.config,
            algorithm=args.algorithm,
            timesteps=args.timesteps,
            save_path=args.save_path,
            ppo_lr=args.ppo_lr,
            ppo_gamma=args.ppo_gamma,
            ppo_gae_lambda=args.ppo_gae_lambda,
            ppo_clip_eps=args.ppo_clip_eps,
            ppo_entropy_coef=args.ppo_entropy_coef,
            ppo_value_coef=args.ppo_value_coef,
            ppo_update_epochs=args.ppo_update_epochs,
            ppo_minibatch_size=args.ppo_minibatch_size,
            ppo_rollout_steps=args.ppo_rollout_steps,
            ppo_max_grad_norm=args.ppo_max_grad_norm,
        )
    elif args.command == "evaluate":
        evaluate_agent(
            model_path=args.model,
            config_path=args.config,
            episodes=args.episodes,
            pass_threshold=args.pass_threshold,
            rubric_profile=args.rubric_profile,
            enable_llm_score=args.enable_llm_score,
            output_path=args.output,
            save_history=not args.no_history,
            history_dir=args.history_dir,
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
