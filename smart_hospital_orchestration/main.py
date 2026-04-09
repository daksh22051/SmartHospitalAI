"""Main CLI for simulation, lightweight training workflow, and evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ==========================================
# SCALER PROXY & ENVIRONMENT CONFIGURATION
# ==========================================
# Ye ensure karta hai ki aapka LLM traffic Scaler ke server se jaye
API_BASE = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")

if API_BASE:
    # Forcefully setting env variables for libraries like OpenAI/LiteLLM
    os.environ["OPENAI_API_BASE"] = API_BASE
    os.environ["OPENAI_API_KEY"] = API_KEY
    # Kuch libraries ye format bhi use karti hain
    os.environ["LITELLM_API_BASE"] = API_BASE
    os.environ["LITELLM_API_KEY"] = API_KEY

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
    enable_llm_score: bool = True,  # FORCE TRUE FOR SCALER PROXY CHECK
    output_path: str = "",
    save_history: bool = True,
    history_dir: str = "results/grader_history",
) -> None:
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
            if str(model_path).lower().endswith(".pt"):
                policy = "ppo"
                resolved_model_path = model_path
            else:
                policy = "heuristic"

    if policy == "ppo" and not resolved_model_path and model_path:
        if str(model_path).lower().endswith(".pt"):
            resolved_model_path = model_path

    # CALLING THE GRADER (Ensure it sees proxy credentials)
    report = grade_environment(
        task=task,
        episodes=max(1, episodes),
        seed_start=42,
        policy=policy,
        model_path=resolved_model_path,
        pass_threshold=pass_threshold,
        rubric_profile=rubric_profile,
        enable_llm_score=True,  # ALWAYS TRUE FOR SUBMISSION
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
    parser = argparse.ArgumentParser(description="Smart Hospital Orchestration Environment")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # --- Simulation Command ---
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument("--config", "-c", required=True)
    sim_parser.add_argument("--agent", "-a", default="random", choices=["random", "heuristic"])
    sim_parser.add_argument("--episodes", "-e", type=int, default=1)
    sim_parser.add_argument("--render", "-r", action="store_true")
    
    # --- Training Command ---
    train_parser = subparsers.add_parser("train", help="Train agent")
    train_parser.add_argument("--config", "-c", required=True)
    train_parser.add_argument("--algorithm", "-alg", default="ppo")
    train_parser.add_argument("--timesteps", "-t", type=int, default=1000000)
    train_parser.add_argument("--save-path", "-s")
    # PPO args...
    train_parser.add_argument("--ppo-lr", type=float, default=1e-4)
    train_parser.add_argument("--ppo-gamma", type=float, default=0.995)
    train_parser.add_argument("--ppo-gae-lambda", type=float, default=0.97)
    train_parser.add_argument("--ppo-clip-eps", type=float, default=0.15)
    train_parser.add_argument("--ppo-entropy-coef", type=float, default=0.002)
    train_parser.add_argument("--ppo-value-coef", type=float, default=0.7)
    train_parser.add_argument("--ppo-update-epochs", type=int, default=6)
    train_parser.add_argument("--ppo-minibatch-size", type=int, default=128)
    train_parser.add_argument("--ppo-rollout-steps", type=int, default=1024)
    train_parser.add_argument("--ppo-max-grad-norm", type=float, default=0.5)
    
    # --- Evaluation Command ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agent")
    eval_parser.add_argument("--model", "-m", default="")
    eval_parser.add_argument("--config", "-c", required=True)
    eval_parser.add_argument("--episodes", "-e", type=int, default=10)
    eval_parser.add_argument("--pass-threshold", type=float, default=None)
    eval_parser.add_argument("--rubric-profile", choices=["hackathon_v1", "balanced"], default="hackathon_v1")
    eval_parser.add_argument("--enable-llm-score", action="store_true", default=True) # Always True
    eval_parser.add_argument("--output", default="")
    eval_parser.add_argument("--history-dir", default="results/grader_history")
    eval_parser.add_argument("--no-history", action="store_true", default=False)
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        run_simulation(args.config, args.agent, args.episodes, args.render)
    elif args.command == "train":
        train_agent(args.config, args.algorithm, args.timesteps, args.save_path, 
                    args.ppo_lr, args.ppo_gamma, args.ppo_gae_lambda, args.ppo_clip_eps, 
                    args.ppo_entropy_coef, args.ppo_value_coef, args.ppo_update_epochs, 
                    args.ppo_minibatch_size, args.ppo_rollout_steps, args.ppo_max_grad_norm)
    elif args.command == "evaluate":
        evaluate_agent(args.model, args.config, args.episodes, args.pass_threshold, 
                       args.rubric_profile, True, args.output, not args.no_history, args.history_dir)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())