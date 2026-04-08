"""Programmatic grader for submission-style evaluation.

This module runs multiple episodes and computes a deterministic scorecard
that can be consumed by CI, scripts, and Docker evaluation services.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request as urlrequest
from urllib.error import URLError

import numpy as np

try:
    from smart_hospital_orchestration.environment import HospitalEnv
    from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from smart_hospital_orchestration.environment import HospitalEnv
    from smart_hospital_orchestration.inference.baseline_inference import HospitalBaselineAgent


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_percent_score(value_0_to_100: float) -> float:
    """Normalize a 0-100 score into a strict 0.0-1.0 score."""
    return _clamp(float(value_0_to_100) / 100.0, 0.0, 1.0)


@dataclass
class EpisodeSummary:
    seed: int
    steps: int
    total_reward: float
    reward_per_step: float
    final_waiting: int
    final_admitted: int
    final_critical_waiting: int
    terminated: bool
    truncated: bool


@dataclass
class GraderReport:
    task: str
    episodes: int
    policy: str
    seed_start: int
    timestamp_utc: str
    aggregate: Dict[str, Any]
    scoring: Dict[str, float]
    rubric: Dict[str, Any]
    llm_scoring: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]
    pass_threshold: float
    passed: bool
    episode_summaries: List[EpisodeSummary]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["episode_summaries"] = [asdict(e) for e in self.episode_summaries]
        return payload


@dataclass
class PPORuntime:
    model: Any
    device: Any


def _load_ppo_runtime(model_path: str) -> PPORuntime:
    import torch

    from smart_hospital_orchestration.training.ppo_trainer import ActorCritic

    ckpt_path = Path(model_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PPO checkpoint not found: {model_path}")

    raw = torch.load(ckpt_path, map_location="cpu")
    obs_dim = int(raw.get("obs_dim", 0))
    action_dim = int(raw.get("action_dim", 0))
    if obs_dim <= 0 or action_dim <= 0:
        raise ValueError("Invalid PPO checkpoint metadata: obs_dim/action_dim missing")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim).to(device)
    model.load_state_dict(raw["model_state_dict"])
    model.eval()
    return PPORuntime(model=model, device=device)


def _pick_action(
    policy: str,
    env: HospitalEnv,
    state: Dict[str, Any],
    agent: Optional[HospitalBaselineAgent],
    ppo_runtime: Optional[PPORuntime],
) -> int:
    if policy == "random":
        return int(env.action_space.sample())
    if policy == "ppo":
        if ppo_runtime is None:
            raise ValueError("PPO runtime is required for policy='ppo'")
        import torch

        obs = np.asarray(state.get("flat", []), dtype=np.float32)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=ppo_runtime.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = ppo_runtime.model(obs_t)
            action = torch.argmax(logits, dim=-1)
        return int(action.item())
    if agent is None:
        agent = HospitalBaselineAgent(verbose=False)
    return int(agent.select_action(state))


def _run_episode(task: str, seed: int, policy: str, ppo_runtime: Optional[PPORuntime] = None) -> EpisodeSummary:
    env = HospitalEnv(task=task)
    agent = HospitalBaselineAgent(verbose=False) if policy == "heuristic" else None

    state, _ = env.reset(seed=seed)
    done = False
    step_count = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not done:
        action = _pick_action(policy=policy, env=env, state=state, agent=agent, ppo_runtime=ppo_runtime)
        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))

    readable = env.state().get("readable", {})
    env.close()

    reward_per_step = total_reward / max(step_count, 1)
    return EpisodeSummary(
        seed=seed,
        steps=step_count,
        total_reward=round(total_reward, 6),
        reward_per_step=round(reward_per_step, 6),
        final_waiting=int(readable.get("waiting", 0)),
        final_admitted=int(readable.get("admitted", 0)),
        final_critical_waiting=int(readable.get("critical_waiting", 0)),
        terminated=terminated,
        truncated=truncated,
    )


def _summarize_policy_runs(
    task: str,
    seed_start: int,
    episodes: int,
    policy: str,
    ppo_runtime: Optional[PPORuntime] = None,
) -> Dict[str, Any]:
    """Run a policy over fixed seeds and return a compact benchmark summary."""
    runs = [
        _run_episode(task=task, seed=seed_start + i, policy=policy, ppo_runtime=ppo_runtime)
        for i in range(episodes)
    ]

    reward_per_step_arr = np.array([e.reward_per_step for e in runs], dtype=np.float64)
    waiting_arr = np.array([e.final_waiting for e in runs], dtype=np.float64)
    admitted_arr = np.array([e.final_admitted for e in runs], dtype=np.float64)
    critical_arr = np.array([e.final_critical_waiting for e in runs], dtype=np.float64)
    completion_arr = np.array([1.0 if e.terminated or e.truncated else 0.0 for e in runs], dtype=np.float64)

    reward_per_step = float(np.mean(reward_per_step_arr))
    final_score = _clamp(((reward_per_step + 20.0) / 40.0) * 100.0, 0.0, 100.0)
    final_score_0_to_1 = _normalize_percent_score(final_score)

    return {
        "policy": policy,
        "episodes": episodes,
        "summary": {
            "avg_reward_per_step": round(reward_per_step, 6),
            "avg_final_waiting": round(float(np.mean(waiting_arr)), 6),
            "avg_final_admitted": round(float(np.mean(admitted_arr)), 6),
            "avg_final_critical_waiting": round(float(np.mean(critical_arr)), 6),
            "completion_rate": round(float(np.mean(completion_arr)), 6),
            "final_score": round(final_score, 3),
            "final_score_0_to_1": round(final_score_0_to_1, 6),
        },
        "episode_summaries": [asdict(e) for e in runs],
    }


def _task_defaults(task: str) -> Dict[str, Any]:
    task = task.lower()
    defaults = {
        "easy": {"pass_threshold": 65.0, "creativity_base": 75.0},
        "medium": {"pass_threshold": 60.0, "creativity_base": 85.0},
        "hard": {"pass_threshold": 55.0, "creativity_base": 92.0},
    }
    return defaults.get(task, defaults["medium"])


def _get_rubric_weights(profile: str) -> Dict[str, float]:
    # Mirrors common hackathon judging factors.
    if profile == "hackathon_v1":
        return {
            "environment_design": 0.24,
            "reward_logic_quality": 0.24,
            "problem_creativity": 0.14,
            "code_quality_proxy": 0.14,
            "ai_learning_capability": 0.24,
        }
    if profile == "balanced":
        return {
            "environment_design": 0.20,
            "reward_logic_quality": 0.20,
            "problem_creativity": 0.20,
            "code_quality_proxy": 0.20,
            "ai_learning_capability": 0.20,
        }
    raise ValueError("Unknown rubric profile. Use one of: hackathon_v1, balanced")


def _compute_llm_scoring(
    *,
    enabled: bool,
    prompt_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "provider": None,
            "used": False,
            "score": None,
            "note": "LLM scoring disabled",
        }

    endpoint = os.getenv("LLM_GRADER_ENDPOINT", "").strip()
    api_key = os.getenv("LLM_GRADER_API_KEY", "").strip()
    model = os.getenv("LLM_GRADER_MODEL", "gpt-4.1-mini")

    if not endpoint:
        # Safe local fallback so hook remains optional and non-blocking.
        fallback_score = 70.0
        return {
            "enabled": True,
            "provider": "local_fallback",
            "used": False,
            "score": fallback_score,
            "note": "Set LLM_GRADER_ENDPOINT to enable live LLM scoring",
        }

    req_body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict RL environment judge. Return ONLY JSON with keys: "
                    "llm_score (0-100), strengths (array), risks (array), verdict (string)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload, separators=(",", ":")),
            },
        ],
        "temperature": 0.0,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        http_req = urlrequest.Request(
            endpoint,
            data=json.dumps(req_body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlrequest.urlopen(http_req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        parsed = json.loads(raw)
        # OpenAI-compatible parser path.
        content = (
            parsed.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "{}")
        )
        content_json = json.loads(content) if isinstance(content, str) else content
        llm_score = float(content_json.get("llm_score", 0.0))
        return {
            "enabled": True,
            "provider": "remote",
            "used": True,
            "score": _clamp(llm_score, 0.0, 100.0),
            "details": content_json,
            "note": "LLM scoring computed from external endpoint",
        }
    except (URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return {
            "enabled": True,
            "provider": "remote",
            "used": False,
            "score": 70.0,
            "note": f"LLM endpoint failed, fallback score used: {exc}",
        }


def write_grader_report(
    report: GraderReport,
    *,
    output_path: str = "",
    save_history: bool = True,
    history_dir: str = "results/grader_history",
) -> Dict[str, str]:
    """Persist report to output file and timestamped history files."""
    payload = report.to_dict()
    writes: Dict[str, str] = {}

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        writes["output"] = str(out_path)

    if save_history:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        hist_dir = Path(history_dir)
        hist_dir.mkdir(parents=True, exist_ok=True)
        hist_file = hist_dir / f"grader_{report.task}_{ts}.json"
        hist_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        writes["history"] = str(hist_file)

        latest_file = hist_dir / "latest_grader_result.json"
        latest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        writes["latest"] = str(latest_file)

    return writes


def grade_environment(
    task: str = "medium",
    episodes: int = 5,
    seed_start: int = 42,
    policy: str = "heuristic",
    model_path: str = "",
    pass_threshold: Optional[float] = None,
    rubric_profile: str = "hackathon_v1",
    enable_llm_score: bool = False,
) -> GraderReport:
    """Run submission-style grading and return a structured report."""
    if policy not in {"heuristic", "random", "ppo"}:
        raise ValueError("policy must be one of: heuristic, random, ppo")
    if episodes < 1:
        raise ValueError("episodes must be >= 1")

    ppo_runtime: Optional[PPORuntime] = None
    if policy == "ppo":
        if not model_path:
            raise ValueError("model_path is required when policy='ppo'")
        ppo_runtime = _load_ppo_runtime(model_path)

    summaries: List[EpisodeSummary] = []
    for i in range(episodes):
        summaries.append(_run_episode(task=task, seed=seed_start + i, policy=policy, ppo_runtime=ppo_runtime))

    reward_per_step_arr = np.array([e.reward_per_step for e in summaries], dtype=np.float64)
    final_waiting_arr = np.array([e.final_waiting for e in summaries], dtype=np.float64)
    final_admitted_arr = np.array([e.final_admitted for e in summaries], dtype=np.float64)
    final_critical_arr = np.array([e.final_critical_waiting for e in summaries], dtype=np.float64)
    terminated_arr = np.array([1.0 if e.terminated or e.truncated else 0.0 for e in summaries], dtype=np.float64)

    avg_reward_per_step = float(np.mean(reward_per_step_arr))
    avg_waiting = float(np.mean(final_waiting_arr))
    avg_admitted = float(np.mean(final_admitted_arr))
    avg_critical_waiting = float(np.mean(final_critical_arr))
    completion_rate = float(np.mean(terminated_arr))

    defaults = _task_defaults(task)
    threshold = float(defaults["pass_threshold"] if pass_threshold is None else pass_threshold)

    # Base metric scores (0-100 each)
    reward_score = _clamp(((avg_reward_per_step + 20.0) / 40.0) * 100.0, 0.0, 100.0)
    throughput_denom = max(avg_admitted + avg_waiting, 1e-6)
    throughput_score = _clamp((avg_admitted / throughput_denom) * 100.0, 0.0, 100.0)
    safety_score = _clamp((1.0 - (avg_critical_waiting / max(avg_waiting + avg_admitted, 1.0))) * 100.0, 0.0, 100.0)
    stability_score = _clamp(completion_rate * 100.0, 0.0, 100.0)

    # Hackathon rubric-aligned factor scores.
    environment_design_score = _clamp((0.55 * stability_score) + (0.45 * safety_score), 0.0, 100.0)
    reward_logic_quality_score = reward_score
    problem_creativity_score = _clamp(float(defaults["creativity_base"]), 0.0, 100.0)
    code_quality_proxy_score = _clamp((0.7 * stability_score) + (0.3 * throughput_score), 0.0, 100.0)
    ai_learning_capability_score = _clamp((0.6 * reward_score) + (0.4 * throughput_score), 0.0, 100.0)

    rubric_weights = _get_rubric_weights(rubric_profile)
    final_score = (
        (rubric_weights["environment_design"] * environment_design_score)
        + (rubric_weights["reward_logic_quality"] * reward_logic_quality_score)
        + (rubric_weights["problem_creativity"] * problem_creativity_score)
        + (rubric_weights["code_quality_proxy"] * code_quality_proxy_score)
        + (rubric_weights["ai_learning_capability"] * ai_learning_capability_score)
    )

    aggregate = {
        "avg_reward_per_step": round(avg_reward_per_step, 6),
        "avg_final_waiting": round(avg_waiting, 6),
        "avg_final_admitted": round(avg_admitted, 6),
        "avg_final_critical_waiting": round(avg_critical_waiting, 6),
        "completion_rate": round(completion_rate, 6),
    }

    scoring = {
        "reward_score": round(reward_score, 3),
        "throughput_score": round(throughput_score, 3),
        "safety_score": round(safety_score, 3),
        "stability_score": round(stability_score, 3),
        "environment_design_score": round(environment_design_score, 3),
        "reward_logic_quality_score": round(reward_logic_quality_score, 3),
        "problem_creativity_score": round(problem_creativity_score, 3),
        "code_quality_proxy_score": round(code_quality_proxy_score, 3),
        "ai_learning_capability_score": round(ai_learning_capability_score, 3),
        "final_score": round(final_score, 3),
    }

    # OpenEnv-facing normalized score outputs (strict 0.0-1.0 contract).
    scoring.update(
        {
            "reward_score_0_to_1": round(_normalize_percent_score(reward_score), 6),
            "throughput_score_0_to_1": round(_normalize_percent_score(throughput_score), 6),
            "safety_score_0_to_1": round(_normalize_percent_score(safety_score), 6),
            "stability_score_0_to_1": round(_normalize_percent_score(stability_score), 6),
            "environment_design_score_0_to_1": round(_normalize_percent_score(environment_design_score), 6),
            "reward_logic_quality_score_0_to_1": round(_normalize_percent_score(reward_logic_quality_score), 6),
            "problem_creativity_score_0_to_1": round(_normalize_percent_score(problem_creativity_score), 6),
            "code_quality_proxy_score_0_to_1": round(_normalize_percent_score(code_quality_proxy_score), 6),
            "ai_learning_capability_score_0_to_1": round(_normalize_percent_score(ai_learning_capability_score), 6),
            "final_score_0_to_1": round(_normalize_percent_score(final_score), 6),
        }
    )

    llm_scoring = _compute_llm_scoring(
        enabled=enable_llm_score,
        prompt_payload={
            "task": task,
            "episodes": episodes,
            "policy": policy,
            "aggregate": aggregate,
            "scoring": scoring,
            "rubric_profile": rubric_profile,
        },
    )

    llm_score = llm_scoring.get("score")
    if isinstance(llm_score, (int, float)):
        # Blend deterministic + optional LLM judgement conservatively.
        final_score = (0.85 * final_score) + (0.15 * float(llm_score))
        scoring["final_score"] = round(final_score, 3)
        scoring["final_score_0_to_1"] = round(_normalize_percent_score(final_score), 6)
        scoring["llm_blend_applied"] = 1.0
    else:
        scoring["llm_blend_applied"] = 0.0

    pass_threshold_0_to_1 = _normalize_percent_score(threshold)
    scoring["pass_threshold_0_to_1"] = round(pass_threshold_0_to_1, 6)

    benchmark_policies: List[str] = ["random", "heuristic"]
    if policy == "ppo" and model_path:
        benchmark_policies.append("ppo")

    benchmark_comparison: Dict[str, Any] = {}
    for bench_policy in benchmark_policies:
        bench_runtime = ppo_runtime if bench_policy == "ppo" else None
        benchmark_comparison[bench_policy] = _summarize_policy_runs(
            task=task,
            seed_start=seed_start,
            episodes=episodes,
            policy=bench_policy,
            ppo_runtime=bench_runtime,
        )

    rubric = {
        "profile": rubric_profile,
        "weights": rubric_weights,
        "task_defaults": defaults,
    }

    return GraderReport(
        task=task,
        episodes=episodes,
        policy=policy,
        seed_start=seed_start,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        aggregate=aggregate,
        scoring=scoring,
        rubric=rubric,
        llm_scoring=llm_scoring,
        benchmark_comparison=benchmark_comparison,
        pass_threshold=threshold,
        passed=final_score >= threshold,
        episode_summaries=summaries,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Programmatic grader for Smart Hospital OpenEnv")
    parser.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=["heuristic", "random", "ppo"], default="heuristic")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--pass-threshold", type=float, default=None)
    parser.add_argument("--rubric-profile", choices=["hackathon_v1", "balanced"], default="hackathon_v1")
    parser.add_argument("--enable-llm-score", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--save-history", action="store_true", default=False)
    parser.add_argument("--history-dir", type=str, default="results/grader_history")
    args = parser.parse_args()

    report = grade_environment(
        task=args.task,
        episodes=args.episodes,
        seed_start=args.seed,
        policy=args.policy,
        model_path=args.model_path,
        pass_threshold=args.pass_threshold,
        rubric_profile=args.rubric_profile,
        enable_llm_score=args.enable_llm_score,
    )
    payload = report.to_dict()
    print("GRADER_RESULT=" + json.dumps(payload, separators=(",", ":")))

    writes = write_grader_report(
        report,
        output_path=args.output,
        save_history=args.save_history,
        history_dir=args.history_dir,
    )
    if writes:
        print("GRADER_WRITES=" + json.dumps(writes, separators=(",", ":")))


if __name__ == "__main__":
    _main()
