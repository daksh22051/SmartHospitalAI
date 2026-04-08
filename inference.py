"""OpenEnv baseline inference runner with typed models and structured logs.

Required log format emitted by this runner:
  [START] {...}
  [STEP]  {...}
  [END]   {...}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np

try:
    from smart_hospital_orchestration.environment import HospitalEnv
    from smart_hospital_orchestration.models import ActionModel, ObservationModel, RewardModel
except ModuleNotFoundError:
    # Allow running as: python inference.py from project root.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from environment import HospitalEnv
    from models import ActionModel, ObservationModel, RewardModel


ACTION_NAMES = {
    0: "WAIT",
    1: "ALLOCATE_RESOURCE",
    2: "ESCALATE_PRIORITY",
    3: "DEFER",
    4: "REASSIGN",
}


def _load_dotenv() -> None:
    """Lightweight .env loader for local runs without extra dependencies."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                value = v.strip()
                # Support quoted values in .env (common when copying API keys).
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Environment loading should not block inference execution.
        return


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
    score_0_to_1: float
    policy_source: str

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
            "score_0_to_1": round(self.score_0_to_1, 6),
            "policy_source": self.policy_source,
        }


def _log(tag: str, payload: Dict[str, Any]) -> None:
    print(f"[{tag}] {json.dumps(payload, separators=(',', ':'))}")


def _normalize_reward(value: float) -> float:
    # Smoothly map unbounded reward values to [0, 1] while preserving variation.
    return float(0.5 * (np.tanh(float(value) / 40.0) + 1.0))


def _build_observation(task: str, readable: Dict[str, Any]) -> ObservationModel:
    return ObservationModel(
        task=task,
        step=int(readable.get("step", 0)),
        waiting=int(readable.get("waiting", 0)),
        admitted=int(readable.get("admitted", 0)),
        critical_waiting=int(readable.get("critical_waiting", 0)),
        available_doctors=int(readable.get("available_doctors", 0)),
        available_beds=int(readable.get("available_beds", 0)),
        raw=readable,
    )


def _heuristic_action(obs: ObservationModel) -> ActionModel:
    waiting = obs.waiting
    critical_waiting = obs.critical_waiting
    available_doctors = obs.available_doctors
    available_beds = obs.available_beds

    if waiting <= 0:
        return ActionModel(action=0, rationale="No patients waiting")

    if critical_waiting > 0 and available_doctors > 0 and available_beds > 0:
        return ActionModel(action=1, rationale="Critical patients with resources available")

    if critical_waiting > 0 and (available_doctors <= 0 or available_beds <= 0):
        return ActionModel(action=2, rationale="Critical queue with constrained resources")

    if available_doctors > 0 and available_beds > 0:
        return ActionModel(action=1, rationale="Resources available for allocation")

    city_available = int(obs.raw.get("city_available_beds", 0))
    if city_available > 0 and available_beds <= 0:
        return ActionModel(action=4, rationale="Reassign to city network")

    if waiting > 8:
        return ActionModel(action=3, rationale="Queue overloaded; defer low priority")

    return ActionModel(action=0, rationale="Hold state")


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from plain text or fenced code blocks."""
    candidate = text.strip()
    if not candidate:
        return None

    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _build_provider_candidates() -> list[tuple[str, str, str, str]]:
    """Return provider candidates as (name, base_url, model_name, api_key)."""
    api_base = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    hf_base = os.getenv("HF_API_BASE_URL", "https://router.huggingface.co/v1").strip()
    hf_model = os.getenv("HF_MODEL", "").strip() or model_name
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    grok_key = os.getenv("GROK_API_KEY", "").strip()
    grok_base = os.getenv("GROK_API_BASE_URL", "https://api.x.ai/v1").strip()
    grok_model = os.getenv("GROK_MODEL", "grok-beta").strip() or "grok-beta"
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    groq_base = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1").strip()
    groq_model = os.getenv("GROQ_MODEL", "").strip() or model_name

    candidates: list[tuple[str, str, str, str]] = []
    if api_base and model_name:
        endpoint = api_base.lower()
        if "huggingface.co" in endpoint or "hf.space" in endpoint:
            primary_key = hf_token
        elif "api.openai.com" in endpoint:
            primary_key = openai_key
        elif "api.x.ai" in endpoint or "x.ai" in endpoint or "grok" in endpoint:
            primary_key = grok_key
        elif "api.groq.com" in endpoint or "groq" in endpoint:
            primary_key = groq_key
        else:
            primary_key = hf_token or openai_key or grok_key or groq_key
        if primary_key:
            candidates.append(("primary", api_base, model_name, primary_key))

    if hf_token and hf_model:
        candidates.append(("huggingface", hf_base, hf_model, hf_token))
    if grok_key:
        candidates.append(("grok", grok_base, grok_model or model_name, grok_key))
    if groq_key:
        candidates.append(("groq", groq_base, groq_model or model_name, groq_key))

    # Keep only usable entries.
    cleaned: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for name, base_url, model, api_key in candidates:
        key = (base_url, model, api_key)
        if base_url and model and api_key and key not in seen:
            seen.add(key)
            cleaned.append((name, base_url, model, api_key))
    return cleaned


def _http_error_payload(error: urlerror.HTTPError) -> Optional[Dict[str, Any]]:
    try:
        raw = error.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _llm_action(obs: ObservationModel) -> Optional[ActionModel]:
    prompt = {
        "task": obs.task,
        "step": obs.step,
        "waiting": obs.waiting,
        "critical_waiting": obs.critical_waiting,
        "available_doctors": obs.available_doctors,
        "available_beds": obs.available_beds,
        "allowed_actions": ACTION_NAMES,
        "instruction": "Return JSON only: {\"action\": 0-4, \"rationale\": \"...\"}",
    }
    base_req_body = {
        "messages": [
            {
                "role": "system",
                "content": "You are a hospital RL baseline selector. Output strict JSON only.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt, separators=(",", ":")),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 80,
        "response_format": {"type": "json_object"},
    }
    candidates = _build_provider_candidates()
    if not candidates:
        return None

    for _, api_base, model_name, api_key in candidates:
        endpoint = api_base.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "smart-hospital-orchestration/1.0",
        }
        req_body = {"model": model_name, **base_req_body}

        for attempt in range(3):
            try:
                req = urlrequest.Request(
                    endpoint,
                    data=json.dumps(req_body).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urlrequest.urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8")
                parsed = json.loads(raw)
                content = parsed.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                payload = content if isinstance(content, dict) else _extract_json_object(str(content))
                if not payload:
                    break

                action = int(payload.get("action", 0))
                if action < 0 or action > 4:
                    action = max(0, min(4, action))
                return ActionModel(
                    action=action,
                    rationale=str(payload.get("rationale", "llm-selected")),
                )
            except urlerror.HTTPError as e:
                payload = _http_error_payload(e)
                error_field = (payload or {}).get("error", {})
                if isinstance(error_field, dict):
                    error_code = str(error_field.get("code", "")).lower()
                else:
                    error_code = str(error_field).lower()
                payload_text = json.dumps(payload).lower() if payload else ""
                transient = e.code in (429, 500, 502, 503, 504) and attempt < 2
                print(f"API ERROR: {e}")
                # Groq OpenAI compatibility: some models reject response_format=json_object.
                if (
                    e.code == 400
                    and "groq" in api_base.lower()
                    and "response_format" in req_body
                    and ("response_format" in payload_text or "unsupported" in payload_text or "invalid" in payload_text)
                ):
                    req_body.pop("response_format", None)
                    continue
                if transient:
                    if e.code == 429 and error_code in {"insufficient_quota", "rate_limit_exceeded", "too_many_requests"}:
                        break
                    time.sleep(0.7 * (attempt + 1))
                    continue
                break
            except Exception as e:
                print(f"API ERROR: {e}")
                break

    return None


def _visible_patients(state: Dict[str, Any]) -> np.ndarray:
    patients = state.get("patients", np.array([]))
    time_stats = state.get("time", np.array([]))
    total_patients = int(time_stats[2]) if len(time_stats) > 2 else len(patients)
    visible = min(total_patients, len(patients))
    return patients[:visible] if visible > 0 else np.array([])


def _choose_action(task: str, state: Dict[str, Any]) -> tuple[ActionModel, str]:
    readable = state.get("readable", {})
    obs = _build_observation(task, readable)

    llm_pick = _llm_action(obs)
    if llm_pick is not None:
        return llm_pick, "openai"

    return _heuristic_action(obs), "heuristic"


def run_episode(task: str, seed: int, max_steps: Optional[int] = None) -> EpisodeResult:
    """Run one full episode from reset() to done using step(action)."""
    env = HospitalEnv(task=task)
    state, _ = env.reset(seed=seed)

    done = False
    terminated = False
    truncated = False
    step_count = 0
    total_reward = 0.0

    api_base = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    policy_source = "heuristic"

    _log(
        "START",
        {
            "task": task,
            "seed": seed,
            "policy_source": policy_source,
            "api_base_configured": bool(api_base),
            "model_name": model_name or None,
        },
    )

    while not done:
        action_obj, action_source = _choose_action(task=task, state=state)
        action = int(action_obj.action)
        if action_source == "openai":
            policy_source = "openai"

        state, reward, done, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))

        reward_obj = RewardModel(
            value=float(reward),
            normalized_0_to_1=_normalize_reward(float(reward)),
        )

        _log(
            "STEP",
            {
                "step": step_count,
                "action": action,
                "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
                "rationale": action_obj.rationale,
                "reward": reward_obj.model_dump(),
                "done": bool(done),
            },
        )

        if max_steps is not None and step_count >= max_steps:
            break

    final = env.state().get("readable", {})
    env.close()

    avg_reward_per_step = total_reward / max(step_count, 1)
    result = EpisodeResult(
        task=task,
        seed=seed,
        steps=step_count,
        total_reward=total_reward,
        terminated=terminated,
        truncated=truncated,
        final_waiting=int(final.get("waiting", 0)),
        final_admitted=int(final.get("admitted", 0)),
        final_critical_waiting=int(final.get("critical_waiting", 0)),
        score_0_to_1=_normalize_reward(float(avg_reward_per_step)),
        policy_source=policy_source,
    )

    _log("END", result.to_dict())
    print("INFERENCE_RESULT=" + json.dumps(result.to_dict(), separators=(",", ":")))
    return result


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(description="OpenEnv inference runner (frontend-independent)")
    parser.add_argument("--task", type=str, default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    run_episode(task=args.task, seed=args.seed, max_steps=args.max_steps)


if __name__ == "__main__":
    main()