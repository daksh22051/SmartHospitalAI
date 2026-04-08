"""Generate a live LLM baseline proof artifact.

This artifact is intended for submission evidence and captures:
- current provider endpoint/model configuration
- direct provider probe response status
- inference run summary and policy_source
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        value = v.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def _provider_probe(api_base: str, model_name: str, api_key: str) -> dict[str, Any]:
    endpoint = api_base.rstrip("/") + "/chat/completions"
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": '{"action":1,"rationale":"probe"}'},
        ],
        "temperature": 0.0,
        "max_tokens": 20,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = urlrequest.Request(endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")

    try:
        with urlrequest.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": True,
                "status": int(resp.status),
                "response_preview": text[:400],
            }
    except urlerror.HTTPError as e:
        payload = e.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": int(e.code),
            "response_preview": payload[:500],
        }
    except Exception as e:  # pragma: no cover
        return {
            "ok": False,
            "status": None,
            "response_preview": f"{type(e).__name__}: {str(e)[:300]}",
        }


def _run_inference(project_root: Path, task: str, seed: int) -> dict[str, Any]:
    cmd = [sys.executable, str(project_root / "inference.py"), "--task", task, "--seed", str(seed)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(project_root), capture_output=True, text=True)
    elapsed = round(time.perf_counter() - started, 3)

    parsed = None
    for line in proc.stdout.splitlines():
        if line.startswith("INFERENCE_RESULT="):
            try:
                parsed = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                parsed = None
            break

    return {
        "exit_code": int(proc.returncode),
        "elapsed_seconds": elapsed,
        "inference_result": parsed,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate live LLM baseline proof artifact")
    parser.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/live_llm_baseline_proof.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    _load_dotenv(project_root / ".env")

    api_base = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    # Match runtime precedence: HF token first, then provider-specific keys.
    api_key = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("GROQ_API_KEY", "").strip()
    )

    probe = _provider_probe(api_base, model_name, api_key) if (api_base and model_name and api_key) else {
        "ok": False,
        "status": None,
        "response_preview": "Missing API_BASE_URL, MODEL_NAME, or provider key",
    }
    run = _run_inference(project_root, task=args.task, seed=args.seed)

    result = run.get("inference_result") or {}
    policy_source = str(result.get("policy_source", ""))

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": {
            "api_base_url": api_base,
            "model_name": model_name,
            "key_present": bool(api_key),
            "probe": probe,
        },
        "inference": run,
        "checks": {
            "probe_success": bool(probe.get("ok", False)),
            "inference_succeeded": run.get("exit_code") == 0,
            "policy_source_is_openai": policy_source == "openai",
            "live_llm_requirement_pass": run.get("exit_code") == 0 and policy_source == "openai",
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Wrote live LLM proof: {output_path}")
    print(json.dumps(artifact["checks"], separators=(",", ":")))


if __name__ == "__main__":
    main()
