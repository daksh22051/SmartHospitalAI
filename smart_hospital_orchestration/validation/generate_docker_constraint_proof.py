"""Generate strict Docker-constrained runtime proof artifact.

This script builds the project image and runs inference inside a container with:
- --cpus=2
- --memory=8g

It writes a JSON artifact suitable for submission evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = round(time.perf_counter() - started, 3)
    return int(proc.returncode), proc.stdout or "", proc.stderr or "", elapsed


def _parse_inference_result(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if line.startswith("INFERENCE_RESULT="):
            try:
                return json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Docker constrained runtime proof")
    parser.add_argument("--image", default="smart-hospital-orchestration:constraint-proof")
    parser.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpus", default="2")
    parser.add_argument("--memory", default="8g")
    parser.add_argument("--runtime-limit-seconds", type=int, default=1200)
    parser.add_argument("--output", default="results/constraint_proof_docker.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    # Build from project_root (which points to smart_hospital_orchestration/),
    # using its Dockerfile and local files as context so COPY paths resolve.
    build_cmd = [
        "docker",
        "build",
        "-t",
        args.image,
        ".",
    ]
    build_code, build_out, build_err, build_elapsed = _run_cmd(build_cmd, cwd=project_root)

    run_stdout = ""
    run_stderr = ""
    run_elapsed = 0.0
    run_code = -1
    inference_result = None

    if build_code == 0:
        run_cmd = [
            "docker",
            "run",
            "--rm",
            "--cpus",
            args.cpus,
            "--memory",
            args.memory,
            args.image,
            "python",
            "inference.py",
            "--task",
            args.task,
            "--seed",
            str(args.seed),
        ]
        run_code, run_stdout, run_stderr, run_elapsed = _run_cmd(run_cmd, cwd=project_root)
        inference_result = _parse_inference_result(run_stdout)

    total_elapsed = round(build_elapsed + run_elapsed, 3)
    runtime_within_limit = total_elapsed < float(args.runtime_limit_seconds)

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "constraints": {
            "cpu": args.cpus,
            "memory": args.memory,
            "runtime_seconds_lt": args.runtime_limit_seconds,
        },
        "docker": {
            "image": args.image,
            "build": {
                "exit_code": build_code,
                "elapsed_seconds": build_elapsed,
                "stdout_tail": "\n".join(build_out.splitlines()[-20:]),
                "stderr_tail": "\n".join(build_err.splitlines()[-20:]),
            },
            "run": {
                "exit_code": run_code,
                "elapsed_seconds": run_elapsed,
                "task": args.task,
                "seed": args.seed,
                "stdout_tail": "\n".join(run_stdout.splitlines()[-40:]),
                "stderr_tail": "\n".join(run_stderr.splitlines()[-20:]),
                "inference_result": inference_result,
            },
        },
        "checks": {
            "build_succeeded": build_code == 0,
            "run_succeeded": run_code == 0,
            "inference_result_present": inference_result is not None,
            "runtime_within_limit": runtime_within_limit,
            "total_elapsed_seconds": total_elapsed,
        },
    }

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Wrote docker constraint proof: {output_path}")
    print(json.dumps(artifact["checks"], separators=(",", ":")))


if __name__ == "__main__":
    main()
