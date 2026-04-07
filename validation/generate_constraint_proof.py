"""Generate a constraint-proof benchmark artifact for submission.

This script records:
- elapsed runtime for one or more inference runs
- observed machine resources (CPU count and RAM)
- pass/fail against target runtime limit (< 20 minutes)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _bytes_to_gb(num_bytes: int) -> float:
    return round(float(num_bytes) / (1024**3), 3)


def _get_total_ram_bytes() -> int:
    # Windows path (this workspace runs on Windows).
    if os.name == "nt":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)

    # POSIX fallback.
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))

    return 0


@dataclass
class RunResult:
    task: str
    seed: int
    elapsed_seconds: float
    exit_code: int
    inference_result: dict[str, Any] | None


def _run_inference(
    python_exe: str,
    inference_path: Path,
    task: str,
    seed: int,
    disable_llm: bool,
) -> RunResult:
    cmd = [python_exe, str(inference_path), "--task", task, "--seed", str(seed)]
    child_env = os.environ.copy()
    if disable_llm:
        # Keep runtime benchmark independent from network/API quota variance.
        child_env["API_BASE_URL"] = ""
        child_env["MODEL_NAME"] = ""
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=child_env)
    elapsed = time.perf_counter() - started

    parsed_result = None
    for line in proc.stdout.splitlines():
        if line.startswith("INFERENCE_RESULT="):
            try:
                parsed_result = json.loads(line.split("=", 1)[1])
            except json.JSONDecodeError:
                parsed_result = None
            break

    return RunResult(
        task=task,
        seed=seed,
        elapsed_seconds=round(elapsed, 3),
        exit_code=int(proc.returncode),
        inference_result=parsed_result,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate runtime/resource constraint-proof artifact")
    parser.add_argument("--tasks", nargs="+", default=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime-limit-seconds", type=int, default=1200)
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable live LLM calls during benchmark runs (default is disabled for stable timing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("results") / "constraint_proof_benchmark.json"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    inference_path = project_root / "inference.py"
    python_exe = sys.executable

    all_results: list[RunResult] = []
    suite_started = time.perf_counter()
    for task in args.tasks:
        all_results.append(
            _run_inference(
                python_exe,
                inference_path,
                task=task,
                seed=args.seed,
                disable_llm=not args.enable_llm,
            )
        )
    suite_elapsed = round(time.perf_counter() - suite_started, 3)

    cpu_count = int(os.cpu_count() or 0)
    total_ram_bytes = _get_total_ram_bytes()
    total_ram_gb = _bytes_to_gb(total_ram_bytes) if total_ram_bytes > 0 else None

    runtime_pass = suite_elapsed < float(args.runtime_limit_seconds)
    all_exit_ok = all(item.exit_code == 0 for item in all_results)

    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": cpu_count,
            "total_ram_gb": total_ram_gb,
        },
        "constraints_target": {
            "runtime_seconds_lt": args.runtime_limit_seconds,
            "cpu": 2,
            "memory_gb": 8,
        },
        "benchmark_suite": {
            "tasks": args.tasks,
            "seed": args.seed,
            "llm_enabled": bool(args.enable_llm),
            "total_elapsed_seconds": suite_elapsed,
            "runs": [asdict(item) for item in all_results],
        },
        "checks": {
            "runtime_within_limit": runtime_pass,
            "all_runs_succeeded": all_exit_ok,
        },
        "notes": [
            "Runtime is measured directly on this host.",
            "By default, live LLM calls are disabled during benchmark generation to avoid network/quota noise.",
            "CPU/RAM values are host-observed resources; enforce exact 2 CPU / 8GB at runtime via container limits when needed.",
            "Suggested container enforcement: docker run --cpus=2 --memory=8g --rm smart-hospital-orchestration",
        ],
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    print(f"Wrote constraint proof: {out_path}")
    print(json.dumps(artifact["checks"], separators=(",", ":")))


if __name__ == "__main__":
    main()
