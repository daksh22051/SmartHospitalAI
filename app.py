from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title="SmartHospital Inference Runner")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/run_inference")
def run_inference() -> JSONResponse | PlainTextResponse:
    """Run inference.py and mirror stdout/stderr to container logs."""
    repo_root = Path(__file__).resolve().parent
    inference_path = repo_root / "inference.py"

    if not inference_path.exists():
        msg = f"inference.py not found at: {inference_path}"
        print(msg, flush=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": msg})

    cmd = [
        sys.executable,
        str(inference_path),
        "--task",
        os.getenv("INFERENCE_TASK", "medium"),
        "--seed",
        os.getenv("INFERENCE_SEED", "42"),
        "--max-steps",
        os.getenv("INFERENCE_MAX_STEPS", "5"),
    ]

    print(f"[RUN_INFERENCE] Executing: {' '.join(cmd)}", flush=True)

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )

        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.stderr:
            print(completed.stderr, end="", flush=True)

        return JSONResponse(
            status_code=200 if completed.returncode == 0 else 500,
            content={
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
            },
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"Inference timed out after {exc.timeout} seconds"
        print(msg, flush=True)
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", flush=True)
        return JSONResponse(status_code=504, content={"ok": False, "error": msg})
    except Exception as exc:  # pragma: no cover
        msg = f"Inference execution failed: {exc}"
        print(msg, flush=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": msg})
