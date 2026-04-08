from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from smart_hospital_orchestration.environment.hospital_env import HospitalEnv

app = FastAPI(title="Smart Hospital OpenEnv API")


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task: Optional[str] = "medium"


class StepRequest(BaseModel):
    action: int


ENV: Optional[HospitalEnv] = None


def _get_env() -> HospitalEnv:
    if ENV is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return ENV


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(request: ResetRequest) -> JSONResponse:
    global ENV
    task_name = request.task.lower().strip() if request.task else "medium"
    ENV = HospitalEnv(task=task_name)
    state, info = ENV.reset(seed=request.seed)
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"ok": True, "state": state, "info": info}),
    )


@app.post("/step")
def step(request: StepRequest) -> JSONResponse:
    env = _get_env()
    state, reward, done, info = env.step(int(request.action))
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder(
            {
                "ok": True,
                "state": state,
                "reward": reward,
                "done": done,
                "info": info,
            }
        ),
    )


@app.get("/state")
def state() -> JSONResponse:
    env = _get_env()
    return JSONResponse(
        status_code=200,
        content=jsonable_encoder({"ok": True, "state": env.state()}),
    )


@app.get("/run_inference")
def run_inference() -> JSONResponse:
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
            content=jsonable_encoder(
                {
                    "ok": completed.returncode == 0,
                    "returncode": completed.returncode,
                }
            ),
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
