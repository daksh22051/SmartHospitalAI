from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from smart_hospital_orchestration.environment.hospital_env import HospitalEnv

app = FastAPI(title="Smart Hospital OpenEnv API")

BASE_DIR = Path(__file__).resolve().parent

# Mount static assets for UI (CSS/JS/images)
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Templates for lightweight UI pages under FastAPI (mirrors Flask templates)
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Root redirect so Space main URL shows dashboard instead of 404
@app.get("/", response_class=HTMLResponse)
def root(_: Request) -> RedirectResponse:
    return RedirectResponse(url="/controls", status_code=307)

# Minimal UI routes to render existing templates directly if FastAPI is serving
@app.get("/controls", response_class=HTMLResponse)
def controls_page_fastapi(request: Request):
    return templates.TemplateResponse("controls.html", {"request": request})

@app.get("/analytics", response_class=HTMLResponse)
def analytics_page_fastapi(request: Request):
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/performance", response_class=HTMLResponse)
def performance_page_fastapi(request: Request):
    return templates.TemplateResponse("performance.html", {"request": request})


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    task: Optional[str] = "medium"


class StepRequest(BaseModel):
    action: int


ENV: Optional[HospitalEnv] = None


def _serialize_payload(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _serialize_payload(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_payload(item) for item in value]
    return value


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
        content={"ok": True, "state": _serialize_payload(state), "info": _serialize_payload(info)},
    )


@app.post("/step")
def step(request: StepRequest) -> JSONResponse:
    env = _get_env()
    state, reward, done, info = env.step(int(request.action))
    return JSONResponse(
        status_code=200,
        content={
            "ok": True,
            "state": _serialize_payload(state),
            "reward": float(reward),
            "done": bool(done),
            "info": _serialize_payload(info),
        },
    )


@app.get("/state")
def state() -> JSONResponse:
    env = _get_env()
    return JSONResponse(
        status_code=200,
        content={"ok": True, "state": _serialize_payload(env.state())},
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
