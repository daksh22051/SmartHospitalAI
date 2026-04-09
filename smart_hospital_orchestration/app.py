from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, pass_context
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

print("[STARTUP] Loading HospitalEnv...", flush=True)

try:
    from smart_hospital_orchestration.environment.hospital_env import HospitalEnv
    print("[STARTUP] HospitalEnv imported successfully", flush=True)
except ImportError as e:
    print(f"[ERROR] Failed to import HospitalEnv: {e}", flush=True)
    HospitalEnv = None

app = FastAPI(title="Smart Hospital OpenEnv API")

BASE_DIR = Path(__file__).resolve().parent

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Use a custom Jinja2 Environment with caching disabled to avoid hash errors
# observed on the HF runtime ("TypeError: unhashable type: 'dict'") when
# Jinja2 tries to create a cache key that includes a dict.
_jinja_env = Environment(
    loader=FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=True,
    cache_size=0,  # disable template caching
)
# Note: We render templates directly with Jinja2 below to avoid version
# mismatch issues in Starlette's Jinja2Templates on HF runtime.


def _url_for(request: Request, name: str, **params: Any) -> str:
    """Jinja-compatible url_for that works with FastAPI/Starlette.

    - Accepts Flask-style `filename=` and maps it to Starlette's `path=`
      for the StaticFiles mount named "static".
    - Falls back gracefully if a route isn't found.
    """
    try:
        if "filename" in params and "path" not in params:
            params["path"] = params.pop("filename")
        return request.app.url_path_for(name, **params)
    except Exception:
        # Best-effort fallback (prevents template crashes)
        # Return mounted path for static assets if possible
        if name == "static":
            p = params.get("path") or params.get("filename") or ""
            return f"/static/{p}".rstrip("/")
        return "/"


@pass_context
def jinja_url_for(ctx, name: str, **params: Any) -> str:  # type: ignore[no-redef]
    """Context-aware url_for for Jinja templates.

    Retrieves the FastAPI Request from the Jinja context and delegates to
    the helper above so base.html can call {{ url_for('static', filename='...') }}.
    """
    request = ctx.get("request")
    return _url_for(request, name, **params) if request is not None else "/"

# Expose url_for globally so extended templates (base.html) can access it
_jinja_env.globals["url_for"] = jinja_url_for


@app.get("/controls", response_class=HTMLResponse)
def controls_page_fastapi(request: Request):
    template = _jinja_env.get_template("controls.html")
    return HTMLResponse(template.render(request=request))


@app.get("/analytics", response_class=HTMLResponse)
def analytics_page_fastapi(request: Request):
    template = _jinja_env.get_template("analytics.html")
    return HTMLResponse(template.render(request=request))


@app.get("/performance", response_class=HTMLResponse)
def performance_page_fastapi(request: Request):
    template = _jinja_env.get_template("performance.html")
    return HTMLResponse(template.render(request=request))


@app.get("/", response_class=HTMLResponse)
def dashboard_page_fastapi(request: Request):
    template = _jinja_env.get_template("dashboard.html")
    return HTMLResponse(template.render(request=request))


@app.get("/benchmark_dashboard", response_class=HTMLResponse)
def benchmark_dashboard_page_fastapi(request: Request):
    template = _jinja_env.get_template("benchmark_dashboard_page.html")
    return HTMLResponse(template.render(request=request))


@app.get("/benchmark_dashboard/raw")
def benchmark_dashboard_raw_fastapi() -> FileResponse:
    dashboard_path = BASE_DIR / "submission_package" / "benchmark_dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Benchmark dashboard not generated yet.")
    return FileResponse(str(dashboard_path), media_type="text/html")


@app.get("/3d_view", response_class=HTMLResponse)
def view_3d_page_fastapi(request: Request):
    template = _jinja_env.get_template("3d_view.html")
    return HTMLResponse(template.render(request=request))


@app.get("/clinical_ops", response_class=HTMLResponse)
def clinical_ops_page_fastapi(request: Request):
    template = _jinja_env.get_template("clinical_ops.html")
    return HTMLResponse(template.render(request=request))


@app.get("/ai_lab", response_class=HTMLResponse)
def ai_lab_page_fastapi(request: Request):
    template = _jinja_env.get_template("ai_lab.html")
    return HTMLResponse(template.render(request=request))


class ResetRequest(BaseModel):
    seed: Optional[int] = None
    options: Optional[Dict[str, Any]] = None


class StepRequest(BaseModel):
    action: int


class EpisodeState:
    def __init__(self):
        self.env = None
        self.task = "medium"
        self.seed = 42
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        self.terminated = False
        self.truncated = False

    def reset(self, task: str = "medium", seed: Optional[int] = None) -> Dict[str, Any]:
        if HospitalEnv is None:
            raise HTTPException(status_code=500, detail="HospitalEnv not available")
        
        if seed is None:
            seed = 42
        
        self.env = HospitalEnv(task=task)
        self.task = task
        self.seed = seed
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        self.terminated = False
        self.truncated = False
        
        api_base = os.getenv("API_BASE_URL", "").strip()
        model_name = os.getenv("MODEL_NAME", "").strip()
        
        print(f"[START] {json.dumps({'task': task, 'seed': seed, 'policy_source': 'heuristic', 'api_base_configured': bool(api_base), 'model_name': model_name or None}, separators=(',', ':'))}", flush=True)
        
        state, info = self.env.reset(seed=seed)
        return state

    def step(self, action: int) -> Dict[str, Any]:
        if self.env is None:
            raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
        
        if self.done:
            raise HTTPException(status_code=400, detail="Episode already finished. Call /reset first.")
        
        action = max(0, min(4, int(action)))
        action_names = {0: "WAIT", 1: "ALLOCATE_RESOURCE", 2: "ESCALATE_PRIORITY", 3: "DEFER", 4: "REASSIGN"}
        
        state, reward, done, info = self.env.step(action)
        
        self.total_reward += float(reward)
        self.step_count += 1
        self.done = bool(done)
        self.terminated = bool(info.get("terminated", False))
        self.truncated = bool(info.get("truncated", False))
        
        reward_normalized = float(0.5 * (np.tanh(float(reward) / 40.0) + 1.0))
        
        print(f"[STEP] {json.dumps({'step': self.step_count, 'action': action, 'action_name': action_names.get(action, 'UNKNOWN'), 'rationale': 'api-call', 'reward': {'value': float(reward), 'normalized_0_to_1': reward_normalized}, 'done': self.done}, separators=(',', ':'))}", flush=True)
        
        return state

    def get_result(self) -> Optional[Dict[str, Any]]:
        if self.env is None:
            return None
        
        final = self.env.state().get("readable", {})
        avg_reward = self.total_reward / max(self.step_count, 1)
        score_normalized = float(0.5 * (np.tanh(float(avg_reward) / 40.0) + 1.0))
        
        result = {
            "task": self.task,
            "seed": self.seed,
            "steps": self.step_count,
            "total_reward": round(self.total_reward, 3),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "final_waiting": int(final.get("waiting", 0)),
            "final_admitted": int(final.get("admitted", 0)),
            "final_critical_waiting": int(final.get("critical_waiting", 0)),
            "score_0_to_1": round(score_normalized, 6),
            "policy_source": "heuristic",
        }
        
        print(f"[END] {json.dumps(result, separators=(',', ':'))}", flush=True)
        print(f"INFERENCE_RESULT={json.dumps(result, separators=(',', ':'))}", flush=True)
        
        if self.env:
            self.env.close()
            self.env = None
        
        return result


episode_state = EpisodeState()


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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = Body(default=None)) -> JSONResponse:
    try:
        seed = request.seed if request and request.seed is not None else None
        task_name = "medium"

        if request and request.options:
            task_name = str(request.options.get("task", task_name)).lower().strip()
            if request.options.get("seed") is not None:
                seed = int(request.options.get("seed"))

        print(f"[DEBUG] /reset called with task={task_name}, seed={seed}, options={request.options if request else None}", flush=True)

        observation = episode_state.reset(task=task_name, seed=seed)
        info = {"task": task_name, "seed": seed}

        return JSONResponse(
            status_code=200,
            content={"observation": _serialize_payload(observation), "info": _serialize_payload(info)},
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] /reset failed: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(request: StepRequest) -> JSONResponse:
    try:
        state = episode_state.step(action=request.action)
        done = episode_state.done
        
        result_data = None
        if done:
            result_data = episode_state.get_result()
        
        return JSONResponse(
            status_code=200,
            content={
                "ok": True,
                "state": _serialize_payload(state),
                "reward": float(episode_state.total_reward) if episode_state.step_count > 0 else 0.0,
                "done": done,
                "info": _serialize_payload(episode_state.env.state()) if episode_state.env else {},
                "result": result_data,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] /step failed: {type(e).__name__}: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state() -> JSONResponse:
    if episode_state.env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    
    return JSONResponse(
        status_code=200,
        content={"ok": True, "state": _serialize_payload(episode_state.env.state())},
    )


@app.get("/run_inference")
def run_inference() -> JSONResponse:
    repo_root = Path(__file__).resolve().parent
    inference_path = repo_root / "inference.py"

    if not inference_path.exists():
        msg = f"inference.py not found at: {inference_path}"
        print(f"[ERROR] {msg}", flush=True)
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
            content=jsonable_encoder({
                "ok": completed.returncode == 0,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-1000:] if completed.stdout else "",
                "stderr": completed.stderr[-1000:] if completed.stderr else "",
            }),
        )
    except subprocess.TimeoutExpired as exc:
        msg = f"Inference timed out after {exc.timeout} seconds"
        print(f"[ERROR] {msg}", flush=True)
        if exc.stdout:
            print(exc.stdout, end="", flush=True)
        if exc.stderr:
            print(exc.stderr, end="", flush=True)
        return JSONResponse(status_code=504, content={"ok": False, "error": msg})
    except Exception as exc:
        msg = f"Inference execution failed: {exc}"
        print(f"[ERROR] {msg}", flush=True)
        return JSONResponse(status_code=500, content={"ok": False, "error": msg})


@app.on_event("startup")
async def startup_event():
    print("[STARTUP] OpenEnv API started successfully", flush=True)
    print("[STARTUP] Available endpoints: /reset, /step, /state, /run_inference, /health", flush=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


@app.on_event("shutdown")
async def shutdown_event():
    print("[SHUTDOWN] OpenEnv API shutting down...", flush=True)
    if episode_state.env:
        episode_state.env.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
