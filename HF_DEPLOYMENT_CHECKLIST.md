# Hugging Face Space Deployment Checklist

## 1. Required Files Present

- `Dockerfile`
- `openenv.yaml`
- `README.md`
- `inference.py`
- `smart_hospital_orchestration/` package code

## 2. Environment Variables

Set these in Hugging Face Space Secrets:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- Optional: `LLM_GRADER_ENDPOINT`, `LLM_GRADER_API_KEY`, `LLM_GRADER_MODEL`

## 3. Local Verification

Run from repository root:

```bash
python smart_hospital_orchestration/inference.py --task medium --seed 42 --max-steps 5
```

Expected log structure:

- `[START] ...`
- `[STEP] ...`
- `[END] ...`
- `INFERENCE_RESULT=...`

## 4. Docker Verification

```bash
docker build -t smart-hospital-orchestration smart_hospital_orchestration
docker run --rm smart-hospital-orchestration
```

## 5. Space Setup

1. Create new Hugging Face Space.
2. Select **Docker** SDK.
3. Push repository.
4. Add required secrets.
5. Confirm build success and runtime logs include `INFERENCE_RESULT=`.

## 6. Submission Readiness

- [x] Inference output follows required tags
- [x] Score varies across runs/tasks (not constant)
- [x] Easy/Medium/Hard tasks all run
- [x] Grader output generated successfully

## 7. Constraint-Proof Artifact (2 CPU, 8GB, < 20 min)

Generate the benchmark artifact from project root:

```bash
python validation/generate_constraint_proof.py --tasks easy medium hard --seed 42 --runtime-limit-seconds 1200 --output results/constraint_proof_benchmark.json
```

Artifact output:

- `results/constraint_proof_benchmark.json`

Notes:

- Runtime check is validated in the artifact under `checks.runtime_within_limit`.
- For strict runtime resource caps in deployment, run container with limits:
	`docker run --cpus=2 --memory=8g --rm smart-hospital-orchestration`

## 8. Current Submission Status

### Completed

- OpenEnv interface: `reset()`, `step()`, `state()` are implemented.
- Typed models: `Observation`, `Action`, `Reward` are implemented.
- Tasks: `easy`, `medium`, `hard` are available.
- Automatic grader: present and runnable.
- Baseline script: `inference.py` emits `[START]`, `[STEP]`, `[END]`, and `INFERENCE_RESULT=`.
- Docker build/run: validated locally.
- Constraint proof: generated in `results/constraint_proof_benchmark.json`.
- Docker-constrained proof: generated in `results/constraint_proof_docker.json`.
- Live LLM baseline proof: generated in `results/live_llm_baseline_proof.json` with `checks.live_llm_requirement_pass = true`.
- Hugging Face Space deployment: live Space is running at `https://huggingface.co/spaces/daksh2205/SmartHospitalAI`.

### Still Pending for Final Submission Proof

- Capture and attach screenshots/log snippets from the live Space build/runtime page for submission evidence.

### Live LLM Baseline Proof Artifact

Generate proof artifact:

```bash
python validation/generate_live_llm_proof.py --task medium --seed 42 --output results/live_llm_baseline_proof.json
```

Artifact output:

- `results/live_llm_baseline_proof.json`

Pass condition in artifact checks:

- `checks.live_llm_requirement_pass == true`

### Exact Proof Commands

```bash
python inference.py --task medium --seed 42
python validation/generate_constraint_proof.py --tasks easy medium hard --seed 42 --runtime-limit-seconds 1200 --output results/constraint_proof_benchmark.json
python validation/generate_docker_constraint_proof.py --image smart-hospital-orchestration:constraint-proof --task medium --seed 42 --cpus 2 --memory 8g --runtime-limit-seconds 1200 --output results/constraint_proof_docker.json
python validation/generate_live_llm_proof.py --task medium --seed 42 --output results/live_llm_baseline_proof.json
```
