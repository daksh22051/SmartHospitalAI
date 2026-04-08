# Hugging Face Proof Capture Template

Use this template while collecting final proof evidence.

## 1. Space Runtime Screenshot

- URL: https://huggingface.co/spaces/daksh2205/SmartHospitalAI
- Capture: Space header showing `Running`
- File name suggestion: `01_space_running.png`

## 2. Build Log Screenshot

- Open Space -> Logs -> Build
- Capture successful image build and start lines
- File name suggestion: `02_build_logs_success.png`

## 3. Container Log Proof Screenshot

- Trigger proof logs by opening this URL once:
  - `https://huggingface.co/spaces/daksh2205/SmartHospitalAI/run_inference`
- Open Space -> Logs -> Container
- Capture lines containing:
  - `[START]`
  - `[STEP]`
  - `[END]`
  - `INFERENCE_RESULT=`
- File name suggestion: `03_container_inference_logs.png`

## 4. Artifact Checklist

Attach these files with screenshots:

- `results/constraint_proof_benchmark.json`
- `results/constraint_proof_docker.json`
- `results/live_llm_baseline_proof.json`
- `results/reward_ablation_report.json`
- `results/stress_test_report.json`
- `results/benchmark_dashboard.html`

## 5. Submission Note (paste-ready)

The SmartHospitalAI Space is deployed and running. Runtime logs include the required inference contract tags (`[START]`, `[STEP]`, `[END]`, `INFERENCE_RESULT=`). Constraint proofs and live baseline proof artifacts are attached.

## 6. Screenshot Captions (exact text)

- `01_space_running.png`: "Space runtime status is Running on Hugging Face Spaces."
- `02_build_logs_success.png`: "Docker build and startup logs show successful deployment."
- `03_container_inference_logs.png`: "Container logs include [START], [STEP], [END], and INFERENCE_RESULT as required by the baseline inference contract."
