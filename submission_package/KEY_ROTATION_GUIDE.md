# API Key Rotation Guide

Rotate all keys that were used during debugging/deployment.

## Keys to rotate

- OpenAI key (`OPENAI_API_KEY`)
- Groq key (`GROQ_API_KEY`)
- Hugging Face token (`HF_TOKEN`)
- Weather provider key (`OPENWEATHER_API_KEY`) if used
- Grader endpoint key (`LLM_GRADER_API_KEY`) if used

## Steps

1. Revoke old keys in provider dashboards (OpenAI, Groq/xAI, Hugging Face, weather provider, grader provider).
2. Generate new keys with minimum required scope.
3. Update Hugging Face Space Secrets only (do not commit secrets to git).
4. Update local `.env` for local testing only.
5. Restart Space and confirm healthy startup.
6. Remove any accidental key-like values from docs/examples before final upload.

## Hugging Face Secrets To Update

- `API_BASE_URL`
- `MODEL_NAME`
- `OPENAI_API_KEY` or provider-specific key (`GROQ_API_KEY` / `GROK_API_KEY`)
- `HF_TOKEN`
- Optional: `LLM_GRADER_ENDPOINT`, `LLM_GRADER_API_KEY`, `LLM_GRADER_MODEL`, `OPENWEATHER_API_KEY`

## Post-Rotation Validation Commands

Run from project root:

```bash
python inference.py --task medium --seed 42 --max-steps 5
python validation/verify_submission_artifacts.py --report-path results/submission_artifact_verification.json
```

## Validation

- `/api/health` returns healthy
- `/run_inference` still emits inference tags
- No old key remains in code, docs, or commit history notes

## Rotation Completion Note (paste-ready)

"All previously used API keys have been revoked and replaced with new keys. Hugging Face Space Secrets were updated, service restarted, and inference contract logs verified after rotation."
