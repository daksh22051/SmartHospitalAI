# API Key Rotation Guide

Rotate all keys that were used during debugging/deployment.

## Keys to rotate

- OpenAI key (`OPENAI_API_KEY`)
- Groq key (`GROQ_API_KEY`)
- Hugging Face token (`HF_TOKEN`)
- Weather provider key (`OPENWEATHER_API_KEY`) if used
- Grader endpoint key (`LLM_GRADER_API_KEY`) if used

## Steps

1. Revoke old keys in provider dashboards.
2. Generate new keys.
3. Update Hugging Face Space Secrets.
4. Update local `.env`.
5. Restart Space and confirm healthy startup.

## Validation

- `/api/health` returns healthy
- `Run Proof Logs` still emits inference tags
- No old key remains in code, docs, or commit history notes
