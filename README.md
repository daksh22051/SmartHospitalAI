---
title: SmartHospitalAI
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# SmartHospitalAI

This repository deploys a Docker-based Hugging Face Space for Smart Hospital Resource Orchestration.

## Deployment

- `Dockerfile` at repository root
- `inference.py` entry point
- `app.py` exposes the FastAPI server on port `7860`
- `openenv.yaml` provides OpenEnv validation metadata

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /run_inference`

## Build and Run

```bash
docker build -t smart-hospital-ai .
docker run --rm -p 7860:7860 smart-hospital-ai
```
