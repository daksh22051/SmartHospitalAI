# Stage 1: Build stage (Dependencies install karne ke liye)
FROM python:3.10-slim AS builder

# Environment variables for better build performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Virtual env create karna security aur size ke liye achha hai
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements install karna
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Stage 2: Runtime stage (Actual execution stage)
FROM python:3.10-slim AS runtime

# Metadata labels
LABEL maintainer="Daksh - Smart Hospital AI"
LABEL version="1.0.0"

# Environment variables setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PORT=7860

# Builder stage se virtual env copy karna
COPY --from=builder /opt/venv /opt/venv

# Working directory set karna
WORKDIR /app

# Non-root user create karna (Hugging Face security ke liye zaroori hai)
RUN useradd --create-home hospital && \
    mkdir -p /app/logs /app/data /app/models && \
    chown -R hospital:hospital /app

# Saari files copy karna
COPY --chown=hospital:hospital . .

# Package ko install karna (taaki imports kaam karein)
RUN pip install -e .

# User switch karna
USER hospital

# Health check (Hugging Face ko batane ke liye ki system ready hai)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import smart_hospital_orchestration; print('Healthy')" || exit 1

# Port expose karna
EXPOSE 7860

# --- FINAL COMMAND ---
# 1. Pehle main.py evaluate chalega taaki simulation result generate ho jaye (Scaler Dashboard fix)
# 2. Phir uvicorn server start hoga (Dashboard UI fix)
CMD ["sh", "-c", "python main.py evaluate --config configs/medium.yaml --episodes 1 && uvicorn app:app --host 0.0.0.0 --port 7860"]