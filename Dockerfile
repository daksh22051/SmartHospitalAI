FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PORT=7860

RUN python -m venv "$VIRTUAL_ENV"

WORKDIR /app

COPY smart_hospital_orchestration/requirements.txt /app/requirements.txt
RUN "$VIRTUAL_ENV/bin/python" -m pip install --upgrade pip && "$VIRTUAL_ENV/bin/pip" install -r requirements.txt

COPY inference.py /app/inference.py
COPY smart_hospital_orchestration /app/smart_hospital_orchestration
RUN "$VIRTUAL_ENV/bin/pip" install -e ./smart_hospital_orchestration

EXPOSE 7860

# Start the FastAPI app directly (root module path)
CMD ["sh", "-lc", "$VIRTUAL_ENV/bin/uvicorn smart_hospital_orchestration.app:app --host 0.0.0.0 --port 7860"]
