FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=7860

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY smart_hospital_orchestration/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY inference.py /app/inference.py
COPY smart_hospital_orchestration /app/smart_hospital_orchestration
RUN pip install -e ./smart_hospital_orchestration

EXPOSE 7860

CMD ["sh", "-lc", "python inference.py --task medium --seed 42 --max-steps 5 && uvicorn app:app --host 0.0.0.0 --port 7860"]
