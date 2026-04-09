# Smart Hospital Resource Orchestration - Docker Configuration
# Hugging Face Spaces optimized deployment

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=7860

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 7860

CMD ["python", "-m", "smart_hospital_orchestration.web_interface"]
