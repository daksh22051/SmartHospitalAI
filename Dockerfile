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

RUN pip install --no-cache-dir -e .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/api/health', timeout=5)" || exit 1

CMD ["python", "web_interface.py"]
