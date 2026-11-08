FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip --default-timeout=300 install --retries=10 --no-cache-dir -r requirements-api.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs/inference \
    && chown -R appuser:appuser /app

COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser config.yaml .
COPY --chown=appuser:appuser deployment/ ./deployment/

USER 10001

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
