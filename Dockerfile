# Base image
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Builder stage
FROM base as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --user -e ".[classification,mlflow]"

# Development stage
FROM base as development

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY pyproject.toml ./
RUN pip install --user -e ".[dev]"

COPY --chown=appuser:appgroup . .
USER appuser

CMD ["python", "-m", "pytest", "tests/", "-v"]

# Production stage
FROM base as production

COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup config.yaml ./

RUN mkdir -p data/raw data/processed models reports/logs && \
    chown -R appuser:appgroup data models reports

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Training stage
FROM production as training
CMD ["python", "-m", "src.pipeline"]
