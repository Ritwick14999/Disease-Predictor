# Multi-stage build: the wheel is built once, the runtime image stays lean and
# never carries a compiler or the test suite.
FROM python:3.11-slim AS builder

WORKDIR /build
RUN pip install --no-cache-dir --upgrade pip build

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --outdir /dist


FROM python:3.11-slim AS runtime

# libgomp is the OpenMP runtime XGBoost links against; the slim image omits it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Never run the service as root.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl "disease-predictor[xgboost,imbalance,api]" \
    && rm -rf /tmp/*.whl

# Bake a demo model so the image is useful before any volume is mounted. Mount a
# real bundle over /app/artifacts to serve a model trained on the real dataset.
RUN mkdir -p /app/artifacts /app/reports \
    && dp train --synthetic --model rules --no-benchmark --no-robustness \
        --artifact-dir /app/artifacts --report-dir /app/reports \
    && chown -R appuser:appuser /app

USER appuser
ENV DISEASE_MODEL_PATH=/app/artifacts/disease_predictor_bundle.joblib \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "disease_predictor.api:app", "--host", "0.0.0.0", "--port", "8000"]
