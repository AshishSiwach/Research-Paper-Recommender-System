# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch to keep image size small (~2GB vs ~8GB for CUDA)
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /install /usr/local

# Application code
COPY app.py .

# Data directory — mount your arxiv_embeddings.pkl here at runtime
RUN mkdir -p data

# Non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV EMBEDDING_MODEL=all-MiniLM-L6-v2
ENV DATA_PATH=data/arxiv_embeddings.pkl
ENV TOP_K_MAX=20

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
