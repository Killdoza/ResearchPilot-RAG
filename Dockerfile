# Resume-ready demo image. Runtime data and .env are mounted/injected at start;
# neither papers nor secrets are copied into the image.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/app/data/runtime/model_cache

WORKDIR /app

# Needed by PDF/document parsing and sentence-transformers on Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt docker/constraints.cpu.txt ./
# sentence-transformers only needs CPU inference in the v1.0 demo. The
# constraint must apply to the complete resolution, otherwise a later
# requirements install can replace CPU PyTorch with a CUDA build.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -c constraints.cpu.txt -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY .env.example README.md ./

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/runtime \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "researchpilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
