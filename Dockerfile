# FinSolve RAG chatbot -- self-contained image.
# The vector store (chroma_db/) and the embedding-model weights are baked in
# at build time by running ingest.py, so the container needs no runtime
# downloads and starts with everything ready. Only GROQ_API_KEY is injected
# at runtime (as an Azure Container Apps secret).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /app

# Install the CPU-only build of torch first. The default torch wheel bundles
# CUDA and is multiple GB; the CPU wheel is a fraction of the size and is all
# we need (no GPU in Container Apps). sentence-transformers then reuses it.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# App source (data/ included) -- see .dockerignore for what's excluded.
COPY . .

# Build the vector store AND warm the HuggingFace model cache into the image.
RUN python ingest.py

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
