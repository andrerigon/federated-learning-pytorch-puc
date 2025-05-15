# ──────────────────────────────────────────────────────────────
# Stage 1 ─ build wheelhouse with native optimisations
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS builder

# deps for numpy / torch wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_NO_CACHE_DIR=1 \
    CFLAGS="-march=native -O2"

RUN pip wheel -r requirements.txt -w /wheels

# ──────────────────────────────────────────────────────────────
# Stage 2 ─ final runtime image
# ──────────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

# basic runtime libs only (no build-toolchain)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# sensible thread caps – honoured by PyTorch & NumPy
ENV OMP_NUM_THREADS=20 \
    MKL_NUM_THREADS=20 \
    NUMEXPR_MAX_THREADS=20 \
    PYTHONDONTWRITEBYTECODE=1

# copy code late so rebuilds when code changes
WORKDIR /app
COPY src/ ./src
RUN mkdir -p /data        

# keep working dir & entrypoint as before
WORKDIR /app
COPY src/ ./src          
RUN mkdir -p /app/runs
ENTRYPOINT ["python", "-u", "src/simulation.py"]