# Use a lightweight Python base image
FROM python:3.12.4-slim AS builder
COPY requirements.txt .

# Install necessary system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN CFLAGS="-march=native -O2" pip install --no-cache-dir -r requirements.txt

FROM python:3.12.4-slim
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

ENV OMP_NUM_THREADS=10
ENV MKL_NUM_THREADS=10
ENV NUMEXPR_MAX_THREADS=10

# Copy source code and application files only in final steps to trigger rebuilds only if source changes
COPY src/ /app/src/
COPY data/ /app/data/

# Set working directory and expose volume for output
WORKDIR /app
RUN mkdir -p /app/output
VOLUME /app/output

# Define the entrypoint for easier parameter passing
ENTRYPOINT ["python", "src/simulation.py"]