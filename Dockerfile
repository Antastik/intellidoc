# ============================================
# Build stage
# ============================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies (for compiling packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies globally (not --user)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

# ============================================
# Production stage
# ============================================
FROM python:3.10-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

# Set working dir
WORKDIR /app

# Copy only necessary application files
COPY src/ ./src/
COPY config/ ./config/
COPY requirements.txt ./

# Create runtime dirs and non-root user
RUN mkdir -p data/uploads data/output data/input temp && \
    useradd --create-home --shell /bin/bash intellidoc && \
    chown -R intellidoc:intellidoc /app

USER intellidoc

# Expose port (Railway will use $PORT env)
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run uvicorn directly (no sh -c needed)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
