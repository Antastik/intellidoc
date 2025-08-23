# Use Python slim image for smaller size
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-railway.txt && \
    pip cache purge

# Copy application code (only necessary files)
COPY src/ ./src/
COPY config/ ./config/
COPY intellidoc.py ./
COPY start_api.py ./

# Create necessary directories
RUN mkdir -p data/uploads data/output data/input models temp && \
    chmod 755 data/uploads data/output data/input models temp

# Create non-root user
RUN useradd --create-home --shell /bin/bash intellidoc && \
    chown -R intellidoc:intellidoc /app
USER intellidoc

# Expose port (Railway will override this)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Railway-compatible command
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
