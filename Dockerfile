FROM python:3.11-slim

WORKDIR /app

# Slim deploy deps (core + hosted providers). faiss-cpu and pypdf ship wheels.
# Use requirements.txt instead if you need the local sentence-transformers provider.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app ./app
COPY frontend ./frontend

# DATA_DIR is where the FAISS index + metadata live. On free hosts this is
# ephemeral (resets on redeploy/sleep), which is fine for a demo.
ENV DATA_DIR=/data
VOLUME ["/data"]

# Hosts like Render/Railway inject $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
