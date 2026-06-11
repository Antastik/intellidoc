FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; faiss-cpu and pypdf ship wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY frontend ./frontend

EXPOSE 8000
ENV DATA_DIR=/data
VOLUME ["/data"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
