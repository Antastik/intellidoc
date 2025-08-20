# IntelliDoc FastAPI Web Service

A production-ready RESTful API for intelligent document processing with OCR and NLP capabilities.

## 🚀 Quick Start

### 1. Start the API Server

```bash
# Method 1: Using the launcher script (recommended)
python start_api.py

# Method 2: Using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Method 3: Using the main module
cd src/api && python main.py
```

### 2. Access the API

- **API Documentation (Swagger)**: http://localhost:8000/api/v1/docs
- **API Documentation (ReDoc)**: http://localhost:8000/api/v1/redoc
- **Health Check**: http://localhost:8000/health
- **System Status**: http://localhost:8000/api/v1/status

## 📋 API Endpoints

### Documents Processing

#### Upload Single Document

**Synchronous Processing** (immediate response):
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "async_processing=false" \
  -F "save_result=true"
```

**Asynchronous Processing** (background job):
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "async_processing=true" \
  -F "save_result=true"
```

#### Batch Upload

**Process multiple documents**:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.png" \
  -F "files=@doc3.docx" \
  -F "async_processing=true" \
  -F "parallel_processing=true"
```

#### Get Supported Formats

```bash
curl -X GET "http://localhost:8000/api/v1/documents/formats"
```

### Job Management

#### Get Job Status

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}"
```

#### List All Jobs

```bash
# Get all jobs
curl -X GET "http://localhost:8000/api/v1/jobs"

# Filter by status
curl -X GET "http://localhost:8000/api/v1/jobs?status=success&limit=10"

# Pagination
curl -X GET "http://localhost:8000/api/v1/jobs?limit=20&offset=40"
```

#### Download Job Results

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/{job_id}/download" \
  -H "Accept: application/json" \
  -o results.json
```

#### Cancel/Delete Job

```bash
curl -X DELETE "http://localhost:8000/api/v1/jobs/{job_id}"
```

#### Get Job Statistics

```bash
curl -X GET "http://localhost:8000/api/v1/jobs/stats/summary"
```

### System Status

#### Get System Status

```bash
curl -X GET "http://localhost:8000/api/v1/status"
```

#### Health Check

```bash
curl -X GET "http://localhost:8000/api/v1/status/health"
```

#### Component Status

```bash
curl -X GET "http://localhost:8000/api/v1/status/components"
```

#### System Metrics

```bash
curl -X GET "http://localhost:8000/api/v1/status/metrics"
```

## 📊 Response Examples

### Successful Document Upload (Sync)

```json
{
  "message": "Document processed successfully",
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "result": {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "success",
    "combined_text": "INVOICE #1001\nDate: January 15, 2024...",
    "entities": [
      {
        "text": "John Doe",
        "label": "PERSON",
        "start": 45,
        "end": 53,
        "confidence": 0.95
      },
      {
        "text": "$2500.00",
        "label": "MONEY",
        "start": 120,
        "end": 128,
        "confidence": 0.99
      }
    ],
    "processing_time": 2.34,
    "errors": []
  }
}
```

### Asynchronous Job Response

```json
{
  "message": "Document uploaded successfully. Processing started.",
  "job_id": "job_abc12345"
}
```

### Job Status Response

```json
{
  "job_id": "job_abc12345",
  "status": "success",
  "created_at": "2024-01-15T10:30:00Z",
  "started_at": "2024-01-15T10:30:05Z",
  "completed_at": "2024-01-15T10:32:15Z",
  "progress": 100.0,
  "total_files": 1,
  "processed_files": 1,
  "result": {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "success",
    "processing_time": 2.34
  },
  "download_url": "/api/v1/jobs/job_abc12345/download"
}
```

### System Status Response

```json
{
  "service": "IntelliDoc API",
  "version": "1.0.0",
  "status": "healthy",
  "uptime_seconds": 3600.0,
  "components": {
    "document_ingestor": {
      "available": true,
      "supported_formats": ["pdf", "png", "jpg", "docx"]
    },
    "ocr_processor": {
      "available": true,
      "engines": {"tesseract": true, "paddleocr": false}
    },
    "nlp_processor": {
      "available": true,
      "components": {"spacy": false, "transformers": true}
    }
  },
  "queue_status": {
    "queued": 2,
    "running": 1,
    "success": 45,
    "failed": 0
  }
}
```

## 🔧 Processing Modes

### Synchronous Processing

- **Best for**: Small files, quick processing (< 30 seconds)
- **Response**: Immediate results in the response body
- **Use case**: Real-time processing for web interfaces

```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@small_doc.pdf" \
  -F "async_processing=false"
```

### Asynchronous Processing

- **Best for**: Large files, batch processing, long-running tasks
- **Response**: Job ID for tracking progress
- **Use case**: Background processing for high-volume scenarios

```bash
# 1. Start processing
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@large_doc.pdf" \
  -F "async_processing=true"

# 2. Check status
curl -X GET "http://localhost:8000/api/v1/jobs/job_abc12345"

# 3. Download results when complete
curl -X GET "http://localhost:8000/api/v1/jobs/job_abc12345/download" -o results.json
```

## 📝 Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Portable Document Format |
| PNG | `.png` | Portable Network Graphics |
| JPEG | `.jpg`, `.jpeg` | Joint Photographic Experts Group |
| TIFF | `.tiff`, `.tif` | Tagged Image File Format |
| DOCX | `.docx` | Microsoft Word Document |

### File Size Limits

- **Maximum file size**: 50MB per document
- **Maximum batch size**: 20 documents per request
- **Processing timeout**: 300 seconds per document

## 🏷️ Extracted Entities

The NLP processor extracts the following entity types:

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| `PERSON` | Names of people | "John Doe", "Jane Smith" |
| `ORG` | Organizations | "Acme Corp", "Microsoft" |
| `MONEY` | Currency amounts | "$1,000.00", "€500" |
| `DATE` | Dates and times | "January 15, 2024", "2024-01-15" |
| `EMAIL` | Email addresses | "user@example.com" |
| `PHONE` | Phone numbers | "(555) 123-4567" |
| `GPE` | Geographical locations | "New York", "California" |

## 🚨 Error Handling

### Error Response Format

```json
{
  "error": "ERROR_CODE",
  "message": "Human-readable error message",
  "details": {
    "additional": "context",
    "path": "/api/v1/documents/upload",
    "method": "POST"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_FILE` | 400 | Unsupported file format or corrupted file |
| `PAYLOAD_TOO_LARGE` | 413 | File size exceeds 50MB limit |
| `BATCH_TOO_LARGE` | 400 | Batch exceeds 20 files limit |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `VALIDATION_ERROR` | 422 | Request parameters invalid |
| `PROCESSING_ERROR` | 500 | Internal processing failure |

### Example Error Responses

**File too large**:
```json
{
  "error": "PAYLOAD_TOO_LARGE",
  "message": "Request payload exceeds maximum allowed size",
  "details": {
    "max_size_mb": 50,
    "path": "/api/v1/documents/upload"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Invalid file format**:
```json
{
  "error": "INVALID_FILE",
  "message": "Unsupported file format: txt. Supported: pdf, png, jpg, jpeg, tiff, docx",
  "details": {
    "filename": "document.txt"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## 🔍 Job Status Tracking

### Job Statuses

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting to be processed |
| `running` | Job is currently being processed |
| `success` | Job completed successfully |
| `partial` | Job completed with some errors |
| `failed` | Job failed to complete |

### Monitoring Job Progress

```bash
# Poll job status every 5 seconds
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/v1/jobs/job_abc12345" | jq -r '.status')
  echo "Job status: $STATUS"
  
  if [ "$STATUS" = "success" ] || [ "$STATUS" = "failed" ]; then
    break
  fi
  
  sleep 5
done
```

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t intellidoc-api .

# Run the container
docker run -d \
  --name intellidoc-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  intellidoc-api

# Check logs
docker logs intellidoc-api

# Health check
curl http://localhost:8000/health
```

### Docker Compose

```yaml
version: '3.8'
services:
  intellidoc-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONPATH=/app
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 📊 Performance & Monitoring

### Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| **Accuracy** | 95%+ | Entity extraction accuracy |
| **Throughput** | 100+ docs/hour | Processing capacity |
| **Latency** | <30s per document | Response time for sync processing |
| **Uptime** | 99.9% | Service availability |

### Monitoring Endpoints

**System Metrics**:
```bash
curl http://localhost:8000/api/v1/status/metrics
```

**Queue Health**:
```bash
curl http://localhost:8000/api/v1/jobs/stats/summary
```

**Component Status**:
```bash
curl http://localhost:8000/api/v1/status/components
```

## 🔒 Security Considerations

### Production Deployment

1. **Enable Authentication**: Add API key or JWT authentication
2. **Configure CORS**: Restrict origins in production
3. **Rate Limiting**: Add rate limiting middleware
4. **HTTPS**: Use TLS/SSL for encrypted communication
5. **File Validation**: Implement virus scanning for uploads
6. **Input Sanitization**: Validate all file inputs

### Example Production Configuration

```python
# In production, configure CORS properly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

## 🧪 Testing

### API Testing with curl

```bash
# Test health endpoint
curl -f http://localhost:8000/health

# Test document upload
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@test_document.pdf" \
  -F "async_processing=false"

# Test system status
curl http://localhost:8000/api/v1/status
```

### Load Testing

```bash
# Install Apache Bench
apt install apache2-utils

# Basic load test
ab -n 100 -c 10 http://localhost:8000/health

# Upload load test (requires test file)
ab -n 50 -c 5 -p test_upload.txt -T "multipart/form-data" \
  http://localhost:8000/api/v1/documents/upload
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
export INTELLIDOC_CONFIG=/path/to/config.yaml
export INTELLIDOC_LOG_LEVEL=INFO
export INTELLIDOC_MAX_WORKERS=4

# Start API with custom config
python start_api.py
```

### Configuration File

See `config/config.yaml` for detailed configuration options including:
- OCR engine settings
- NLP model configurations
- Processing parameters
- File size limits
- Timeout settings

## 📚 API Reference

For complete API documentation with interactive examples, visit:
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc

## 🆘 Troubleshooting

### Common Issues

1. **NumPy compatibility errors**: 
   ```bash
   pip install 'numpy<2.0'
   ```

2. **Missing OCR dependencies**:
   ```bash
   pip install pytesseract paddleocr
   ```

3. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Port already in use**:
   ```bash
   # Change port in start_api.py or use:
   uvicorn src.api.main:app --port 8001
   ```

### Health Check

```bash
# Quick system check
curl http://localhost:8000/api/v1/status/health

# Detailed component status
curl http://localhost:8000/api/v1/status/components
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**IntelliDoc API** - Built with ❤️ using FastAPI, OCR, and NLP technologies
