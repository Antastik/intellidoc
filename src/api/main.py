"""
IntelliDoc FastAPI Application

Main FastAPI application with middleware, error handling, and API routes.
Provides RESTful endpoints for document processing with OCR and NLP.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger
import uvicorn

# Import API routes
from .routes import documents, jobs, status

# Import error handling
from .schemas.pipeline import ErrorResponse

# Add the parent directory to Python path for pipeline imports
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import pipeline components for error handling
try:
    from src.utils.document_ingestion import DocumentIngestionError
except ImportError:
    # Fallback if import fails
    class DocumentIngestionError(Exception):
        pass


# Initialize FastAPI application
app = FastAPI(
    title="IntelliDoc API",
    description="""
    ## IntelliDoc - Intelligent Document Processing API
    
    A comprehensive RESTful API for document processing using OCR and NLP technologies.
    
    ### Features
    - **Multi-format Support**: PDF, images (PNG, JPG, TIFF), and DOCX files
    - **OCR Processing**: Extract text from images and scanned documents
    - **NLP Analysis**: Entity extraction, classification, and structured output
    - **Async Processing**: Background job processing for large documents
    - **Batch Processing**: Handle multiple documents simultaneously
    - **Job Tracking**: Monitor processing status and download results
    
    ### Processing Modes
    - **Synchronous**: Immediate processing and response (< 30s)
    - **Asynchronous**: Background processing with job tracking (> 30s)
    
    ### API Endpoints
    - **Documents**: Upload and process single or multiple documents
    - **Jobs**: Track background processing status and download results
    - **Status**: System health checks and component monitoring
    
    ### Authentication
    Currently no authentication required (configurable for production).
    
    ### Rate Limits
    - Max file size: 50MB per document
    - Max batch size: 20 documents
    - Processing timeout: 300 seconds per document
    """,
    version="1.0.0",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    openapi_url="/api/v1/openapi.json",
    contact={
        "name": "IntelliDoc API Support",
        "email": "support@intellidoc.ai",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests and response times"""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response: {response.status_code} "
        f"({process_time:.3f}s) "
        f"{request.method} {request.url.path}"
    )
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Error handlers
@app.exception_handler(DocumentIngestionError)
async def document_ingestion_error_handler(request: Request, exc: DocumentIngestionError):
    """Handle document ingestion errors"""
    logger.error(f"Document ingestion error: {exc}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "DOCUMENT_INGESTION_ERROR",
            "message": str(exc),
            "details": {
                "path": str(request.url.path),
                "method": request.method
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    logger.error(f"Validation error: {exc}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
                "validation_errors": exc.errors(),
                "path": str(request.url.path),
                "method": request.method
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(413)
async def payload_too_large_handler(request: Request, exc):
    """Handle payload too large errors"""
    logger.error(f"Payload too large: {request.url.path}")
    
    return JSONResponse(
        status_code=413,
        content={
            "error": "PAYLOAD_TOO_LARGE",
            "message": "Request payload exceeds maximum allowed size",
            "details": {
                "max_size_mb": 50,  # TODO: Get from config
                "path": str(request.url.path)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request: Request, exc: Exception):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred",
            "details": {
                "path": str(request.url.path),
                "method": request.method
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent format"""
    
    # If detail is already a dict, use it as is
    if isinstance(exc.detail, dict):
        content = exc.detail
    else:
        content = {
            "error": f"HTTP_{exc.status_code}",
            "message": str(exc.detail),
            "details": {
                "status_code": exc.status_code,
                "path": str(request.url.path),
                "method": request.method
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    return JSONResponse(
        status_code=exc.status_code,
        content=content
    )


# Include API routers
app.include_router(
    documents.router,
    prefix="/api/v1",
    tags=["Documents"]
)

app.include_router(
    jobs.router,
    prefix="/api/v1",
    tags=["Jobs"]
)

app.include_router(
    status.router,
    prefix="/api/v1",
    tags=["System Status"]
)


# Root endpoint
@app.get(
    "/",
    summary="API Information",
    description="Get basic information about the IntelliDoc API",
    tags=["General"]
)
async def root():
    """Get API information and available endpoints"""
    return {
        "service": "IntelliDoc API",
        "version": "1.0.0",
        "description": "Intelligent Document Processing with OCR and NLP",
        "endpoints": {
            "docs": "/api/v1/docs",
            "redoc": "/api/v1/redoc",
            "openapi": "/api/v1/openapi.json",
            "status": "/api/v1/status",
            "health": "/api/v1/status/health"
        },
        "features": [
            "Document upload and processing",
            "OCR text extraction",
            "NLP entity extraction",
            "Asynchronous processing",
            "Batch processing",
            "Job tracking and management"
        ],
        "supported_formats": ["PDF", "PNG", "JPG", "JPEG", "TIFF", "DOCX"],
        "timestamp": datetime.utcnow().isoformat()
    }


# Health check endpoint (simple)
@app.get(
    "/health",
    summary="Simple Health Check",
    description="Simple health check endpoint for load balancers",
    tags=["General"]
)
async def health():
    """Simple health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# Application startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting IntelliDoc API...")
    
    # Initialize required directories
    required_dirs = [
        "data/uploads",
        "data/output",
        "data/input",
        "models",
        "temp"
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("IntelliDoc API started successfully")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("Shutting down IntelliDoc API...")
    
    # Cleanup temporary files
    temp_dir = Path("data/uploads")
    if temp_dir.exists():
        for temp_file in temp_dir.glob("tmp*"):
            try:
                temp_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {temp_file}: {e}")
    
    logger.info("IntelliDoc API shutdown complete")


# For running with uvicorn directly
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
