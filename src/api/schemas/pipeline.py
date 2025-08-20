"""
Pydantic schemas for IntelliDoc API request/response models.

Maps the internal PipelineResult to API-friendly schemas with validation,
examples, and proper documentation for OpenAPI generation.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic.types import constr


class JobStatus(str, Enum):
    """Job processing status enumeration"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class EntitySchema(BaseModel):
    """Extracted entity from NLP processing"""
    text: str = Field(..., description="The extracted entity text")
    label: str = Field(..., description="Entity type/category")
    start: int = Field(..., description="Start position in text", ge=0)
    end: int = Field(..., description="End position in text", ge=0)
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional entity metadata")

    class Config:
        schema_extra = {
            "example": {
                "text": "John Doe",
                "label": "PERSON",
                "start": 15,
                "end": 23,
                "confidence": 0.95,
                "metadata": {"spacy_label": "PERSON"}
            }
        }


class DocumentMetadataSchema(BaseModel):
    """Document metadata information"""
    file_name: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes", ge=0)
    mime_type: str = Field(..., description="MIME type of the file")
    format: str = Field(..., description="File format/extension")
    pages: int = Field(1, description="Number of pages", ge=1)
    created_at: Optional[str] = Field(None, description="File creation timestamp")
    modified_at: Optional[str] = Field(None, description="File modification timestamp")

    class Config:
        schema_extra = {
            "example": {
                "file_name": "invoice_001.pdf",
                "file_size": 245760,
                "mime_type": "application/pdf",
                "format": "pdf",
                "pages": 2,
                "created_at": "1692547200.0",
                "modified_at": "1692547200.0"
            }
        }


class ProcessingResultSchema(BaseModel):
    """Document processing result"""
    document_id: str = Field(..., description="Unique document identifier")
    status: JobStatus = Field(..., description="Processing status")
    combined_text: str = Field("", description="Extracted text content")
    entities: List[EntitySchema] = Field([], description="Extracted entities")
    processing_time: float = Field(..., description="Processing time in seconds", ge=0)
    errors: List[str] = Field([], description="Processing errors if any")
    document_metadata: Optional[DocumentMetadataSchema] = Field(None, description="Document metadata")

    class Config:
        schema_extra = {
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "success",
                "combined_text": "INVOICE #1001\nDate: January 15, 2024\nBill To: John Doe...",
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
                "errors": [],
                "document_metadata": {
                    "file_name": "invoice_001.pdf",
                    "file_size": 245760,
                    "mime_type": "application/pdf",
                    "format": "pdf",
                    "pages": 2
                }
            }
        }


class UploadResponse(BaseModel):
    """Response for single document upload"""
    message: str = Field(..., description="Response message")
    document_id: Optional[str] = Field(None, description="Document ID if processing completed")
    job_id: Optional[str] = Field(None, description="Job ID if processing async")
    result: Optional[ProcessingResultSchema] = Field(None, description="Processing result if completed synchronously")

    class Config:
        schema_extra = {
            "example": {
                "message": "Document uploaded and processed successfully",
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "result": {
                    "document_id": "550e8400-e29b-41d4-a716-446655440000",
                    "status": "success",
                    "combined_text": "Sample extracted text...",
                    "entities": [],
                    "processing_time": 1.23,
                    "errors": []
                }
            }
        }


class BatchUploadResponse(BaseModel):
    """Response for batch document upload"""
    message: str = Field(..., description="Response message")
    total_files: int = Field(..., description="Total number of files", ge=0)
    job_id: Optional[str] = Field(None, description="Batch job ID if processing async")
    results: Optional[List[ProcessingResultSchema]] = Field(None, description="Processing results if completed synchronously")

    class Config:
        schema_extra = {
            "example": {
                "message": "Batch processing completed",
                "total_files": 3,
                "results": [
                    {
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "success",
                        "processing_time": 1.23,
                        "entities": []
                    }
                ]
            }
        }


class JobSchema(BaseModel):
    """Background job information"""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    progress: float = Field(0.0, description="Completion progress percentage", ge=0.0, le=100.0)
    total_files: int = Field(1, description="Total files to process", ge=1)
    processed_files: int = Field(0, description="Files processed so far", ge=0)
    result: Optional[Union[ProcessingResultSchema, List[ProcessingResultSchema]]] = Field(
        None, description="Processing result(s) when completed"
    )
    error_message: Optional[str] = Field(None, description="Error message if failed")
    download_url: Optional[str] = Field(None, description="URL to download results")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "job_550e8400-e29b-41d4-a716-446655440000",
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
                "download_url": "/api/v1/jobs/job_550e8400-e29b-41d4-a716-446655440000/download"
            }
        }


class JobListResponse(BaseModel):
    """Paginated job list response"""
    jobs: List[JobSchema] = Field(..., description="List of jobs")
    total: int = Field(..., description="Total number of jobs", ge=0)
    page: int = Field(..., description="Current page number", ge=1)
    per_page: int = Field(..., description="Items per page", ge=1, le=100)
    total_pages: int = Field(..., description="Total number of pages", ge=0)

    class Config:
        schema_extra = {
            "example": {
                "jobs": [
                    {
                        "job_id": "job_1",
                        "status": "success",
                        "created_at": "2024-01-15T10:30:00Z",
                        "progress": 100.0,
                        "total_files": 1,
                        "processed_files": 1
                    }
                ],
                "total": 25,
                "page": 1,
                "per_page": 10,
                "total_pages": 3
            }
        }


class SystemStatusSchema(BaseModel):
    """System status and health information"""
    service: str = Field("IntelliDoc API", description="Service name")
    version: str = Field("1.0.0", description="Service version")
    status: str = Field("healthy", description="Overall system status")
    uptime_seconds: float = Field(..., description="System uptime in seconds", ge=0)
    components: Dict[str, Dict[str, Any]] = Field(..., description="Component status details")
    queue_status: Dict[str, int] = Field(..., description="Job queue statistics")

    class Config:
        schema_extra = {
            "example": {
                "service": "IntelliDoc API",
                "version": "1.0.0",
                "status": "healthy",
                "uptime_seconds": 3600.0,
                "components": {
                    "document_ingestor": {
                        "available": True,
                        "supported_formats": ["pdf", "png", "jpg", "docx"]
                    },
                    "ocr_processor": {
                        "available": True,
                        "engines": {"tesseract": True, "paddleocr": True}
                    },
                    "nlp_processor": {
                        "available": True,
                        "components": {"spacy": True, "transformers": True}
                    }
                },
                "queue_status": {
                    "queued": 2,
                    "running": 1,
                    "completed": 45
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema"""
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        schema_extra = {
            "example": {
                "error": "DOCUMENT_TOO_LARGE",
                "message": "Document size exceeds maximum allowed limit of 50MB",
                "details": {
                    "file_size": 104857600,
                    "max_size": 52428800,
                    "filename": "large_document.pdf"
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# Request schemas for validation
class UploadRequest(BaseModel):
    """Document upload request parameters"""
    async_processing: bool = Field(False, description="Enable asynchronous processing")
    save_result: bool = Field(True, description="Save processing result to file")
    
    class Config:
        schema_extra = {
            "example": {
                "async_processing": True,
                "save_result": True
            }
        }


class BatchUploadRequest(BaseModel):
    """Batch document upload request parameters"""
    async_processing: bool = Field(False, description="Enable asynchronous processing")
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    save_individual: bool = Field(True, description="Save individual results to files")
    save_batch_summary: bool = Field(True, description="Save batch summary")
    
    class Config:
        schema_extra = {
            "example": {
                "async_processing": True,
                "parallel_processing": True,
                "save_individual": True,
                "save_batch_summary": True
            }
        }
