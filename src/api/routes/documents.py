"""
Document processing endpoints for IntelliDoc API.

Provides RESTful endpoints for single and batch document processing
with both synchronous and asynchronous operation modes.
"""

from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse
from loguru import logger

from ..schemas.pipeline import (
    UploadResponse, BatchUploadResponse, ProcessingResultSchema, ErrorResponse
)
from ..services.pipeline_service import get_pipeline_service, PipelineService


# Create router
router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        413: {"model": ErrorResponse, "description": "Payload Too Large"},
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload and process single document",
    description="""
    Upload a single document for OCR and NLP processing.
    
    **Supported formats:** PDF, PNG, JPG, JPEG, TIFF, DOCX
    
    **Processing modes:**
    - `async_processing=false` (default): Synchronous processing, returns results immediately
    - `async_processing=true`: Asynchronous processing, returns job ID for tracking
    
    **File size limit:** 50MB by default (configurable)
    """
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    async_processing: bool = Query(False, description="Enable asynchronous processing"),
    save_result: bool = Query(True, description="Save processing result to file"),
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """Upload and process a single document"""
    
    # Validate uploaded file
    validation_error = pipeline_service.validate_upload_file(file)
    if validation_error:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_FILE",
            "message": validation_error,
            "details": {"filename": file.filename}
        })
    
    try:
        if async_processing:
            # Asynchronous processing
            job_id = await pipeline_service.process_document_async(
                upload_file=file,
                background_tasks=background_tasks,
                save_result=save_result
            )
            
            return UploadResponse(
                message="Document uploaded successfully. Processing started.",
                job_id=job_id
            )
        
        else:
            # Synchronous processing
            result = await pipeline_service.process_document_sync(
                upload_file=file,
                save_result=save_result
            )
            
            return UploadResponse(
                message="Document processed successfully",
                document_id=result.document_id,
                result=result
            )
    
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "PROCESSING_ERROR",
            "message": f"Failed to process document: {str(e)}",
            "details": {"filename": file.filename}
        })


@router.post(
    "/batch",
    response_model=BatchUploadResponse,
    summary="Upload and process multiple documents",
    description="""
    Upload multiple documents for batch OCR and NLP processing.
    
    **Supported formats:** PDF, PNG, JPG, JPEG, TIFF, DOCX
    
    **Processing modes:**
    - `async_processing=false` (default): Synchronous processing, returns all results immediately
    - `async_processing=true`: Asynchronous processing, returns job ID for tracking
    
    **Processing options:**
    - `parallel_processing=true` (default): Process files in parallel for faster throughput
    - `save_individual=true` (default): Save individual processing results
    - `save_batch_summary=true` (default): Save batch processing summary
    
    **Limits:** Maximum 20 files per batch, 50MB per file
    """
)
async def upload_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Document files to process"),
    async_processing: bool = Query(False, description="Enable asynchronous processing"),
    parallel_processing: bool = Query(True, description="Enable parallel processing"),
    save_individual: bool = Query(True, description="Save individual results to files"),
    save_batch_summary: bool = Query(True, description="Save batch summary"),
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """Upload and process multiple documents in batch"""
    
    # Validate batch size
    max_batch_size = 20  # Configurable limit
    if len(files) > max_batch_size:
        raise HTTPException(status_code=400, detail={
            "error": "BATCH_TOO_LARGE",
            "message": f"Batch size exceeds maximum limit of {max_batch_size} files",
            "details": {"file_count": len(files), "max_files": max_batch_size}
        })
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail={
            "error": "EMPTY_BATCH",
            "message": "No files provided for processing",
            "details": {}
        })
    
    # Validate all uploaded files
    validation_errors = []
    for i, file in enumerate(files):
        validation_error = pipeline_service.validate_upload_file(file)
        if validation_error:
            validation_errors.append({
                "file_index": i,
                "filename": file.filename,
                "error": validation_error
            })
    
    if validation_errors:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_FILES",
            "message": f"Validation failed for {len(validation_errors)} files",
            "details": {"validation_errors": validation_errors}
        })
    
    try:
        if async_processing:
            # Asynchronous batch processing
            job_id = await pipeline_service.process_batch_async(
                upload_files=files,
                background_tasks=background_tasks,
                parallel=parallel_processing,
                save_individual=save_individual,
                save_batch_summary=save_batch_summary
            )
            
            return BatchUploadResponse(
                message="Batch uploaded successfully. Processing started.",
                total_files=len(files),
                job_id=job_id
            )
        
        else:
            # Synchronous batch processing
            results = await pipeline_service.process_batch_sync(
                upload_files=files,
                parallel=parallel_processing,
                save_individual=save_individual,
                save_batch_summary=save_batch_summary
            )
            
            return BatchUploadResponse(
                message="Batch processed successfully",
                total_files=len(files),
                results=results
            )
    
    except Exception as e:
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "BATCH_PROCESSING_ERROR",
            "message": f"Failed to process batch: {str(e)}",
            "details": {"file_count": len(files)}
        })


@router.get(
    "/formats",
    summary="Get supported document formats",
    description="Returns list of supported document formats and processing capabilities"
)
async def get_supported_formats(
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """Get supported document formats and processing information"""
    
    try:
        # Get system status to determine supported formats
        status = pipeline_service.get_system_status()
        
        supported_formats = []
        if 'document_ingestor' in status and status['document_ingestor']['available']:
            supported_formats = status['document_ingestor']['supported_formats']
        
        # Get OCR engines
        ocr_engines = []
        if 'ocr_processor' in status and status['ocr_processor']['available']:
            engines = status['ocr_processor']['engines']
            ocr_engines = [name for name, available in engines.items() if available]
        
        # Get NLP components
        nlp_components = []
        if 'nlp_processor' in status and status['nlp_processor']['available']:
            components = status['nlp_processor']['components']
            nlp_components = [name for name, available in components.items() if available]
        
        return {
            "supported_formats": supported_formats,
            "max_file_size_mb": 50,  # TODO: Get from config
            "max_batch_size": 20,
            "processing_capabilities": {
                "ocr_engines": ocr_engines,
                "nlp_components": nlp_components,
                "parallel_processing": True,
                "async_processing": True
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "STATUS_ERROR",
            "message": f"Failed to get format information: {str(e)}"
        })
