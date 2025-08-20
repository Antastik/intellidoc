"""
System status and health check endpoints for IntelliDoc API.

Provides endpoints for monitoring system health, component availability,
and service metrics for operational monitoring and debugging.
"""

import time
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from ..schemas.pipeline import SystemStatusSchema, ErrorResponse
from ..services.pipeline_service import get_pipeline_service, PipelineService
from ..jobs.job_store import get_job_store, JobStore


# Track service start time for uptime calculation
_service_start_time = time.time()


# Create router
router = APIRouter(
    prefix="/status",
    tags=["System Status"],
    responses={
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)


@router.get(
    "",
    response_model=SystemStatusSchema,
    summary="Get system status",
    description="""
    Get comprehensive system health and status information.
    
    **Returns:**
    - Service information and uptime
    - Component availability (OCR, NLP, document ingestion)
    - Queue statistics and job health
    - System resource information
    
    **Use cases:**
    - Health checks for load balancers
    - Monitoring and alerting
    - Debugging component issues
    - Service discovery
    """
)
async def get_system_status(
    pipeline_service: PipelineService = Depends(get_pipeline_service),
    job_store: JobStore = Depends(get_job_store)
):
    """Get comprehensive system status and health information"""
    
    try:
        # Calculate uptime
        uptime_seconds = time.time() - _service_start_time
        
        # Get pipeline component status
        pipeline_status = pipeline_service.get_system_status()
        
        # Get job queue statistics
        job_stats = job_store.get_job_stats()
        
        # Determine overall system health
        overall_status = "healthy"
        
        # Check if critical components are available
        if not pipeline_status.get('document_ingestor', {}).get('available', False):
            overall_status = "degraded"
        
        if not pipeline_status.get('ocr_processor', {}).get('available', False):
            overall_status = "degraded"
            
        if not pipeline_status.get('nlp_processor', {}).get('available', False):
            overall_status = "degraded"
        
        # Check if there are too many failed jobs
        failed_jobs = job_stats.get('failed', 0)
        total_jobs = sum(job_stats.values())
        
        if total_jobs > 0 and (failed_jobs / total_jobs) > 0.5:
            overall_status = "unhealthy"
        
        return SystemStatusSchema(
            service="IntelliDoc API",
            version="1.0.0",
            status=overall_status,
            uptime_seconds=uptime_seconds,
            components=pipeline_status,
            queue_status=job_stats
        )
    
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        
        # Return degraded status with error information
        return SystemStatusSchema(
            service="IntelliDoc API",
            version="1.0.0",
            status="degraded",
            uptime_seconds=time.time() - _service_start_time,
            components={
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            },
            queue_status={"error": "Unable to retrieve queue status"}
        )


@router.get(
    "/health",
    summary="Health check endpoint",
    description="""
    Simple health check endpoint for load balancers and monitoring systems.
    
    **Returns:**
    - HTTP 200 if service is healthy
    - HTTP 503 if service is unhealthy
    - Minimal response body with status
    
    **Response format:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2024-01-15T10:30:00Z"
    }
    ```
    """
)
async def health_check(
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """Simple health check for load balancers"""
    
    try:
        # Quick check of critical components
        status = pipeline_service.get_system_status()
        
        # Check if document ingestion is available (minimum requirement)
        if not status.get('document_ingestor', {}).get('available', False):
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "unhealthy",
                    "reason": "Document ingestion not available",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "reason": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get(
    "/components",
    summary="Get component status details",
    description="""
    Get detailed status information for all system components.
    
    **Returns:**
    - Document ingestion component status
    - OCR processor engines and availability
    - NLP processor components and models
    - Configuration and capabilities
    """
)
async def get_component_status(
    pipeline_service: PipelineService = Depends(get_pipeline_service)
):
    """Get detailed component status information"""
    
    try:
        status = pipeline_service.get_system_status()
        
        # Enhance with additional details
        enhanced_status = {
            "document_ingestor": status.get('document_ingestor', {}),
            "ocr_processor": status.get('ocr_processor', {}),
            "nlp_processor": status.get('nlp_processor', {}),
            "pipeline_config": status.get('pipeline', {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return enhanced_status
    
    except Exception as e:
        logger.error(f"Failed to get component status: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "COMPONENT_STATUS_ERROR",
                "message": f"Failed to retrieve component status: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@router.get(
    "/metrics",
    summary="Get system metrics",
    description="""
    Get system performance metrics and statistics.
    
    **Returns:**
    - Service uptime and performance
    - Job processing statistics
    - Queue health metrics
    - Resource utilization (if available)
    """
)
async def get_system_metrics(
    job_store: JobStore = Depends(get_job_store)
):
    """Get system performance metrics"""
    
    try:
        uptime_seconds = time.time() - _service_start_time
        
        # Get job statistics
        job_stats = job_store.get_job_stats()
        total_jobs = sum(job_stats.values())
        
        # Calculate success rate
        successful_jobs = job_stats.get('success', 0) + job_stats.get('partial', 0)
        success_rate = (successful_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        # Calculate failure rate
        failed_jobs = job_stats.get('failed', 0)
        failure_rate = (failed_jobs / total_jobs * 100) if total_jobs > 0 else 0
        
        return {
            "service_metrics": {
                "uptime_seconds": uptime_seconds,
                "uptime_human": _format_uptime(uptime_seconds),
                "start_time": datetime.fromtimestamp(_service_start_time).isoformat()
            },
            "job_metrics": {
                "total_jobs": total_jobs,
                "success_rate_percent": round(success_rate, 2),
                "failure_rate_percent": round(failure_rate, 2),
                "jobs_by_status": job_stats,
                "active_jobs": job_stats.get('running', 0),
                "queued_jobs": job_stats.get('queued', 0)
            },
            "system_health": {
                "status": "healthy" if failure_rate < 10 else "degraded",
                "load_level": _calculate_load_level(job_stats),
                "queue_depth": job_stats.get('queued', 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "METRICS_ERROR",
                "message": f"Failed to retrieve system metrics: {str(e)}",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


def _format_uptime(seconds: float) -> str:
    """Format uptime seconds into human-readable string"""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def _calculate_load_level(job_stats: Dict[str, int]) -> str:
    """Calculate system load level based on job statistics"""
    active_jobs = job_stats.get('running', 0)
    queued_jobs = job_stats.get('queued', 0)
    
    total_active = active_jobs + queued_jobs
    
    if total_active == 0:
        return "idle"
    elif total_active <= 5:
        return "low"
    elif total_active <= 20:
        return "medium"
    else:
        return "high"
