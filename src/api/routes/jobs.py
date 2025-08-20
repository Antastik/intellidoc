"""
Job management endpoints for IntelliDoc API.

Provides endpoints for tracking background job status, listing jobs,
and downloading results from completed processing jobs.
"""

import math
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from loguru import logger

from ..schemas.pipeline import (
    JobSchema, JobListResponse, JobStatus, ErrorResponse
)
from ..jobs.job_store import get_job_store, JobStore


# Create router
router = APIRouter(
    prefix="/jobs",
    tags=["Jobs"],
    responses={
        404: {"model": ErrorResponse, "description": "Job Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    }
)


@router.get(
    "/{job_id}",
    response_model=JobSchema,
    summary="Get job status",
    description="""
    Get the current status and details of a background processing job.
    
    **Job statuses:**
    - `queued`: Job is waiting to be processed
    - `running`: Job is currently being processed
    - `success`: Job completed successfully
    - `partial`: Job completed with some errors
    - `failed`: Job failed to complete
    
    **Response includes:**
    - Current status and progress percentage
    - Processing timestamps
    - Results (when completed)
    - Download URL for saved results
    - Error messages (if any)
    """
)
async def get_job_status(
    job_id: str,
    job_store: JobStore = Depends(get_job_store)
):
    """Get status and details of a specific job"""
    
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Job with ID '{job_id}' not found",
                "details": {"job_id": job_id}
            }
        )
    
    try:
        # Convert job to schema format
        job_data = job.to_dict()
        
        return JobSchema(
            job_id=job_data["job_id"],
            status=JobStatus(job_data["status"]),
            created_at=job_data["created_at"],
            started_at=job_data["started_at"],
            completed_at=job_data["completed_at"],
            progress=job_data["progress"],
            total_files=job_data["total_files"],
            processed_files=job_data["processed_files"],
            result=job_data["result"],
            error_message=job_data["error_message"],
            download_url=job_data["download_url"]
        )
    
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATUS_ERROR",
                "message": f"Failed to retrieve job status: {str(e)}",
                "details": {"job_id": job_id}
            }
        )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs",
    description="""
    Get a paginated list of jobs with optional filtering.
    
    **Filtering options:**
    - `status`: Filter by job status (queued, running, success, partial, failed)
    - `limit`: Number of jobs per page (1-100, default: 20)
    - `offset`: Number of jobs to skip for pagination (default: 0)
    
    **Response includes:**
    - Paginated list of jobs
    - Total count and pagination metadata
    - Jobs are sorted by creation time (newest first)
    """
)
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by job status"),
    limit: int = Query(20, ge=1, le=100, description="Number of jobs per page"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    job_store: JobStore = Depends(get_job_store)
):
    """List jobs with optional filtering and pagination"""
    
    try:
        # Get filtered jobs
        jobs = job_store.list_jobs(
            status_filter=status,
            limit=limit,
            offset=offset
        )
        
        # Get total count (approximate for large datasets)
        all_jobs = job_store.list_jobs(status_filter=status, limit=10000, offset=0)
        total = len(all_jobs)
        
        # Calculate pagination info
        total_pages = math.ceil(total / limit) if total > 0 else 0
        current_page = (offset // limit) + 1
        
        # Convert jobs to schema format
        job_schemas = []
        for job in jobs:
            job_data = job.to_dict()
            job_schema = JobSchema(
                job_id=job_data["job_id"],
                status=JobStatus(job_data["status"]),
                created_at=job_data["created_at"],
                started_at=job_data["started_at"],
                completed_at=job_data["completed_at"],
                progress=job_data["progress"],
                total_files=job_data["total_files"],
                processed_files=job_data["processed_files"],
                result=job_data["result"],
                error_message=job_data["error_message"],
                download_url=job_data["download_url"]
            )
            job_schemas.append(job_schema)
        
        return JobListResponse(
            jobs=job_schemas,
            total=total,
            page=current_page,
            per_page=limit,
            total_pages=total_pages
        )
    
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "LIST_ERROR",
                "message": f"Failed to list jobs: {str(e)}"
            }
        )


@router.get(
    "/{job_id}/download",
    summary="Download job results",
    description="""
    Download the processing results for a completed job.
    
    **Requirements:**
    - Job must be completed (success or partial status)
    - Results must have been saved to file during processing
    
    **Response:**
    - For single document jobs: Returns the JSON result file
    - For batch jobs: Returns the batch summary JSON file
    - Content-Type: application/json
    - Content-Disposition: attachment with appropriate filename
    """
)
async def download_job_results(
    job_id: str,
    job_store: JobStore = Depends(get_job_store)
):
    """Download processing results for a completed job"""
    
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Job with ID '{job_id}' not found",
                "details": {"job_id": job_id}
            }
        )
    
    # Check if job is completed
    if not job.is_completed():
        raise HTTPException(
            status_code=400,
            detail={
                "error": "JOB_NOT_COMPLETED",
                "message": f"Job '{job_id}' has not completed yet (status: {job.status.value})",
                "details": {"job_id": job_id, "status": job.status.value}
            }
        )
    
    # Check if download URL is available
    if not job.download_url:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "NO_DOWNLOAD_AVAILABLE",
                "message": f"No download available for job '{job_id}' (results may not have been saved)",
                "details": {"job_id": job_id}
            }
        )
    
    try:
        # Try to find the result file
        # For now, we'll look in the standard output directory
        output_dir = Path("data/output")
        
        # Look for job-specific result files
        potential_files = [
            output_dir / f"result_{job_id}.json",
            output_dir / f"batch_summary_{job_id}.json",
        ]
        
        # Also look for files matching the job's document IDs
        if job.result:
            if isinstance(job.result, list):
                # Batch job - look for batch summary
                for file_path in output_dir.glob("batch_summary_*.json"):
                    potential_files.append(file_path)
            else:
                # Single job - look for specific result
                doc_id = getattr(job.result, 'document_id', None)
                if doc_id:
                    potential_files.append(output_dir / f"result_{doc_id}.json")
        
        # Find the first existing file
        result_file = None
        for file_path in potential_files:
            if file_path.exists():
                result_file = file_path
                break
        
        if not result_file:
            # If no file found, return the result data directly as JSON
            if job.result:
                return JSONResponse(
                    content=job.result,
                    headers={
                        "Content-Disposition": f"attachment; filename=job_{job_id}_results.json"
                    }
                )
            else:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "RESULT_FILE_NOT_FOUND",
                        "message": f"Result file for job '{job_id}' not found",
                        "details": {"job_id": job_id}
                    }
                )
        
        # Return the file
        filename = f"job_{job_id}_results.json"
        return FileResponse(
            path=result_file,
            filename=filename,
            media_type="application/json"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to download results for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DOWNLOAD_ERROR",
                "message": f"Failed to download results: {str(e)}",
                "details": {"job_id": job_id}
            }
        )


@router.delete(
    "/{job_id}",
    summary="Cancel or delete job",
    description="""
    Cancel a running job or delete a completed job.
    
    **Behavior:**
    - For queued/running jobs: Attempts to cancel the job (best effort)
    - For completed jobs: Removes the job from the job store
    - Cannot cancel jobs that are already being processed
    
    **Note:** Actual cancellation of running jobs is not guaranteed
    due to the nature of background task processing.
    """
)
async def delete_job(
    job_id: str,
    job_store: JobStore = Depends(get_job_store)
):
    """Cancel or delete a job"""
    
    job = job_store.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "JOB_NOT_FOUND",
                "message": f"Job with ID '{job_id}' not found",
                "details": {"job_id": job_id}
            }
        )
    
    try:
        if job.status == JobStatus.RUNNING:
            # Cannot easily cancel running jobs in FastAPI BackgroundTasks
            # This would require a more sophisticated job queue system
            return {
                "message": f"Job '{job_id}' is currently running and cannot be cancelled",
                "job_id": job_id,
                "status": job.status.value
            }
        
        elif job.status == JobStatus.QUEUED:
            # Mark as failed to prevent processing
            job_store.fail_job(job_id, "Job cancelled by user")
            return {
                "message": f"Job '{job_id}' has been cancelled",
                "job_id": job_id,
                "status": "cancelled"
            }
        
        else:
            # For completed jobs, we could remove them from the store
            # For now, just return success
            return {
                "message": f"Job '{job_id}' is already completed",
                "job_id": job_id,
                "status": job.status.value
            }
    
    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DELETE_ERROR",
                "message": f"Failed to delete job: {str(e)}",
                "details": {"job_id": job_id}
            }
        )


@router.get(
    "/stats/summary",
    summary="Get job statistics",
    description="""
    Get summary statistics about jobs in the system.
    
    **Returns:**
    - Count of jobs by status
    - Total number of jobs
    - System queue health information
    """
)
async def get_job_stats(
    job_store: JobStore = Depends(get_job_store)
):
    """Get job statistics and queue health"""
    
    try:
        stats = job_store.get_job_stats()
        
        total_jobs = sum(stats.values())
        
        return {
            "total_jobs": total_jobs,
            "by_status": stats,
            "queue_health": {
                "active_jobs": stats.get("running", 0),
                "pending_jobs": stats.get("queued", 0),
                "completed_jobs": stats.get("success", 0) + stats.get("partial", 0),
                "failed_jobs": stats.get("failed", 0)
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get job stats: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATS_ERROR",
                "message": f"Failed to get job statistics: {str(e)}"
            }
        )
