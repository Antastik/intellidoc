"""
In-memory job tracking system for IntelliDoc API.

Provides thread-safe storage and management of background processing jobs
with status tracking, progress updates, and result storage.
"""

import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..schemas.pipeline import JobStatus, ProcessingResultSchema


@dataclass
class Job:
    """Represents a background processing job"""
    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    total_files: int = 1
    processed_files: int = 0
    result: Optional[Union[ProcessingResultSchema, List[ProcessingResultSchema]]] = None
    error_message: Optional[str] = None
    download_url: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_progress(self, processed: int, total: Optional[int] = None):
        """Update job progress"""
        if total is not None:
            self.total_files = total
        self.processed_files = processed
        self.progress = (processed / self.total_files * 100.0) if self.total_files > 0 else 0.0

    def mark_started(self):
        """Mark job as started"""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
        logger.info(f"Job {self.job_id} started")

    def mark_completed(self, result: Union[ProcessingResultSchema, List[ProcessingResultSchema]], 
                      status: JobStatus = JobStatus.SUCCESS):
        """Mark job as completed with result"""
        self.status = status
        self.completed_at = datetime.utcnow()
        self.result = result
        self.progress = 100.0
        self.processed_files = self.total_files
        logger.info(f"Job {self.job_id} completed with status: {status}")

    def mark_failed(self, error_message: str):
        """Mark job as failed with error message"""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error_message
        logger.error(f"Job {self.job_id} failed: {error_message}")

    def is_completed(self) -> bool:
        """Check if job is in a completed state"""
        return self.status in [JobStatus.SUCCESS, JobStatus.PARTIAL, JobStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for API responses"""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "total_files": self.total_files,
            "processed_files": self.processed_files,
            "result": self.result,
            "error_message": self.error_message,
            "download_url": self.download_url
        }


class JobStore:
    """
    Thread-safe in-memory storage for background jobs.
    
    Provides CRUD operations for managing job lifecycle and status tracking.
    Includes automatic cleanup of old completed jobs.
    """
    
    def __init__(self, max_completed_jobs: int = 1000):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()
        self.max_completed_jobs = max_completed_jobs
        logger.info(f"JobStore initialized with max_completed_jobs={max_completed_jobs}")

    def create_job(self, total_files: int = 1, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new job and return its ID
        
        Args:
            total_files: Total number of files to process
            metadata: Additional job metadata
            
        Returns:
            str: Generated job ID
        """
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        
        with self._lock:
            job = Job(
                job_id=job_id,
                total_files=total_files,
                metadata=metadata or {}
            )
            self._jobs[job_id] = job
            
        logger.info(f"Created job {job_id} for {total_files} files")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job object or None if not found
        """
        with self._lock:
            return self._jobs.get(job_id)

    def update_job_status(self, job_id: str, status: JobStatus) -> bool:
        """
        Update job status
        
        Args:
            job_id: Job identifier
            status: New status
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = status
                if status == JobStatus.RUNNING and not job.started_at:
                    job.mark_started()
                return True
            return False

    def update_job_progress(self, job_id: str, processed: int, total: Optional[int] = None) -> bool:
        """
        Update job progress
        
        Args:
            job_id: Job identifier
            processed: Number of files processed
            total: Total files (optional, updates total if provided)
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.update_progress(processed, total)
                return True
            return False

    def complete_job(self, job_id: str, 
                    result: Union[ProcessingResultSchema, List[ProcessingResultSchema]],
                    status: JobStatus = JobStatus.SUCCESS,
                    download_url: Optional[str] = None) -> bool:
        """
        Mark job as completed with result
        
        Args:
            job_id: Job identifier
            result: Processing result(s)
            status: Final status (success, partial, failed)
            download_url: Optional download URL for results
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.mark_completed(result, status)
                if download_url:
                    job.download_url = download_url
                
                # Trigger cleanup if we have too many completed jobs
                self._cleanup_old_jobs()
                return True
            return False

    def fail_job(self, job_id: str, error_message: str) -> bool:
        """
        Mark job as failed
        
        Args:
            job_id: Job identifier
            error_message: Error description
            
        Returns:
            bool: True if updated successfully
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.mark_failed(error_message)
                return True
            return False

    def list_jobs(self, status_filter: Optional[JobStatus] = None, 
                  limit: int = 50, offset: int = 0) -> List[Job]:
        """
        List jobs with optional filtering and pagination
        
        Args:
            status_filter: Filter by job status
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            
        Returns:
            List of Job objects
        """
        with self._lock:
            jobs = list(self._jobs.values())
            
            # Filter by status if requested
            if status_filter:
                jobs = [job for job in jobs if job.status == status_filter]
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda j: j.created_at, reverse=True)
            
            # Apply pagination
            return jobs[offset:offset + limit]

    def get_job_stats(self) -> Dict[str, int]:
        """
        Get job statistics by status
        
        Returns:
            Dictionary with counts by status
        """
        with self._lock:
            stats = {status.value: 0 for status in JobStatus}
            
            for job in self._jobs.values():
                stats[job.status.value] += 1
                
            return stats

    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """
        Remove completed jobs older than specified age
        
        Args:
            max_age_hours: Maximum age in hours for completed jobs
            
        Returns:
            Number of jobs removed
        """
        from datetime import timedelta
        
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        with self._lock:
            jobs_to_remove = []
            
            for job_id, job in self._jobs.items():
                if (job.is_completed() and 
                    job.completed_at and 
                    job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)
            
            for job_id in jobs_to_remove:
                del self._jobs[job_id]
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old completed jobs")
        
        return removed_count

    def _cleanup_old_jobs(self):
        """Internal method to cleanup old jobs when limit is exceeded"""
        if len(self._jobs) <= self.max_completed_jobs:
            return
        
        # Get completed jobs sorted by completion time
        completed_jobs = [
            (job_id, job) for job_id, job in self._jobs.items() 
            if job.is_completed() and job.completed_at
        ]
        completed_jobs.sort(key=lambda x: x[1].completed_at)
        
        # Remove oldest jobs to get under the limit
        jobs_to_remove = len(self._jobs) - self.max_completed_jobs
        for i in range(min(jobs_to_remove, len(completed_jobs))):
            job_id = completed_jobs[i][0]
            del self._jobs[job_id]
        
        logger.info(f"Auto-cleaned {min(jobs_to_remove, len(completed_jobs))} old jobs")

    def clear_all_jobs(self):
        """Clear all jobs (for testing/admin purposes)"""
        with self._lock:
            count = len(self._jobs)
            self._jobs.clear()
            logger.warning(f"Cleared all {count} jobs from store")


# Global job store instance
_job_store: Optional[JobStore] = None
_store_lock = threading.Lock()


def get_job_store() -> JobStore:
    """
    Get the global job store instance (singleton pattern)
    
    Returns:
        JobStore: Global job store instance
    """
    global _job_store
    
    if _job_store is None:
        with _store_lock:
            if _job_store is None:
                _job_store = JobStore()
    
    return _job_store


def reset_job_store():
    """Reset the global job store (for testing)"""
    global _job_store
    with _store_lock:
        _job_store = None
