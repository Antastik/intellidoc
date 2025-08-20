"""
Service layer for integrating IntelliDoc pipeline with FastAPI.

Provides async wrappers around the synchronous pipeline processing
with job tracking, background task management, and error handling.
"""

import os
import tempfile
import asyncio
from pathlib import Path
from typing import List, Optional, Union, Any, Dict
from fastapi import UploadFile, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from loguru import logger

# Import pipeline components
import sys
pipeline_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(pipeline_path))

from pipeline import IntelliDocPipeline, PipelineResult
from utils.document_ingestion import DocumentIngestionError

# Import API components
from ..jobs.job_store import get_job_store, JobStatus
from ..schemas.pipeline import ProcessingResultSchema, EntitySchema, DocumentMetadataSchema


class PipelineService:
    """
    Service layer for document processing pipeline.
    
    Provides async wrappers around the synchronous IntelliDocPipeline
    with background job management and result tracking.
    """
    
    def __init__(self):
        self._pipeline: Optional[IntelliDocPipeline] = None
        self.upload_dir = Path("data/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.job_store = get_job_store()
        
    @property
    def pipeline(self) -> IntelliDocPipeline:
        """Lazy initialization of the pipeline"""
        if self._pipeline is None:
            self._pipeline = IntelliDocPipeline()
            logger.info("IntelliDocPipeline initialized")
        return self._pipeline
    
    async def save_upload_file(self, upload_file: UploadFile) -> str:
        """
        Save uploaded file to temporary location
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            str: Path to saved file
        """
        # Create unique filename
        file_extension = Path(upload_file.filename or "").suffix
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension,
            dir=self.upload_dir
        )
        
        try:
            # Read and save file content
            content = await upload_file.read()
            temp_file.write(content)
            temp_file.flush()
            
            logger.info(f"Saved upload file: {upload_file.filename} -> {temp_file.name}")
            return temp_file.name
            
        finally:
            temp_file.close()
    
    def _convert_pipeline_result_to_schema(self, result: PipelineResult) -> ProcessingResultSchema:
        """Convert internal PipelineResult to API schema"""
        # Convert entities
        entities = []
        for entity_dict in result.final_entities:
            entities.append(EntitySchema(
                text=entity_dict.get('text', ''),
                label=entity_dict.get('label', ''),
                start=entity_dict.get('start', 0),
                end=entity_dict.get('end', 0),
                confidence=entity_dict.get('confidence', 0.0),
                metadata=entity_dict.get('metadata')
            ))
        
        # Convert document metadata
        doc_metadata = None
        if result.document_metadata:
            doc_metadata = DocumentMetadataSchema(
                file_name=result.document_metadata.get('file_name', ''),
                file_size=result.document_metadata.get('file_size', 0),
                mime_type=result.document_metadata.get('mime_type', ''),
                format=result.document_metadata.get('format', ''),
                pages=result.document_metadata.get('pages', 1),
                created_at=result.document_metadata.get('created_at'),
                modified_at=result.document_metadata.get('modified_at')
            )
        
        # Map status
        status_mapping = {
            'success': JobStatus.SUCCESS,
            'partial': JobStatus.PARTIAL,
            'failed': JobStatus.FAILED
        }
        
        return ProcessingResultSchema(
            document_id=result.document_id,
            status=status_mapping.get(result.status, JobStatus.FAILED),
            combined_text=result.combined_text,
            entities=entities,
            processing_time=result.processing_time,
            errors=result.errors,
            document_metadata=doc_metadata
        )
    
    async def process_document_sync(
        self,
        upload_file: UploadFile,
        save_result: bool = True
    ) -> ProcessingResultSchema:
        """
        Process single document synchronously
        
        Args:
            upload_file: Uploaded file
            save_result: Whether to save processing result
            
        Returns:
            ProcessingResultSchema: Processing result
        """
        file_path = None
        try:
            # Save uploaded file
            file_path = await self.save_upload_file(upload_file)
            
            # Process document in thread pool
            result = await run_in_threadpool(
                self.pipeline.process_single_document,
                file_path=file_path,
                save_output=save_result
            )
            
            # Convert to API schema
            return self._convert_pipeline_result_to_schema(result)
            
        finally:
            # Cleanup temporary file
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
    
    async def process_document_async(
        self,
        upload_file: UploadFile,
        background_tasks: BackgroundTasks,
        save_result: bool = True
    ) -> str:
        """
        Process single document asynchronously
        
        Args:
            upload_file: Uploaded file
            background_tasks: FastAPI background tasks
            save_result: Whether to save processing result
            
        Returns:
            str: Job ID for tracking
        """
        # Create job
        job_id = self.job_store.create_job(
            total_files=1,
            metadata={
                "filename": upload_file.filename,
                "content_type": upload_file.content_type,
                "save_result": save_result
            }
        )
        
        # Save uploaded file
        file_path = await self.save_upload_file(upload_file)
        
        # Schedule background processing
        background_tasks.add_task(
            self._process_document_background,
            job_id=job_id,
            file_path=file_path,
            save_result=save_result
        )
        
        return job_id
    
    def _process_document_background(
        self,
        job_id: str,
        file_path: str,
        save_result: bool = True
    ):
        """
        Background task for document processing
        
        Args:
            job_id: Job identifier
            file_path: Path to document file
            save_result: Whether to save processing result
        """
        try:
            # Mark job as started
            self.job_store.update_job_status(job_id, JobStatus.RUNNING)
            
            # Process document
            result = self.pipeline.process_single_document(
                file_path=file_path,
                save_output=save_result
            )
            
            # Convert to API schema
            api_result = self._convert_pipeline_result_to_schema(result)
            
            # Determine final status
            final_status = JobStatus.SUCCESS
            if result.errors:
                final_status = JobStatus.PARTIAL if result.combined_text else JobStatus.FAILED
            
            # Create download URL if result was saved
            download_url = None
            if save_result and hasattr(result, 'output_path'):
                download_url = f"/api/v1/jobs/{job_id}/download"
            
            # Mark job as completed
            self.job_store.complete_job(
                job_id=job_id,
                result=api_result,
                status=final_status,
                download_url=download_url
            )
            
        except Exception as e:
            logger.error(f"Background processing failed for job {job_id}: {e}")
            self.job_store.fail_job(job_id, str(e))
            
        finally:
            # Cleanup temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)
    
    async def process_batch_sync(
        self,
        upload_files: List[UploadFile],
        parallel: bool = True,
        save_individual: bool = True,
        save_batch_summary: bool = True
    ) -> List[ProcessingResultSchema]:
        """
        Process multiple documents synchronously
        
        Args:
            upload_files: List of uploaded files
            parallel: Whether to use parallel processing
            save_individual: Whether to save individual results
            save_batch_summary: Whether to save batch summary
            
        Returns:
            List[ProcessingResultSchema]: Processing results
        """
        file_paths = []
        try:
            # Save all uploaded files
            for upload_file in upload_files:
                file_path = await self.save_upload_file(upload_file)
                file_paths.append(file_path)
            
            # Process batch in thread pool
            results = await run_in_threadpool(
                self.pipeline.process_batch,
                file_paths=file_paths,
                parallel=parallel,
                save_individual=save_individual,
                save_batch_summary=save_batch_summary
            )
            
            # Convert to API schemas
            api_results = []
            for result in results:
                api_result = self._convert_pipeline_result_to_schema(result)
                api_results.append(api_result)
            
            return api_results
            
        finally:
            # Cleanup temporary files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
    async def process_batch_async(
        self,
        upload_files: List[UploadFile],
        background_tasks: BackgroundTasks,
        parallel: bool = True,
        save_individual: bool = True,
        save_batch_summary: bool = True
    ) -> str:
        """
        Process multiple documents asynchronously
        
        Args:
            upload_files: List of uploaded files
            background_tasks: FastAPI background tasks
            parallel: Whether to use parallel processing
            save_individual: Whether to save individual results
            save_batch_summary: Whether to save batch summary
            
        Returns:
            str: Job ID for tracking
        """
        # Create job
        job_id = self.job_store.create_job(
            total_files=len(upload_files),
            metadata={
                "filenames": [f.filename for f in upload_files],
                "parallel": parallel,
                "save_individual": save_individual,
                "save_batch_summary": save_batch_summary
            }
        )
        
        # Save all uploaded files
        file_paths = []
        for upload_file in upload_files:
            file_path = await self.save_upload_file(upload_file)
            file_paths.append(file_path)
        
        # Schedule background processing
        background_tasks.add_task(
            self._process_batch_background,
            job_id=job_id,
            file_paths=file_paths,
            parallel=parallel,
            save_individual=save_individual,
            save_batch_summary=save_batch_summary
        )
        
        return job_id
    
    def _process_batch_background(
        self,
        job_id: str,
        file_paths: List[str],
        parallel: bool = True,
        save_individual: bool = True,
        save_batch_summary: bool = True
    ):
        """
        Background task for batch processing
        
        Args:
            job_id: Job identifier
            file_paths: List of file paths
            parallel: Whether to use parallel processing
            save_individual: Whether to save individual results
            save_batch_summary: Whether to save batch summary
        """
        try:
            # Mark job as started
            self.job_store.update_job_status(job_id, JobStatus.RUNNING)
            
            # Process batch
            results = self.pipeline.process_batch(
                file_paths=file_paths,
                parallel=parallel,
                save_individual=save_individual,
                save_batch_summary=save_batch_summary
            )
            
            # Convert to API schemas
            api_results = []
            successful_count = 0
            partial_count = 0
            
            for result in results:
                api_result = self._convert_pipeline_result_to_schema(result)
                api_results.append(api_result)
                
                if result.status == 'success':
                    successful_count += 1
                elif result.status == 'partial':
                    partial_count += 1
            
            # Determine final status
            if successful_count == len(results):
                final_status = JobStatus.SUCCESS
            elif successful_count + partial_count > 0:
                final_status = JobStatus.PARTIAL
            else:
                final_status = JobStatus.FAILED
            
            # Create download URL if batch summary was saved
            download_url = None
            if save_batch_summary:
                download_url = f"/api/v1/jobs/{job_id}/download"
            
            # Mark job as completed
            self.job_store.complete_job(
                job_id=job_id,
                result=api_results,
                status=final_status,
                download_url=download_url
            )
            
        except Exception as e:
            logger.error(f"Background batch processing failed for job {job_id}: {e}")
            self.job_store.fail_job(job_id, str(e))
            
        finally:
            # Cleanup temporary files
            for file_path in file_paths:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status from pipeline"""
        try:
            status = self.pipeline.get_system_status()
            
            # Add job queue statistics
            job_stats = self.job_store.get_job_stats()
            status['queue_status'] = job_stats
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                'error': str(e),
                'queue_status': self.job_store.get_job_stats()
            }
    
    def validate_upload_file(self, upload_file: UploadFile) -> Optional[str]:
        """
        Validate uploaded file
        
        Args:
            upload_file: FastAPI UploadFile object
            
        Returns:
            str: Error message if validation fails, None if valid
        """
        if not upload_file.filename:
            return "Filename is required"
        
        # Check file extension
        file_ext = Path(upload_file.filename).suffix.lower().lstrip('.')
        supported_formats = self.pipeline.document_ingestor.supported_formats
        
        if file_ext not in supported_formats:
            return f"Unsupported file format: {file_ext}. Supported: {', '.join(supported_formats)}"
        
        # Check file size (if available)
        if hasattr(upload_file, 'size') and upload_file.size:
            max_size = self.pipeline.document_ingestor.max_file_size
            if upload_file.size > max_size:
                return f"File too large: {upload_file.size} bytes (max: {max_size})"
        
        return None


# Global service instance
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get the global pipeline service instance (singleton)"""
    global _pipeline_service
    
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    
    return _pipeline_service
