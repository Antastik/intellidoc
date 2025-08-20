"""
IntelliDoc Pipeline Orchestrator

Main pipeline that coordinates document ingestion, OCR processing, and NLP analysis.
Designed for scalability, reliability, and ease of use.
"""

import time
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Generator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import yaml
from loguru import logger

# Import our custom modules
from utils.document_ingestion import DocumentIngestor, ProcessedDocument, DocumentIngestionError
from ocr.ocr_processor import OCRProcessor, OCRResult
from nlp.nlp_processor import NLPProcessor, NLPResult


@dataclass
class PipelineResult:
    """Complete pipeline processing result"""
    document_id: str
    document_metadata: Dict[str, Any]
    ocr_results: List[OCRResult]
    nlp_results: List[NLPResult]
    combined_text: str
    final_entities: List[Dict[str, Any]]
    processing_time: float
    status: str  # 'success', 'partial', 'failed'
    errors: List[str]


class IntelliDocPipeline:
    """
    Main pipeline orchestrator for the IntelliDoc system.
    
    Coordinates document ingestion, OCR processing, and NLP analysis
    with support for batch processing and parallel execution.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.document_ingestor = DocumentIngestor(config_path)
        self.ocr_processor = None
        self.nlp_processor = None
        
        # Pipeline settings
        self.max_workers = self.config["pipeline"]["max_workers"]
        self.timeout_seconds = self.config["pipeline"]["timeout_seconds"]
        self.retry_attempts = self.config["pipeline"]["retry_attempts"]
        
        # Output settings
        self.output_format = self.config["output"]["format"]
        self.include_confidence = self.config["output"]["include_confidence"]
        self.include_metadata = self.config["output"]["include_metadata"]
        
        # Initialize processors
        self._initialize_processors()
        
        # Setup paths
        self.output_dir = Path(self.config["paths"]["output"])
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"IntelliDocPipeline initialized with {self.max_workers} workers")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return minimal default config
            return {
                "pipeline": {"max_workers": 2, "timeout_seconds": 300, "retry_attempts": 2},
                "output": {"format": "json", "include_confidence": True, "include_metadata": True},
                "paths": {"output": "data/output"}
            }
    
    def _initialize_processors(self):
        """Initialize OCR and NLP processors"""
        try:
            self.ocr_processor = OCRProcessor()
            logger.info("OCR processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OCR processor: {e}")
            self.ocr_processor = None
        
        try:
            self.nlp_processor = NLPProcessor()
            logger.info("NLP processor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize NLP processor: {e}")
            self.nlp_processor = None
        
        if not self.ocr_processor and not self.nlp_processor:
            raise RuntimeError("Both OCR and NLP processors failed to initialize")
    
    def process_single_document(self, file_path: str, 
                              save_output: bool = True,
                              output_filename: Optional[str] = None) -> PipelineResult:
        """
        Process a single document through the complete pipeline
        
        Args:
            file_path: Path to the document to process
            save_output: Whether to save results to file
            output_filename: Custom output filename
            
        Returns:
            PipelineResult: Complete processing results
        """
        start_time = time.time()
        document_id = str(uuid.uuid4())
        errors = []
        
        logger.info(f"Processing document: {Path(file_path).name}")
        
        try:
            # Step 1: Document Ingestion
            logger.info("Step 1: Document ingestion")
            processed_doc = self.document_ingestor.ingest_single_document(file_path)
            
            # Step 2: OCR Processing
            logger.info("Step 2: OCR processing")
            ocr_results = []
            
            if self.ocr_processor and hasattr(processed_doc.content, '__iter__') and not isinstance(processed_doc.content, str):
                # Content is images (list of PIL Images)
                ocr_results = self.ocr_processor.process_batch(processed_doc.content)
            elif isinstance(processed_doc.content, str):
                # Content is already text (e.g., from DOCX)
                from ocr.ocr_processor import OCRResult
                ocr_results = [OCRResult(
                    text=processed_doc.content,
                    confidence=1.0,
                    engine="direct_text",
                    processing_time=0.0,
                    metadata={'source': 'direct_extraction'}
                )]
            else:
                errors.append("Invalid content type for OCR processing")
                logger.error("Invalid content type for OCR processing")
            
            # Step 3: Combine OCR texts
            combined_text = ""
            if ocr_results:
                combined_text = "\n".join([result.text for result in ocr_results if result.text])
            
            # Step 4: NLP Processing
            logger.info("Step 3: NLP processing")
            nlp_results = []
            
            if self.nlp_processor and combined_text:
                # Split text into chunks if too long
                text_chunks = self._split_text(combined_text, max_length=5000)
                nlp_results = self.nlp_processor.process_batch(text_chunks)
            else:
                if not combined_text:
                    errors.append("No text extracted for NLP processing")
                if not self.nlp_processor:
                    errors.append("NLP processor not available")
            
            # Step 5: Combine and deduplicate entities
            final_entities = self._combine_entities(nlp_results)
            
            # Step 6: Create result
            processing_time = time.time() - start_time
            status = "success" if not errors else ("partial" if ocr_results or nlp_results else "failed")
            
            result = PipelineResult(
                document_id=document_id,
                document_metadata=asdict(processed_doc.metadata),
                ocr_results=ocr_results,
                nlp_results=nlp_results,
                combined_text=combined_text,
                final_entities=final_entities,
                processing_time=processing_time,
                status=status,
                errors=errors
            )
            
            # Step 7: Save output if requested
            if save_output:
                output_path = self._save_result(result, output_filename)
                logger.info(f"Results saved to: {output_path}")
            
            logger.info(f"Document processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Pipeline processing failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            
            return PipelineResult(
                document_id=document_id,
                document_metadata={},
                ocr_results=[],
                nlp_results=[],
                combined_text="",
                final_entities=[],
                processing_time=processing_time,
                status="failed",
                errors=errors
            )
    
    def process_batch(self, file_paths: List[str],
                     parallel: bool = True,
                     save_individual: bool = True,
                     save_batch_summary: bool = True) -> List[PipelineResult]:
        """
        Process multiple documents in batch
        
        Args:
            file_paths: List of file paths to process
            parallel: Whether to use parallel processing
            save_individual: Whether to save individual results
            save_batch_summary: Whether to save batch summary
            
        Returns:
            List[PipelineResult]: Results for each document
        """
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        start_time = time.time()
        
        results = []
        
        if parallel and len(file_paths) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_path = {
                    executor.submit(self.process_single_document, path, save_individual): path
                    for path in file_paths
                }
                
                for future in as_completed(future_to_path):
                    file_path = future_to_path[future]
                    try:
                        result = future.result(timeout=self.timeout_seconds)
                        results.append(result)
                        logger.info(f"Completed: {Path(file_path).name}")
                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        # Create failed result
                        results.append(PipelineResult(
                            document_id=str(uuid.uuid4()),
                            document_metadata={'file_path': file_path},
                            ocr_results=[],
                            nlp_results=[],
                            combined_text="",
                            final_entities=[],
                            processing_time=0.0,
                            status="failed",
                            errors=[str(e)]
                        ))
        else:
            # Sequential processing
            for file_path in file_paths:
                result = self.process_single_document(file_path, save_individual)
                results.append(result)
        
        # Save batch summary
        if save_batch_summary:
            summary_path = self._save_batch_summary(results, len(file_paths))
            logger.info(f"Batch summary saved to: {summary_path}")
        
        total_time = time.time() - start_time
        successful = len([r for r in results if r.status == "success"])
        logger.info(f"Batch processing completed: {successful}/{len(file_paths)} successful in {total_time:.2f}s")
        
        return results
    
    def process_directory(self, directory_path: str,
                         recursive: bool = True,
                         **kwargs) -> List[PipelineResult]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            **kwargs: Additional arguments for batch processing
            
        Returns:
            List[PipelineResult]: Results for each document
        """
        logger.info(f"Processing directory: {directory_path}")
        
        # Find all supported files
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Use the document ingestor to find files
        supported_formats = self.document_ingestor.supported_formats
        pattern = "**/*" if recursive else "*"
        file_paths = []
        
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_ext = file_path.suffix.lower().lstrip('.')
                if file_ext in supported_formats:
                    file_paths.append(str(file_path))
        
        if not file_paths:
            logger.warning(f"No supported documents found in {directory_path}")
            return []
        
        logger.info(f"Found {len(file_paths)} supported documents")
        return self.process_batch(file_paths, **kwargs)
    
    def _split_text(self, text: str, max_length: int = 5000, overlap: int = 200) -> List[str]:
        """Split long text into overlapping chunks"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            
            # Try to find a sentence boundary
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + max_length // 2:
                    end = sentence_end + 1
            
            chunks.append(text[start:end])
            start = max(start + max_length - overlap, end)
        
        return chunks
    
    def _combine_entities(self, nlp_results: List[NLPResult]) -> List[Dict[str, Any]]:
        """Combine and deduplicate entities from multiple NLP results"""
        all_entities = []
        
        for nlp_result in nlp_results:
            for entity in nlp_result.entities:
                entity_dict = asdict(entity)
                all_entities.append(entity_dict)
        
        # Simple deduplication based on text and label
        unique_entities = []
        seen = set()
        
        for entity in all_entities:
            key = (entity['text'].lower(), entity['label'])
            if key not in seen:
                unique_entities.append(entity)
                seen.add(key)
        
        return unique_entities
    
    def _save_result(self, result: PipelineResult, filename: Optional[str] = None) -> str:
        """Save pipeline result to file"""
        if filename is None:
            filename = f"result_{result.document_id}.json"
        
        output_path = self.output_dir / filename
        
        # Convert result to dictionary
        result_dict = asdict(result)
        
        # Remove non-serializable fields if needed
        if not self.include_metadata:
            result_dict.pop('document_metadata', None)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
        
        return str(output_path)
    
    def _save_batch_summary(self, results: List[PipelineResult], total_files: int) -> str:
        """Save batch processing summary"""
        summary = {
            'batch_id': str(uuid.uuid4()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_files': total_files,
            'processed_files': len(results),
            'successful': len([r for r in results if r.status == 'success']),
            'partial': len([r for r in results if r.status == 'partial']),
            'failed': len([r for r in results if r.status == 'failed']),
            'total_processing_time': sum(r.processing_time for r in results),
            'average_processing_time': sum(r.processing_time for r in results) / len(results) if results else 0,
            'results': [
                {
                    'document_id': r.document_id,
                    'file_name': r.document_metadata.get('file_name', 'unknown'),
                    'status': r.status,
                    'processing_time': r.processing_time,
                    'entity_count': len(r.final_entities),
                    'text_length': len(r.combined_text),
                    'errors': r.errors
                }
                for r in results
            ]
        }
        
        summary_path = self.output_dir / f"batch_summary_{summary['batch_id']}.json"
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return str(summary_path)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components"""
        return {
            'document_ingestor': {
                'available': True,
                'supported_formats': self.document_ingestor.supported_formats
            },
            'ocr_processor': {
                'available': self.ocr_processor is not None,
                'engines': self.ocr_processor.get_engine_status() if self.ocr_processor else {}
            },
            'nlp_processor': {
                'available': self.nlp_processor is not None,
                'components': self.nlp_processor.get_component_status() if self.nlp_processor else {}
            },
            'pipeline': {
                'max_workers': self.max_workers,
                'timeout_seconds': self.timeout_seconds,
                'output_format': self.output_format
            }
        }


if __name__ == "__main__":
    # Test the pipeline
    pipeline = IntelliDocPipeline()
    
    # Show system status
    status = pipeline.get_system_status()
    print("Pipeline System Status:")
    print(f"  Document Ingestion: {'✓' if status['document_ingestor']['available'] else '✗'}")
    print(f"  OCR Processing: {'✓' if status['ocr_processor']['available'] else '✗'}")
    print(f"  NLP Processing: {'✓' if status['nlp_processor']['available'] else '✗'}")
    print(f"  Max Workers: {status['pipeline']['max_workers']}")
    
    # Test with sample directory if it exists
    input_dir = Path("data/input")
    if input_dir.exists():
        print(f"\nProcessing documents in {input_dir}...")
        results = pipeline.process_directory(str(input_dir))
        print(f"Processed {len(results)} documents")
        
        for result in results:
            print(f"  {result.document_metadata.get('file_name', 'unknown')}: {result.status}")
    else:
        print(f"\nInput directory {input_dir} not found. Create it and add documents to test the pipeline.")
