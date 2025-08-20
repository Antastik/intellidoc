#!/usr/bin/env python3
"""
IntelliDoc CLI

Simple command-line interface for the IntelliDoc document processing pipeline.
Designed for ease of use with minimal steps required.
"""

import sys
import click
from pathlib import Path
import json
from typing import Optional
from loguru import logger

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from pipeline import IntelliDocPipeline
except ImportError as e:
    print(f"Error importing pipeline: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    if verbose:
        logger.add(sys.stderr, level="DEBUG", format="{time} | {level} | {message}")
    else:
        logger.add(sys.stderr, level="INFO", format="{level} | {message}")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, verbose):
    """
    IntelliDoc - Intelligent Document Processing Pipeline
    
    Process documents with OCR and NLP to extract structured information.
    """
    setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('file_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output filename (optional)')
@click.option('--no-save', is_flag=True, help='Do not save results to file')
@click.pass_context
def process(ctx, file_path: str, output: Optional[str], no_save: bool):
    """
    Process a single document through the OCR + NLP pipeline.
    
    FILE_PATH: Path to the document to process
    
    Examples:
        intellidoc process document.pdf
        intellidoc process image.png --output result.json
        intellidoc process document.docx --no-save
    """
    try:
        click.echo(f"🔄 Processing document: {Path(file_path).name}")
        
        # Initialize pipeline
        pipeline = IntelliDocPipeline()
        
        # Process document
        result = pipeline.process_single_document(
            file_path=file_path,
            save_output=not no_save,
            output_filename=output
        )
        
        # Display results
        click.echo(f"✅ Processing completed in {result.processing_time:.2f}s")
        click.echo(f"📊 Status: {result.status}")
        
        if result.combined_text:
            click.echo(f"📝 Extracted text: {len(result.combined_text)} characters")
        
        if result.final_entities:
            click.echo(f"🏷️  Extracted entities: {len(result.final_entities)}")
            
            # Show top entities
            entity_types = {}
            for entity in result.final_entities:
                label = entity['label']
                entity_types[label] = entity_types.get(label, 0) + 1
            
            for label, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                click.echo(f"   - {label}: {count}")
        
        if result.errors:
            click.echo(f"⚠️  Errors: {len(result.errors)}")
            for error in result.errors:
                click.echo(f"   - {error}")
        
        # Show sample entities
        if result.final_entities and not ctx.obj.get('verbose'):
            click.echo("\n🔍 Sample entities:")
            for entity in result.final_entities[:5]:
                click.echo(f"   {entity['label']}: {entity['text']} (conf: {entity['confidence']:.2f})")
            
            if len(result.final_entities) > 5:
                click.echo(f"   ... and {len(result.final_entities) - 5} more")
        
        return result
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('directory_path', type=click.Path(exists=True, file_okay=False))
@click.option('--recursive/--no-recursive', default=True, help='Process subdirectories recursively')
@click.option('--parallel/--sequential', default=True, help='Use parallel processing')
@click.option('--workers', '-w', type=int, help='Number of parallel workers')
@click.pass_context
def batch(ctx, directory_path: str, recursive: bool, parallel: bool, workers: Optional[int]):
    """
    Process all supported documents in a directory.
    
    DIRECTORY_PATH: Path to directory containing documents
    
    Examples:
        intellidoc batch ./documents
        intellidoc batch ./invoices --no-recursive
        intellidoc batch ./docs --sequential --workers 2
    """
    try:
        directory = Path(directory_path)
        click.echo(f"📁 Processing directory: {directory.name}")
        click.echo(f"🔍 Recursive: {recursive}")
        click.echo(f"⚡ Parallel: {parallel}")
        
        # Initialize pipeline
        pipeline = IntelliDocPipeline()
        
        # Override workers if specified
        if workers:
            pipeline.max_workers = workers
            click.echo(f"👷 Workers: {workers}")
        
        # Process directory
        results = pipeline.process_directory(
            directory_path=directory_path,
            recursive=recursive,
            parallel=parallel
        )
        
        if not results:
            click.echo("⚠️  No supported documents found")
            return
        
        # Display summary
        successful = len([r for r in results if r.status == 'success'])
        partial = len([r for r in results if r.status == 'partial'])
        failed = len([r for r in results if r.status == 'failed'])
        total_time = sum(r.processing_time for r in results)
        
        click.echo(f"\n📊 Batch Processing Summary:")
        click.echo(f"   Total documents: {len(results)}")
        click.echo(f"   ✅ Successful: {successful}")
        click.echo(f"   ⚠️  Partial: {partial}")
        click.echo(f"   ❌ Failed: {failed}")
        click.echo(f"   ⏱️  Total time: {total_time:.2f}s")
        click.echo(f"   📊 Average time: {total_time/len(results):.2f}s per document")
        
        # Show individual results if verbose
        if ctx.obj.get('verbose'):
            click.echo("\n📄 Individual Results:")
            for result in results:
                status_icon = {"success": "✅", "partial": "⚠️", "failed": "❌"}[result.status]
                file_name = result.document_metadata.get('file_name', 'unknown')
                entities_count = len(result.final_entities)
                click.echo(f"   {status_icon} {file_name}: {entities_count} entities ({result.processing_time:.2f}s)")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def status():
    """
    Check the status of pipeline components.
    
    Shows which OCR engines and NLP models are available.
    """
    try:
        click.echo("🔧 IntelliDoc System Status")
        
        # Initialize pipeline
        pipeline = IntelliDocPipeline()
        status_info = pipeline.get_system_status()
        
        # Document ingestion
        doc_status = status_info['document_ingestor']
        click.echo(f"\n📄 Document Ingestion: {'✅' if doc_status['available'] else '❌'}")
        if doc_status['available']:
            formats = ', '.join(doc_status['supported_formats'])
            click.echo(f"   Supported formats: {formats}")
        
        # OCR processing
        ocr_status = status_info['ocr_processor']
        click.echo(f"\n👁️  OCR Processing: {'✅' if ocr_status['available'] else '❌'}")
        if ocr_status['available']:
            for engine, available in ocr_status['engines'].items():
                icon = '✅' if available else '❌'
                click.echo(f"   {engine}: {icon}")
        
        # NLP processing
        nlp_status = status_info['nlp_processor']
        click.echo(f"\n🧠 NLP Processing: {'✅' if nlp_status['available'] else '❌'}")
        if nlp_status['available']:
            for component, available in nlp_status['components'].items():
                icon = '✅' if available else '❌'
                click.echo(f"   {component}: {icon}")
        
        # Pipeline settings
        pipeline_info = status_info['pipeline']
        click.echo(f"\n⚙️  Pipeline Configuration:")
        click.echo(f"   Max workers: {pipeline_info['max_workers']}")
        click.echo(f"   Timeout: {pipeline_info['timeout_seconds']}s")
        click.echo(f"   Output format: {pipeline_info['output_format']}")
        
        return status_info
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--count', '-c', default=1, help='Number of sample documents to create')
def demo(count: int):
    """
    Create sample documents for testing the pipeline.
    
    Creates sample documents in the data/input directory.
    """
    try:
        click.echo(f"🎯 Creating {count} sample document(s)")
        
        input_dir = Path("data/input")
        input_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(count):
            # Create sample document content
            sample_content = f"""
INVOICE #{1000 + i}

Date: January {15 + i}, 2024
Invoice ID: INV-2024-{1000 + i}

Bill To:
John Doe {i+1}
Email: john.doe{i+1}@example.com
Phone: (555) {123 + i:03d}-{4567 + i:04d}

From:
Acme Corporation
123 Business Street
New York, NY 10001
Phone: (555) 123-0000

Description                    Amount
Consulting Services           ${2500 + (i * 100)}.00
Software License             ${1000 + (i * 50)}.00
Support Package              ${500 + (i * 25)}.00

Total: ${4000 + (i * 175)}.00

Payment Terms: Net 30 days
Thank you for your business!
            """.strip()
            
            # Save sample document
            filename = f"sample_invoice_{i+1}.txt"
            file_path = input_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(sample_content)
            
            click.echo(f"   📄 Created: {filename}")
        
        click.echo(f"\n✅ Sample documents created in {input_dir}")
        click.echo("💡 Try: intellidoc batch data/input")
        
    except Exception as e:
        logger.error(f"Demo creation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--setup-dev', is_flag=True, help='Setup development environment')
def setup(setup_dev: bool):
    """
    Setup IntelliDoc environment and check dependencies.
    
    Verifies installation and provides setup guidance.
    """
    try:
        click.echo("🚀 IntelliDoc Setup")
        
        # Create directories
        directories = ['data/input', 'data/output', 'models', 'temp']
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            click.echo(f"   📁 Created directory: {dir_path}")
        
        # Check dependencies
        click.echo("\n🔍 Checking dependencies...")
        
        try:
            import torch
            click.echo("   ✅ PyTorch available")
        except ImportError:
            click.echo("   ❌ PyTorch not found - install with: pip install torch")
        
        try:
            import transformers
            click.echo("   ✅ Transformers available")
        except ImportError:
            click.echo("   ❌ Transformers not found - install with: pip install transformers")
        
        try:
            import spacy
            click.echo("   ✅ spaCy available")
            # Check for English model
            try:
                import spacy
                nlp = spacy.load("en_core_web_sm")
                click.echo("   ✅ spaCy English model available")
            except OSError:
                click.echo("   ⚠️  spaCy English model not found - install with: python -m spacy download en_core_web_sm")
        except ImportError:
            click.echo("   ❌ spaCy not found - install with: pip install spacy")
        
        # OCR engines
        try:
            import pytesseract
            click.echo("   ✅ Tesseract available")
        except ImportError:
            click.echo("   ❌ Tesseract not found - install with: pip install pytesseract")
        
        try:
            from paddleocr import PaddleOCR
            click.echo("   ✅ PaddleOCR available")
        except ImportError:
            click.echo("   ⚠️  PaddleOCR not found - install with: pip install paddleocr")
        
        # Test pipeline
        click.echo("\n🧪 Testing pipeline initialization...")
        try:
            pipeline = IntelliDocPipeline()
            click.echo("   ✅ Pipeline initialization successful")
        except Exception as e:
            click.echo(f"   ⚠️  Pipeline initialization failed: {e}")
        
        click.echo("\n✅ Setup completed!")
        click.echo("💡 Try: intellidoc demo --count 3 && intellidoc batch data/input")
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
