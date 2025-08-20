# IntelliDoc Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Download spaCy English model (optional but recommended)
python -m spacy download en_core_web_sm
```

### 2. Setup Environment
```bash
# Initialize directories and check dependencies
python intellidoc.py setup
```

### 3. Process Documents
```bash
# Create sample documents for testing
python intellidoc.py demo --count 3

# Process all documents in a directory
python intellidoc.py batch data/input

# Process a single document
python intellidoc.py process data/input/sample_invoice_1.txt
```

## 📋 Usage Examples

### Check System Status
```bash
python intellidoc.py status
```
Shows which OCR engines and NLP models are available.

### Process Single Document
```bash
# Basic processing
python intellidoc.py process document.pdf

# Custom output filename
python intellidoc.py process invoice.png --output my_result.json

# Don't save to file (just show results)
python intellidoc.py process contract.docx --no-save
```

### Batch Processing
```bash
# Process all documents in directory
python intellidoc.py batch ./documents

# Non-recursive (current directory only)
python intellidoc.py batch ./invoices --no-recursive

# Sequential processing (no parallelization)
python intellidoc.py batch ./docs --sequential

# Custom number of workers
python intellidoc.py batch ./files --workers 8

# Verbose output
python intellidoc.py -v batch ./documents
```

## 📁 File Structure

```
intellidoc/
├── intellidoc.py          # Main CLI interface
├── requirements.txt       # Python dependencies
├── config/config.yaml     # Configuration settings
├── src/                   # Source code
│   ├── pipeline.py        # Main pipeline orchestrator
│   ├── ocr/              # OCR processing
│   ├── nlp/              # NLP processing
│   └── utils/            # Document ingestion
├── data/
│   ├── input/            # Input documents
│   └── output/           # Processing results
└── models/               # ML model storage
```

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- **OCR engines**: Choose between Tesseract and PaddleOCR
- **NLP models**: Configure entity types and classification models
- **Processing**: Set worker count, timeouts, batch sizes
- **Output**: Control format and metadata inclusion

## 📊 Output Format

Results are saved as JSON files with:

```json
{
  "document_id": "unique-id",
  "status": "success",
  "combined_text": "extracted text...",
  "final_entities": [
    {
      "text": "John Doe",
      "label": "PERSON",
      "confidence": 0.95
    }
  ],
  "processing_time": 2.34,
  "metadata": { ... }
}
```

## 🎯 Performance Targets

- **Accuracy**: 95%+ entity extraction accuracy
- **Throughput**: 100+ documents per hour
- **Scalability**: Configurable parallel processing
- **Reliability**: Automatic fallback between OCR engines

## 🔍 Supported Formats

- **Documents**: PDF, DOCX
- **Images**: PNG, JPG, JPEG, TIFF
- **Text**: Direct text extraction from DOCX files

## 🏷️ Extracted Entities

- **PERSON**: Names of people
- **ORG**: Organizations and companies
- **MONEY**: Currency amounts ($1,000.00)
- **DATE**: Dates and times
- **EMAIL**: Email addresses
- **PHONE**: Phone numbers
- **GPE**: Geographical locations

## ⚙️ Advanced Usage

### Custom Configuration
```bash
# Use custom config file
export INTELLIDOC_CONFIG=/path/to/custom/config.yaml
python intellidoc.py process document.pdf
```

### Programmatic Usage
```python
from src.pipeline import IntelliDocPipeline

# Initialize pipeline
pipeline = IntelliDocPipeline()

# Process single document
result = pipeline.process_single_document("document.pdf")

# Process directory
results = pipeline.process_directory("./documents")
```

## 🐛 Troubleshooting

### Common Issues

1. **OCR engines not available**
   ```bash
   pip install pytesseract paddleocr
   ```

2. **spaCy model missing**
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Memory issues with large documents**
   - Reduce `max_workers` in config
   - Increase `timeout_seconds`
   - Process files sequentially

### Check Dependencies
```bash
python intellidoc.py setup
```

### Verbose Logging
```bash
python intellidoc.py -v process document.pdf
```

## 🚀 Next Steps

1. **Install additional OCR engines** for better accuracy
2. **Fine-tune NLP models** for domain-specific entities
3. **Scale to cloud deployment** using Docker and AWS
4. **Add monitoring and metrics** for production use

---

For more details, see the full README.md and code documentation.
