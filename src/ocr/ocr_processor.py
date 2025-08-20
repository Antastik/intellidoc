"""
OCR Processing Module

Handles text extraction from images using multiple OCR engines with fallback support.
Supports Tesseract and PaddleOCR with confidence scoring and error handling.
"""

import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
import yaml
from loguru import logger

# OCR engine imports with error handling
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available")


@dataclass
class OCRResult:
    """Container for OCR processing results"""
    text: str
    confidence: float
    engine: str
    processing_time: float
    word_boxes: Optional[List[Dict]] = None  # Word-level bounding boxes and confidence
    metadata: Optional[Dict[str, Any]] = None


class OCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    @abstractmethod
    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extract text from an image"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the OCR engine is available"""
        pass


class TesseractOCR(OCREngine):
    """Tesseract OCR engine implementation"""
    
    def __init__(self, config: str = "--oem 3 --psm 6", languages: List[str] = None):
        self.config = config
        self.languages = languages or ["eng"]
        self.lang_string = "+".join(self.languages)
        
        if not self.is_available():
            raise RuntimeError("Tesseract is not available")
        
        logger.info(f"TesseractOCR initialized with languages: {self.languages}")
    
    def is_available(self) -> bool:
        """Check if Tesseract is available"""
        return TESSERACT_AVAILABLE
    
    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extract text using Tesseract OCR"""
        start_time = time.time()
        
        try:
            # Extract text with confidence data
            data = pytesseract.image_to_data(
                image, 
                config=self.config,
                lang=self.lang_string,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract plain text
            text = pytesseract.image_to_string(
                image,
                config=self.config,
                lang=self.lang_string
            ).strip()
            
            # Calculate average confidence
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Extract word-level information
            word_boxes = []
            for i, word in enumerate(data['text']):
                if word.strip() and int(data['conf'][i]) > 0:
                    word_boxes.append({
                        'text': word,
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'left': data['left'][i],
                            'top': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=text,
                confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                engine="tesseract",
                processing_time=processing_time,
                word_boxes=word_boxes,
                metadata={
                    'languages': self.languages,
                    'config': self.config,
                    'word_count': len([w for w in data['text'] if w.strip()])
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Tesseract OCR failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                engine="tesseract",
                processing_time=processing_time,
                metadata={'error': str(e)}
            )


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine implementation"""
    
    def __init__(self, use_angle_cls: bool = True, lang: str = "en", use_gpu: bool = False):
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        
        if not self.is_available():
            raise RuntimeError("PaddleOCR is not available")
        
        # Initialize PaddleOCR
        self.ocr = PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False
        )
        
        logger.info(f"PaddleOCR initialized with language: {lang}, GPU: {use_gpu}")
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is available"""
        return PADDLEOCR_AVAILABLE
    
    def extract_text(self, image: Image.Image) -> OCRResult:
        """Extract text using PaddleOCR"""
        start_time = time.time()
        
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            # Run OCR
            results = self.ocr.ocr(img_array, cls=self.use_angle_cls)
            
            # Process results
            text_lines = []
            word_boxes = []
            confidences = []
            
            if results and results[0]:
                for result in results[0]:
                    if result:
                        bbox, (text, confidence) = result
                        text_lines.append(text)
                        confidences.append(confidence)
                        
                        # Convert bbox to word box format
                        word_boxes.append({
                            'text': text,
                            'confidence': confidence * 100,  # Convert to 0-100 scale
                            'bbox': {
                                'coordinates': bbox,
                                'left': min([point[0] for point in bbox]),
                                'top': min([point[1] for point in bbox]),
                                'width': max([point[0] for point in bbox]) - min([point[0] for point in bbox]),
                                'height': max([point[1] for point in bbox]) - min([point[1] for point in bbox])
                            }
                        })
            
            # Combine text and calculate average confidence
            full_text = "\n".join(text_lines)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                engine="paddleocr",
                processing_time=processing_time,
                word_boxes=word_boxes,
                metadata={
                    'language': self.lang,
                    'use_gpu': self.use_gpu,
                    'line_count': len(text_lines)
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PaddleOCR failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                engine="paddleocr",
                processing_time=processing_time,
                metadata={'error': str(e)}
            )


class OCRProcessor:
    """
    Main OCR processor with multiple engine support and fallback mechanisms.
    Designed for scalability and reliability.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.engines = {}
        self.engine_order = self.config["ocr"]["engines"]
        self.min_confidence_threshold = 0.5  # Minimum confidence for results
        
        self._initialize_engines()
        
        if not self.engines:
            raise RuntimeError("No OCR engines are available")
        
        logger.info(f"OCRProcessor initialized with engines: {list(self.engines.keys())}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                "ocr": {
                    "engines": ["tesseract", "paddleocr"],
                    "tesseract": {
                        "config": "--oem 3 --psm 6",
                        "languages": ["eng"]
                    },
                    "paddleocr": {
                        "use_angle_cls": True,
                        "lang": "en",
                        "use_gpu": False
                    }
                }
            }
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        ocr_config = self.config["ocr"]
        
        # Initialize Tesseract if available
        if "tesseract" in self.engine_order and TESSERACT_AVAILABLE:
            try:
                tesseract_config = ocr_config.get("tesseract", {})
                self.engines["tesseract"] = TesseractOCR(
                    config=tesseract_config.get("config", "--oem 3 --psm 6"),
                    languages=tesseract_config.get("languages", ["eng"])
                )
                logger.info("Tesseract OCR engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Tesseract: {e}")
        
        # Initialize PaddleOCR if available
        if "paddleocr" in self.engine_order and PADDLEOCR_AVAILABLE:
            try:
                paddle_config = ocr_config.get("paddleocr", {})
                self.engines["paddleocr"] = PaddleOCREngine(
                    use_angle_cls=paddle_config.get("use_angle_cls", True),
                    lang=paddle_config.get("lang", "en"),
                    use_gpu=paddle_config.get("use_gpu", False)
                )
                logger.info("PaddleOCR engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
    
    def process_image(self, image: Image.Image, use_fallback: bool = True) -> OCRResult:
        """
        Process a single image with OCR
        
        Args:
            image: PIL Image to process
            use_fallback: Whether to use fallback engines if primary fails
            
        Returns:
            OCRResult: Best OCR result from available engines
        """
        results = []
        
        # Try engines in order
        for engine_name in self.engine_order:
            if engine_name not in self.engines:
                continue
            
            logger.info(f"Processing image with {engine_name}")
            result = self.engines[engine_name].extract_text(image)
            results.append(result)
            
            # If we got a good result, use it (unless fallback is disabled)
            if result.confidence >= self.min_confidence_threshold:
                logger.info(f"{engine_name} produced good result (confidence: {result.confidence:.2f})")
                if not use_fallback:
                    return result
            
            # If fallback is disabled and this is the first engine, return regardless
            if not use_fallback:
                return result
        
        # Select best result based on confidence
        if results:
            best_result = max(results, key=lambda r: r.confidence)
            logger.info(f"Best OCR result from {best_result.engine} (confidence: {best_result.confidence:.2f})")
            return best_result
        
        # No engines worked
        logger.error("All OCR engines failed")
        return OCRResult(
            text="",
            confidence=0.0,
            engine="none",
            processing_time=0.0,
            metadata={'error': 'All OCR engines failed'}
        )
    
    def process_batch(self, images: List[Image.Image]) -> List[OCRResult]:
        """
        Process multiple images in batch
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List[OCRResult]: OCR results for each image
        """
        logger.info(f"Processing batch of {len(images)} images")
        results = []
        
        for i, image in enumerate(images, 1):
            logger.info(f"Processing image {i}/{len(images)}")
            result = self.process_image(image)
            results.append(result)
        
        return results
    
    def get_engine_status(self) -> Dict[str, bool]:
        """Get status of all OCR engines"""
        return {
            'tesseract': TESSERACT_AVAILABLE,
            'paddleocr': PADDLEOCR_AVAILABLE
        }


def preprocess_image(image: Image.Image, enhance: bool = True) -> Image.Image:
    """
    Preprocess image for better OCR results
    
    Args:
        image: Input PIL Image
        enhance: Whether to apply enhancement filters
        
    Returns:
        Image.Image: Preprocessed image
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if enhance:
            # Basic enhancement can be added here
            # For now, just ensure proper format
            pass
        
        return image
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return image


if __name__ == "__main__":
    # Test the OCR processor
    processor = OCRProcessor()
    
    # Show engine status
    status = processor.get_engine_status()
    print("OCR Engine Status:")
    for engine, available in status.items():
        print(f"  {engine}: {'✓' if available else '✗'}")
    
    # Create a simple test image with text
    from PIL import Image, ImageDraw, ImageFont
    
    # Create test image
    img = Image.new('RGB', (400, 100), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a system font
        font = ImageFont.load_default()
    except:
        font = None
    
    draw.text((10, 30), "Test Document - OCR Processing", fill='black', font=font)
    
    # Test OCR
    result = processor.process_image(img)
    print(f"\nOCR Result:")
    print(f"Engine: {result.engine}")
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Processing time: {result.processing_time:.3f}s")
