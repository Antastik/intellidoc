"""
NLP Processing Module

Handles text analysis including entity extraction, document classification,
and structured output generation for the IntelliDoc pipeline.
"""

import time
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import yaml
from loguru import logger

# NLP library imports with error handling
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not available")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available")

try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available")


@dataclass
class Entity:
    """Extracted entity information"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Classification:
    """Document classification result"""
    label: str
    confidence: float
    categories: Optional[Dict[str, float]] = None


@dataclass
class NLPResult:
    """Complete NLP processing result"""
    original_text: str
    cleaned_text: str
    entities: List[Entity]
    classification: Optional[Classification]
    summary: Optional[str]
    keywords: List[str]
    processing_time: float
    metadata: Dict[str, Any]


class TextProcessor:
    """Text preprocessing and cleaning utilities"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char.isspace())
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def extract_email_addresses(text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text, re.IGNORECASE)
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Extract phone numbers from text"""
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # US format with parentheses
            r'\b\d{10,15}\b'  # Generic number
        ]
        
        phone_numbers = []
        for pattern in phone_patterns:
            phone_numbers.extend(re.findall(pattern, text))
        
        return list(set(phone_numbers))  # Remove duplicates
    
    @staticmethod
    def extract_currency_amounts(text: str) -> List[str]:
        """Extract currency amounts from text"""
        currency_pattern = r'[\$€£¥]\s?[\d,]+\.?\d*|\b\d+\.?\d*\s?(?:dollars?|euros?|pounds?|yen)\b'
        return re.findall(currency_pattern, text, re.IGNORECASE)


class SpacyEntityExtractor:
    """Entity extraction using spaCy"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise RuntimeError("spaCy is not available")
        
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"spaCy model {model_name} not found, using blank model")
            self.nlp = spacy.blank("en")
    
    def extract_entities(self, text: str, entity_types: Optional[Set[str]] = None) -> List[Entity]:
        """Extract named entities from text"""
        if not text:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            if entity_types is None or ent.label_ in entity_types:
                entities.append(Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=1.0,  # spaCy doesn't provide confidence scores by default
                    metadata={'spacy_label': ent.label_}
                ))
        
        return entities


class TransformersClassifier:
    """Document classification using Transformers"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library is not available")
        
        self.model_name = model_name
        try:
            # Initialize classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True
            )
            logger.info(f"Loaded Transformers model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.classifier = None
    
    def classify_document(self, text: str, top_k: int = 5) -> Optional[Classification]:
        """Classify document content"""
        if not self.classifier or not text:
            return None
        
        try:
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            results = self.classifier(text)
            
            if results:
                # Get top result
                top_result = results[0][0] if results[0] else None
                if top_result:
                    # Build categories dictionary
                    categories = {item['label']: item['score'] for item in results[0][:top_k]}
                    
                    return Classification(
                        label=top_result['label'],
                        confidence=top_result['score'],
                        categories=categories
                    )
        except Exception as e:
            logger.error(f"Classification failed: {e}")
        
        return None


class NLPProcessor:
    """
    Main NLP processor that coordinates all text analysis tasks.
    Designed for scalability and modularity.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = self._load_config(config_path)
        self.entity_types = set(self.config["nlp"]["entity_types"])
        
        # Initialize components
        self.text_processor = TextProcessor()
        self.entity_extractor = None
        self.classifier = None
        
        self._initialize_components()
        
        logger.info("NLPProcessor initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return {
                "nlp": {
                    "models": {
                        "classification": "distilbert-base-uncased",
                        "ner": "en_core_web_sm"
                    },
                    "entity_types": ["PERSON", "ORG", "MONEY", "DATE", "GPE", "PRODUCT"]
                }
            }
    
    def _initialize_components(self):
        """Initialize NLP components"""
        nlp_config = self.config["nlp"]["models"]
        
        # Initialize entity extractor
        if SPACY_AVAILABLE:
            try:
                ner_model = nlp_config.get("ner", "en_core_web_sm")
                self.entity_extractor = SpacyEntityExtractor(ner_model)
                logger.info("Entity extractor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize entity extractor: {e}")
        
        # Initialize classifier
        if TRANSFORMERS_AVAILABLE:
            try:
                classification_model = nlp_config.get("classification", "distilbert-base-uncased")
                self.classifier = TransformersClassifier(classification_model)
                logger.info("Document classifier initialized")
            except Exception as e:
                logger.error(f"Failed to initialize classifier: {e}")
    
    def extract_custom_entities(self, text: str) -> List[Entity]:
        """Extract custom entities using regex patterns"""
        entities = []
        
        # Extract emails
        emails = self.text_processor.extract_email_addresses(text)
        for email in emails:
            start = text.find(email)
            if start != -1:
                entities.append(Entity(
                    text=email,
                    label="EMAIL",
                    start=start,
                    end=start + len(email),
                    confidence=0.95,
                    metadata={'extraction_method': 'regex'}
                ))
        
        # Extract phone numbers
        phones = self.text_processor.extract_phone_numbers(text)
        for phone in phones:
            start = text.find(phone)
            if start != -1:
                entities.append(Entity(
                    text=phone,
                    label="PHONE",
                    start=start,
                    end=start + len(phone),
                    confidence=0.9,
                    metadata={'extraction_method': 'regex'}
                ))
        
        # Extract currency amounts
        amounts = self.text_processor.extract_currency_amounts(text)
        for amount in amounts:
            start = text.find(amount)
            if start != -1:
                entities.append(Entity(
                    text=amount,
                    label="MONEY",
                    start=start,
                    end=start + len(amount),
                    confidence=0.9,
                    metadata={'extraction_method': 'regex'}
                ))
        
        return entities
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        if not text:
            return []
        
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these', 
            'those', 'his', 'her', 'its', 'their', 'our', 'your', 'was', 'were', 
            'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may', 
            'might', 'must', 'can', 'are', 'is', 'was', 'were', 'being', 'been'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in keywords[:max_keywords]]
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> Optional[str]:
        """Generate a simple extractive summary"""
        if not text:
            return None
        
        # Simple extractive summary - take first few sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= max_sentences:
            return text
        
        # Return first few sentences
        summary_sentences = sentences[:max_sentences]
        return '. '.join(summary_sentences) + '.'
    
    def process_text(self, text: str) -> NLPResult:
        """
        Process text through the complete NLP pipeline
        
        Args:
            text: Input text to process
            
        Returns:
            NLPResult: Complete analysis results
        """
        start_time = time.time()
        
        # Clean text
        cleaned_text = self.text_processor.clean_text(text)
        
        # Extract entities
        entities = []
        
        # Use spaCy for standard NER if available
        if self.entity_extractor:
            spacy_entities = self.entity_extractor.extract_entities(cleaned_text, self.entity_types)
            entities.extend(spacy_entities)
        
        # Add custom entity extraction
        custom_entities = self.extract_custom_entities(cleaned_text)
        entities.extend(custom_entities)
        
        # Remove duplicate entities (simple deduplication)
        unique_entities = []
        seen_entities = set()
        for entity in entities:
            entity_key = (entity.text.lower(), entity.label)
            if entity_key not in seen_entities:
                unique_entities.append(entity)
                seen_entities.add(entity_key)
        
        # Classify document
        classification = None
        if self.classifier:
            classification = self.classifier.classify_document(cleaned_text)
        
        # Generate summary
        summary = self.generate_summary(cleaned_text)
        
        # Extract keywords
        keywords = self.extract_keywords(cleaned_text)
        
        processing_time = time.time() - start_time
        
        # Create metadata
        metadata = {
            'text_length': len(cleaned_text),
            'entity_count': len(unique_entities),
            'keyword_count': len(keywords),
            'has_classification': classification is not None,
            'processing_time': processing_time
        }
        
        return NLPResult(
            original_text=text,
            cleaned_text=cleaned_text,
            entities=unique_entities,
            classification=classification,
            summary=summary,
            keywords=keywords,
            processing_time=processing_time,
            metadata=metadata
        )
    
    def process_batch(self, texts: List[str]) -> List[NLPResult]:
        """Process multiple texts in batch"""
        logger.info(f"Processing batch of {len(texts)} texts")
        results = []
        
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{len(texts)}")
            result = self.process_text(text)
            results.append(result)
        
        return results
    
    def export_results(self, result: NLPResult, format: str = "json") -> Dict[str, Any]:
        """Export NLP results in specified format"""
        if format.lower() == "json":
            # Convert dataclasses to dict
            result_dict = asdict(result)
            return result_dict
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of NLP components"""
        return {
            'spacy': SPACY_AVAILABLE and self.entity_extractor is not None,
            'transformers': TRANSFORMERS_AVAILABLE and self.classifier is not None,
            'nltk': NLTK_AVAILABLE,
            'text_processor': True
        }


if __name__ == "__main__":
    # Test the NLP processor
    processor = NLPProcessor()
    
    # Show component status
    status = processor.get_component_status()
    print("NLP Component Status:")
    for component, available in status.items():
        print(f"  {component}: {'✓' if available else '✗'}")
    
    # Test text
    sample_text = """
    Invoice #12345
    Date: January 15, 2024
    
    From: Acme Corporation
    123 Business Street
    New York, NY 10001
    
    To: John Doe
    Email: john.doe@email.com
    Phone: (555) 123-4567
    
    Services rendered:
    - Consulting services: $2,500.00
    - Software license: $1,000.00
    
    Total Amount: $3,500.00
    
    Payment due within 30 days.
    """
    
    # Process the text
    result = processor.process_text(sample_text)
    
    print(f"\nNLP Processing Results:")
    print(f"Text length: {len(result.cleaned_text)} characters")
    print(f"Processing time: {result.processing_time:.3f}s")
    
    print(f"\nExtracted Entities ({len(result.entities)}):")
    for entity in result.entities:
        print(f"  {entity.label}: {entity.text} (confidence: {entity.confidence:.2f})")
    
    if result.classification:
        print(f"\nClassification:")
        print(f"  Label: {result.classification.label}")
        print(f"  Confidence: {result.classification.confidence:.3f}")
    
    print(f"\nKeywords: {', '.join(result.keywords[:5])}")
    
    if result.summary:
        print(f"\nSummary: {result.summary[:100]}...")
    
    # Export results
    exported = processor.export_results(result)
    print(f"\nExported result contains {len(exported)} fields")
