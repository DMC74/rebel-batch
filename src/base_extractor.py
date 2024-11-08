# base_extractor.py
import os
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from transformers import pipeline, AutoTokenizer
import logging
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

class BaseRelationExtractor:
    """Base class for relation extraction containing shared functionality."""
    
    def __init__(self, 
                 model_name: str = 'Babelscape/mrebel-large-32', 
                 device: int = 0,
                 max_length: int = 200):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.safe_length = int(0.9 * max_length)
        
        # Set seed for consistent language detection
        DetectorFactory.seed = 0
        
        # Setup logging
        self._setup_logging()
        self._initialize_nltk()
        self._initialize_model()
        
        # Initialize language mappings
        self._initialize_language_mappings()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _initialize_nltk(self):
        """Initialize NLTK components."""
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            self.logger.warning(f"Failed to download NLTK data: {str(e)}")

    def _initialize_model(self):
        """Initialize the transformer model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.extractor = pipeline(
                'translation_xx_to_yy',
                model=self.model_name,
                tokenizer=self.model_name,
                device=self.device if isinstance(self.device, str) else 'cpu'
            )
            self.logger.info(f"Successfully loaded model {self.model_name} on device {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def _initialize_language_mappings(self):
        """Initialize language code mappings."""
        self.supported_languages = {
            'ar': 'ar_AR', 'cs': 'cs_CZ', 'de': 'de_DE', 'en': 'en_XX', 
            'es': 'es_XX', 'fr': 'fr_XX', 'hi': 'hi_IN', 'it': 'it_IT',
            'ja': 'ja_XX', 'ko': 'ko_KR', 'nl': 'nl_XX', 'pl': 'pl_PL',
            'pt': 'pt_XX', 'ru': 'ru_RU', 'tr': 'tr_TR', 'vi': 'vi_VN',
            'zh': 'zh_CN'
        }
        
        self.lang_mapping = {
            'zh-cn': 'zh', 'zh-tw': 'zh', 'zh_cn': 'zh', 'zh_tw': 'zh',
            'zh-hans': 'zh', 'zh-hant': 'zh', 'en-us': 'en', 'en-gb': 'en',
            'pt-br': 'pt', 'pt-pt': 'pt', 'es-es': 'es', 'es-mx': 'es',
            'fr-fr': 'fr', 'fr-ca': 'fr', 'de-de': 'de', 'de-at': 'de',
            'de-ch': 'de', 'ru-ru': 'ru', 'it-it': 'it', 'ja-jp': 'ja',
            'ko-kr': 'ko', 'ar-sa': 'ar', 'ar-ae': 'ar', 'nl-nl': 'nl',
            'nl-be': 'nl', 'pl-pl': 'pl', 'tr-tr': 'tr', 'vi-vn': 'vi'
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of the input text."""
        try:
            sample_text = text[:1000]
            detected_lang = detect(sample_text)
            normalized_lang = self.lang_mapping.get(detected_lang.lower(), detected_lang.split('-')[0])
            
            if normalized_lang in self.supported_languages:
                model_lang_code = self.supported_languages[normalized_lang]
                self.logger.info(
                    f"Detected language: {detected_lang} -> normalized: {normalized_lang} "
                    f"-> model code: {model_lang_code}"
                )
                return model_lang_code
            else:
                self.logger.warning(
                    f"Detected language '{detected_lang}' not supported. Falling back to English."
                )
                return 'en_XX'
                
        except LangDetectException as e:
            self.logger.error(f"Language detection failed: {str(e)}. Falling back to English.")
            return 'en_XX'
        except Exception as e:
            self.logger.error(f"Unexpected error in language detection: {str(e)}. Falling back to English.")
            return 'en_XX'

    def get_token_length(self, text: str) -> int:
        """Get the actual token length of a text string."""
        return len(self.tokenizer.encode(text))

    def segment_document(self, text: str) -> List[str]:
        """Segment document into chunks that respect the token limit."""
        segments = []
        sentences = sent_tokenize(text)
        current_segment = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.get_token_length(sentence)
            
            if sentence_tokens > self.safe_length:
                if current_segment:
                    segments.append(' '.join(current_segment))
                    current_segment = []
                    current_length = 0
                
                words = sentence.split()
                temp_segment = []
                temp_length = 0
                
                for word in words:
                    word_tokens = self.get_token_length(word + ' ')
                    if temp_length + word_tokens > self.safe_length:
                        if temp_segment:
                            segments.append(' '.join(temp_segment))
                        temp_segment = [word]
                        temp_length = word_tokens
                    else:
                        temp_segment.append(word)
                        temp_length += word_tokens
                
                if temp_segment:
                    segments.append(' '.join(temp_segment))
            
            elif current_length + sentence_tokens > self.safe_length:
                if current_segment:
                    segments.append(' '.join(current_segment))
                current_segment = [sentence]
                current_length = sentence_tokens
            else:
                current_segment.append(sentence)
                current_length += sentence_tokens
        
        if current_segment:
            segments.append(' '.join(current_segment))
        
        return segments

    def process_segment(self, segment: str, src_lang: str) -> List[Dict[str, str]]:
        """Process a single document segment."""
        try:
            if self.get_token_length(segment) > self.safe_length:
                self.logger.warning(f"Segment too long: {len(segment)} chars")
                return []
            
            result = self.extractor(
                segment,
                decoder_start_token_id=250058,
                src_lang=src_lang,
                tgt_lang="<triplet>",
                max_length=self.max_length,
                return_tensors=True,
                return_text=False
            )
            
            if not result or not isinstance(result, list) or len(result) == 0:
                return []
            
            translation_ids = result[0].get("translation_token_ids")
            if translation_ids is None:
                return []
            
            decoded_text = self.extractor.tokenizer.batch_decode([translation_ids])
            if not decoded_text:
                return []
            
            return self.extract_triplets_typed(decoded_text[0])
            
        except Exception as e:
            self.logger.error(f"Error processing segment: {str(e)}")
            return []

    def extract_triplets_typed(self, text: str) -> List[Dict[str, str]]:
        """Parse the generated text and extract typed triplets."""
        if not text:
            return []
            
        triplets = []
        text = text.strip()
        current = 'x'
        subject, relation, object_, object_type, subject_type = '', '', '', '', ''

        try:
            tokens = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "") \
                        .replace("tp_XX", "").replace("__en__", "").split()

            for token in tokens:
                if token == "<triplet>" or token == "<relation>":
                    current = 't'
                    if relation != '':
                        triplet = {
                            'head': subject.strip(),
                            'head_type': subject_type,
                            'type': relation.strip(),
                            'tail': object_.strip(),
                            'tail_type': object_type
                        }
                        if all(triplet.values()):
                            triplets.append(triplet)
                        relation = ''
                    subject = ''
                elif token.startswith("<") and token.endswith(">"):
                    if current == 't' or current == 'o':
                        current = 's'
                        if relation != '':
                            triplet = {
                                'head': subject.strip(),
                                'head_type': subject_type,
                                'type': relation.strip(),
                                'tail': object_.strip(),
                                'tail_type': object_type
                            }
                            if all(triplet.values()):
                                triplets.append(triplet)
                        object_ = ''
                        subject_type = token[1:-1]
                    else:
                        current = 'o'
                        object_type = token[1:-1]
                        relation = ''
                else:
                    if current == 't':
                        subject += ' ' + token
                    elif current == 's':
                        object_ += ' ' + token
                    elif current == 'o':
                        relation += ' ' + token

            if all(x != '' for x in [subject, relation, object_, object_type, subject_type]):
                triplets.append({
                    'head': subject.strip(),
                    'head_type': subject_type,
                    'type': relation.strip(),
                    'tail': object_.strip(),
                    'tail_type': object_type
                })
        except Exception as e:
            self.logger.error(f"Error parsing triplets: {str(e)}")
            return []

        return triplets

